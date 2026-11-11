"""Temporary FULL-KV ownership independent of the GDN resume checkpoint.

An agent turn can match more FULL KV than its latest reusable GDN snapshot.
Native prefill recomputes that gap and deduplicates it at commit. Keep those
canonical bytes through that commit while allowing unrelated branches to evict.
"""

from dataclasses import dataclass


@dataclass
class FullPrefixPin:
    cache: object
    nodes: tuple
    matched_tokens: int
    hold_until_finish: bool = False
    released: bool = False

    def release(self):
        if self.released:
            return False
        # split_at copies full refcounts to new prefix ancestors. Walking the
        # CURRENT parent chain therefore releases exactly this pin after splits.
        for node in self.nodes:
            current = node
            while not current.is_root():
                if current.ref_count <= 0:
                    raise RuntimeError("FULL-only prefix pin lost its ownership")
                current.ref_count -= 1
                if current.ref_count == 0:
                    self.cache.full_protected -= current.length
                    self.cache.full_evictable += current.length
                current = current.parent
        self.released = True
        return True


@dataclass(frozen=True)
class FullPrefixDecision:
    pin: FullPrefixPin | None
    purge_allowed: bool
    matched_tokens: int
    extra_tokens: int
    slack_tokens: int
    fallback_reason: str | None = None


def _match_full_read_only(cache, input_ids):
    """Return (anchor, matched length, optional prefix length inside anchor).

    Unlike _walk this does not update timestamps or split nodes. Capacity/GDN
    fallback must preserve the native eviction candidates and their LRU order.
    """
    node = cache.root
    matched = 0
    while matched < len(input_ids):
        child = node.children.get(cache.key_fn(input_ids[matched:]))
        if child is None:
            return node, matched, None
        part = child.get_match_len(input_ids[matched:])
        part = part // cache.page_size * cache.page_size
        matched += part
        if part != child.length:
            return child, matched, part
        node = child
    return node, matched, None


def reserve_full_prefix(
    cache, input_ids, *, reused_tokens, required_tokens, available_tokens, allow_pinning=True
):
    """Caller already holds the normal matched GDN handle lock.

    available_tokens is the manager's current logical available capacity AFTER
    that lock; required_tokens is the SAME input/output reservation used by the
    scheduler, including earlier admissions in this scheduling pass. If the
    whole extra FULL prefix cannot fit, do not pin or aggressively purge: let
    native allocation/eviction preserve its original maximum-context behavior.
    """
    node, matched, partial = _match_full_read_only(cache, input_ids)
    hold_until_finish = int(matched) == len(input_ids) and int(matched) > 0
    nodes = [node]
    if hold_until_finish:
        # No observed divergence: generated continuation may match any already
        # resident descendant. Keep the complete related subtree, never guess
        # future tokens or partially pin it.
        nodes = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current.is_leaf():
                nodes.append(current)
            else:
                stack.extend(current.children.values())
    # Price only currently unprotected nodes, once even for shared ancestors.
    seen = set()
    extra = 0
    for leaf in nodes:
        current = leaf
        while not current.is_root() and id(current) not in seen:
            seen.add(id(current))
            length = (
                partial
                if (current is node and partial is not None and not hold_until_finish)
                else current.length
            )
            if current.ref_count == 0:
                extra += length
            current = current.parent
    slack = max(0, int(available_tokens) - int(required_tokens))
    if extra > slack:
        return FullPrefixDecision(None, False, int(matched), extra, slack, "full_capacity")
    needs_pin = hold_until_finish or int(matched) > int(reused_tokens)
    if needs_pin and not allow_pinning:
        return FullPrefixDecision(None, False, int(matched), extra, slack, "gdn_pressure")
    if not needs_pin:
        return FullPrefixDecision(None, True, int(matched), 0, slack)
    if partial is not None and not hold_until_finish:
        # A successful known-divergence reservation needs only the matching
        # portion. split_at preserves old timestamps and inherited refcounts.
        node = node.split_at(partial)
        nodes = [node]
    for leaf in nodes:
        current = leaf
        while not current.is_root():
            if current.ref_count == 0:
                cache.full_evictable -= current.length
                cache.full_protected += current.length
            current.ref_count += 1
            current = current.parent
    # Deliberately no mamba_ref_count/mamba_{protected,evictable} mutation.
    pin = FullPrefixPin(cache, tuple(nodes), int(matched), hold_until_finish)
    return FullPrefixDecision(pin, True, int(matched), extra, slack)
