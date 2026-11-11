"""Request leases for immutable read-only metadata of sealed partial graphs.

Model parameters/topology and scalar configuration are immutable during generate.
Full native signatures run at first use and again before generate returns.
Dynamic tensor layouts/state storage and per-step math flags remain checked.
Concurrent mutation/reversion of model metadata is outside this lease contract.
No graph, kernel, pager, residency or tensor value is changed here.
"""

from contextlib import contextmanager


class RequestGraphValidation:
    def __init__(self, manager, engine):
        attention = getattr(manager, "attention_decode_graph", None)
        front = getattr(manager, "ffn_front_graph", None)
        if getattr(manager, "request_graph_validation", None) is not None:
            raise RuntimeError("Request graph validation already installed")
        if any(
            c is None
            or c.manager is not manager
            or c.engine is not engine
            or c.state != "sealed"
            or c.step_active
            for c in (attention, front)
        ):
            raise ValueError("Idle sealed attention/front graph controllers required")
        self.manager, self.engine = manager, engine
        self.attention, self.front = attention, front
        self.state, self.failure, self.active = "installed", None, False
        self.frames = {}
        self.bindings = []
        self.counts = dict(
            requests=0,
            decode_steps=0,
            full_attention_signatures=0,
            full_front_signatures=0,
            cached_attention_signatures=0,
            cached_front_signatures=0,
            dynamic_checks=0,
            post_validations=0,
        )
        self.native_a = attention._signature
        self.native_f = front._signature
        globals_a = self.native_a.__func__.__globals__
        globals_f = self.native_f.__func__.__globals__
        self.layout_a, self.storage = globals_a["_layout"], globals_a["_storage"]
        self.layout_f = globals_f["_layout"]
        self._bind(
            attention,
            "_signature",
            lambda module, x, states: self.signature("attention", module, x, states),
        )
        self._bind(
            front, "_signature", lambda module, x, ids: self.signature("front", module, x, ids)
        )
        original = attention.begin_decode

        def begin(batch):
            self._owner()
            if not self.active:
                raise RuntimeError("Decode requires an immutable request lease")
            if attention._math_flags() != self.flags:
                raise RuntimeError("Math flags changed within the request lease")
            self.counts["decode_steps"] += 1
            return original(batch)

        self._bind(attention, "begin_decode", begin)
        manager.request_graph_validation = self

    def _bind(self, owner, name, hook):
        self.bindings.append((owner, name, hook, vars(owner).get(name)))
        setattr(owner, name, hook)

    def _owner(self):
        if (
            self.state != "installed"
            or getattr(self.manager, "request_graph_validation", None) is not self
        ):
            raise RuntimeError("Request graph lease owner is unavailable")
        if (
            self.manager.attention_decode_graph is not self.attention
            or self.manager.ffn_front_graph is not self.front
        ):
            raise RuntimeError("Installed graph controllers changed")
        if any(getattr(owner, name) is not hook for owner, name, hook, _ in self.bindings):
            raise RuntimeError("Graph validation bindings changed")

    def dynamic(self, kind, x, argument):
        if kind == "attention":
            return self.layout_a(x), tuple(self.storage(t) for t in argument)
        return self.layout_f(x), self.layout_f(argument)

    def signature(self, kind, module, x, argument):
        self._owner()
        if not self.active:
            raise RuntimeError("Static signatures require an immutable request lease")
        key = kind, id(module)
        dynamic = self.dynamic(kind, x, argument)
        self.counts["dynamic_checks"] += 1
        native = self.native_a if kind == "attention" else self.native_f
        if key not in self.frames:
            result = native(module, x, argument)
            self.counts["full_" + kind + "_signatures"] += 1
            self.frames[key] = (module, x, argument, dynamic, result)
        else:
            if self.frames[key][3] != dynamic:
                raise RuntimeError("Dynamic graph input/state storage changed inside a request")
            self.counts["cached_" + kind + "_signatures"] += 1
        return self.frames[key][4]

    @contextmanager
    def request(self):
        self._owner()
        if self.active:
            raise RuntimeError("Nested or concurrent request graph lease")
        if any(c.state != "sealed" or c.step_active for c in (self.attention, self.front)):
            raise RuntimeError("Request graph controllers are not idle and sealed")
        self.flags = self.attention._math_flags()
        self.frames = {}
        self.active = True
        try:
            yield self
            # Validation is inside the timed generate call. A persistent
            # metadata mutation invalidates the request before it returns.
            for (kind, _), (module, x, argument, _, cached) in self.frames.items():
                native = self.native_a if kind == "attention" else self.native_f
                if kind == "attention":
                    argument = self.attention._state_tensors(module)
                result = native(module, x, argument)
                self.counts["full_" + kind + "_signatures"] += 1
                if result[0] != cached[0]:
                    raise RuntimeError("Immutable graph metadata changed during generate")
                self.counts["post_validations"] += 1
            self.counts["requests"] += 1
        except BaseException as error:
            self.state, self.failure = "failed", repr(error)
            raise
        finally:
            self.frames.clear()
            self.active = False

    def snapshot(self):
        return dict(
            enabled=self.state == "installed",
            state=self.state,
            failure=self.failure,
            mechanism="request_scoped_static_graph_validation",
            counters=dict(self.counts),
            native_full_validation_at_first_use_and_before_return=True,
            validation_inside_timed_generate=True,
            dynamic_input_and_state_storage_checked=True,
            per_step_math_flags_checked=True,
            concurrent_model_metadata_mutation_supported=False,
            pager_and_kernels_changed=False,
            extra_expert_payload_bytes=0,
            retained_input_frames=len(self.frames),
        )

    def close(self):
        if self.state == "closed":
            return
        if self.active:
            raise RuntimeError("Cannot close an active request lease")
        for owner, name, hook, _ in self.bindings:
            if getattr(owner, name) is not hook:
                raise RuntimeError("Remove outer observers before closing")
        for owner, name, _, previous in reversed(self.bindings):
            if previous is None:
                delattr(owner, name)
            else:
                setattr(owner, name, previous)
        if getattr(self.manager, "request_graph_validation", None) is self:
            delattr(self.manager, "request_graph_validation")
        self.state = "closed"


def install_request_graph_validation(manager, engine):
    return RequestGraphValidation(manager, engine)
