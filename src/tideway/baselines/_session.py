"""Reversible bindings for process-local baseline adapters."""

from dataclasses import dataclass, field
from typing import Any

_ABSENT = object()


@dataclass
class BaselineSession:
    manager: Any = None
    bindings: list = field(default_factory=list)
    closed: bool = False

    def remember(self, owner, *names):
        for name in names:
            self.bindings.append((owner, name, vars(owner).get(name, _ABSENT)))

    def bind(self, owner, name, value):
        self.remember(owner, name)
        setattr(owner, name, value)

    def uninstall(self):
        """Drain the manager and restore bindings after the caller stops inference."""
        if self.closed:
            return
        error = None
        try:
            if self.manager is not None:
                self.manager.close()
        except BaseException as failure:
            error = failure
        for owner, name, previous in reversed(self.bindings):
            try:
                if previous is _ABSENT:
                    if name in vars(owner):
                        delattr(owner, name)
                else:
                    setattr(owner, name, previous)
            except BaseException as failure:
                if error is None:
                    error = failure
                else:
                    error.add_note(f"Baseline binding restoration failed: {failure!r}")
        self.bindings.clear()
        self.closed = True
        if error is not None:
            raise error
