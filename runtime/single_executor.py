# -*- coding: utf-8 -*-
"""Single-entry dispatch guard for ``DecisionRequest``.

It is an integration boundary, not a vehicle controller.  A hardware adapter
can register exactly one executor and return its own acknowledgement payload.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from runtime.decision_request import from_decision_request
from schema import layered as sl
from tree import global_mode as gm


class ExecutorRegistrationError(RuntimeError):
    pass


class SingleExecutorArbiter:
    """Permit one registered command executor for a session."""

    def __init__(self):
        self._executor_name: str | None = None
        self._executor: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None
        self._dispatch_sequence = 0
        self._history: list[dict[str, Any]] = []

    @property
    def executor_name(self) -> str | None:
        return self._executor_name

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._history)

    def register(self, name: str, executor: Callable[[Mapping[str, Any]], Mapping[str, Any] | None]) -> None:
        if not name:
            raise ValueError("executor name is required")
        if not callable(executor):
            raise TypeError("executor must be callable")
        if self._executor_name is not None and self._executor_name != name:
            raise ExecutorRegistrationError(
                f"executor already registered: {self._executor_name}")
        self._executor_name = str(name)
        self._executor = executor

    def unregister(self, name: str) -> None:
        if self._executor_name is None:
            return
        if name != self._executor_name:
            raise ExecutorRegistrationError("only the active executor can unregister")
        self._executor_name = None
        self._executor = None

    def dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self._executor is None or self._executor_name is None:
            raise ExecutorRegistrationError("no executor is registered")
        if request.get("interface") != "DecisionRequest":
            raise ValueError("expected DecisionRequest")
        if request.get("sentry_state") not in gm.SENTRY_STATES:
            raise ValueError("invalid sentry state")
        valid, errors = sl.validate(from_decision_request(dict(request)))
        if not valid:
            raise ValueError(f"schema-invalid DecisionRequest: {errors}")

        self._dispatch_sequence += 1
        response = self._executor(dict(request))
        receipt = {
            "dispatch_id": self._dispatch_sequence,
            "executor": self._executor_name,
            "request_t": request.get("t"),
            "accepted_for_dispatch": True,
            "controller_response": dict(response or {}),
        }
        self._history.append(receipt)
        return dict(receipt)
