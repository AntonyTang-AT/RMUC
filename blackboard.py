# -*- coding: utf-8 -*-
"""M2 异步黑板: 线程安全共享存储, 解耦 RL 产生线程 / LLM 慢环线程 / 融合消费线程。
无 torch, 纯标准库, 离线可测。"""
import threading
import time


class Blackboard:
    def __init__(self):
        self._data = {}
        self._cond = threading.Condition(threading.Lock())
        self._subs = {}

    def put(self, key, value):
        with self._cond:
            self._data[key] = value
            callbacks = list(self._subs.get(key, ()))
            self._cond.notify_all()
        for cb in callbacks:
            try:
                cb(key, value)
            except Exception:
                pass

    def get(self, key, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while key not in self._data:
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise KeyError(f'blackboard timeout: {key}')
                    self._cond.wait(remaining)
            return self._data[key]

    def get_or_default(self, key, default):
        with self._cond:
            return self._data.get(key, default)

    def snapshot(self):
        with self._cond:
            return dict(self._data)

    def subscribe(self, key, callback):
        with self._cond:
            self._subs.setdefault(key, []).append(callback)

    def close(self):
        with self._cond:
            self._subs = {}
            self._cond.notify_all()
