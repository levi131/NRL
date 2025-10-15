# Copyright (c) 2025 levi131. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NRL logging utilities.

Provides an NRLLogger wrapper class around Python's logging.Logger
and a global singleton `global_logger`. The logging level is controllable
through the `NRL_LOG_LEVEL` environment variable.

Public API:
- NRLLogger: wrapper with methods debug/info/warn/error/exception
- get_logger(name): returns an NRLLogger for the given name
- get_global_logger(): returns the singleton NRLLogger
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _get_level_from_env(default: int = logging.INFO) -> int:
    v = os.environ.get("NRL_LOG_LEVEL")
    if not v:
        return default
    v = v.strip().upper()
    return _LEVEL_MAP.get(v, default)


def _make_formatter() -> logging.Formatter:
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    return logging.Formatter(fmt=fmt, datefmt=datefmt)


class NRLLogger:
    """A thin wrapper around logging.Logger with convenient methods.

    Instances wrap an underlying logging.Logger. Calling `get_logger(name)`
    will return a wrapper (cached) for a given name. Handlers are added
    once per underlying logger to avoid duplicate output.
    """

    def __init__(self) -> None:
        # single global logger name
        self._name = "nrl_logger"
        self._logger = logging.getLogger(self._name)
        self._configure()

    def _configure(self) -> None:
        # Always update the level from environment, but avoid adding
        # duplicate handlers if the logger was previously configured.
        level = _get_level_from_env()
        self._logger.setLevel(level)
        if not self._logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(_make_formatter())
            self._logger.addHandler(ch)
        else:
            # update existing handlers to the current level
            for h in self._logger.handlers:
                h.setLevel(level)

        # mark configured so other systems know we've set it up
        self._logger._nrl_configured = True  # type: ignore[attr-defined]

    # convenience methods
    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warn(self, msg: str, *args, **kwargs) -> None:  # keep warn alias
        self._logger.warning(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, exc_info: bool = True, **kwargs) -> None:
        self._logger.exception(msg, *args, exc_info=exc_info, **kwargs)

    # expose underlying logger if needed
    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def set_level(self, level: int) -> None:
        self._logger.setLevel(level)
        for h in self._logger.handlers:
            h.setLevel(level)


# Global singleton (thread-safe)
_GLOBAL_LOCK = threading.Lock()
_GLOBAL_LOGGER: Optional[NRLLogger] = None


def get_global_logger() -> NRLLogger:
    """Return the module-level singleton logger (lazy, thread-safe)."""
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_LOGGER is None:
                _GLOBAL_LOGGER = NRLLogger()
    return _GLOBAL_LOGGER


def set_global_logger(logger: NRLLogger) -> None:
    """Replace the module-level singleton logger (for tests or customization)."""
    global _GLOBAL_LOGGER
    with _GLOBAL_LOCK:
        _GLOBAL_LOGGER = logger


def get_logger() -> NRLLogger:
    """Compatibility alias: return the global NRLLogger singleton."""
    return get_global_logger()


# module-level convenience alias
global_logger = get_global_logger()
