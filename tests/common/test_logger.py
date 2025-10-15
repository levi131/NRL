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

import importlib
import logging


class TestNRLLogger:
    def test_handlers_and_capture(self, caplog):
        # import module directly to avoid stale references
        mod = importlib.import_module("nrl.common.logger")
        importlib.reload(mod)

        get_nrl = mod.get_nrl_logger
        l1 = get_nrl()

        # handlers should not be duplicated
        assert len(l1.logger.handlers) == 1

        # check that info logs are captured
        with caplog.at_level(logging.INFO):
            l1.info("hello from global")
        assert any(r.levelno == logging.INFO and "hello from global" in r.getMessage() for r in caplog.records)

    def test_singleton_and_setter(self, caplog):
        mod = importlib.import_module("nrl.common.logger")
        importlib.reload(mod)

        get_nrl = mod.get_nrl_logger
        set_nrl = mod.set_nrl_logger
        NRLLogger = mod.NRLLogger

        g1 = get_nrl()
        g2 = get_nrl()
        assert g1 is g2

        # replace global
        new = NRLLogger()
        set_nrl(new)
        assert get_nrl() is new

        with caplog.at_level(logging.INFO):
            get_nrl().info("global replaced")
        assert any(r.levelno == logging.INFO and "global replaced" in r.getMessage() for r in caplog.records)

    def test_env_level_controls_new_logger(self, caplog, monkeypatch):
        # ensure clean import and no cached wrappers
        module = importlib.import_module("nrl.common.logger")
        # set env to DEBUG
        monkeypatch.setenv("NRL_LOG_LEVEL", "DEBUG")
        importlib.reload(module)

        lg = module.get_nrl_logger()

        # capture debug output
        with caplog.at_level(logging.DEBUG):
            lg.debug("debugging env_test")

        assert any(r.levelno == logging.DEBUG and "debugging env_test" in r.getMessage() for r in caplog.records)

    def test_exception_logs_exc_info(self, caplog):
        mod = importlib.import_module("nrl.common.logger")
        importlib.reload(mod)
        lg = mod.get_nrl_logger()

        try:
            raise ValueError("boom")
        except ValueError:
            with caplog.at_level(logging.ERROR):
                lg.exception("caught error")

        found = [r for r in caplog.records if r.levelno == logging.ERROR and "caught error" in r.getMessage()]
        assert found, "exception log should be emitted"
        # ensure exc_info present on the record
        assert any(r.exc_info is not None for r in found)
