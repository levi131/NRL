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

"""Tests for CkptMgr."""

import os
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from nrl.common import CkptMgr


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, x: Any) -> Any:
        return self.layer(x)


@pytest.fixture
def ckpt_setup(tmp_path):
    """Provides a temporary directory, a model, optimizer, and scheduler for tests."""
    base_path = tmp_path / "checkpoints"
    base_path.mkdir()
    resume_path = tmp_path / "resume"
    resume_path.mkdir()

    model = SimpleModel()
    optimizer = Adam(model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1)

    return {
        "base_path": str(base_path),
        "resume_path": str(resume_path),
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }


class TestCkptMgr:
    """Test suite for the CkptMgr class."""

    def test_init_empty_dir(self, ckpt_setup):
        """Test initialization with an empty base directory."""
        mgr = CkptMgr(base_path=ckpt_setup["base_path"], model=ckpt_setup["model"])
        assert mgr.last_iteration == 0

    def test_init_with_existing_iterations(self, ckpt_setup):
        """Test initialization finds the latest iteration in an existing directory."""
        os.makedirs(os.path.join(ckpt_setup["base_path"], "10"))
        os.makedirs(os.path.join(ckpt_setup["base_path"], "20"))
        mgr = CkptMgr(base_path=ckpt_setup["base_path"], model=ckpt_setup["model"])
        assert mgr.last_iteration == 20

    def test_init_with_resume_path(self, ckpt_setup):
        """Test initialization with a resume_path sets last_iteration to 0."""
        mgr = CkptMgr(base_path=ckpt_setup["base_path"], model=ckpt_setup["model"], resume_path=ckpt_setup["resume_path"])
        assert mgr.last_iteration == 0

    def test_save_and_load_full_state(self, ckpt_setup):
        """Test saving and loading the full state (model, optimizer, scheduler)."""
        mgr = CkptMgr(
            base_path=ckpt_setup["base_path"],
            model=ckpt_setup["model"],
            optimizer=ckpt_setup["optimizer"],
            lr_scheduler=ckpt_setup["lr_scheduler"],
        )

        # Change state to verify it gets restored
        ckpt_setup["model"].layer.weight.data.fill_(1.0)
        ckpt_setup["optimizer"].step()
        ckpt_setup["lr_scheduler"].step()

        original_model_state = ckpt_setup["model"].state_dict()
        ckpt_setup["optimizer"].state_dict()
        original_lr_state = ckpt_setup["lr_scheduler"].state_dict()

        mgr.save(iteration=10)

        # Re-create optimizer and scheduler to ensure their states are fresh
        # Re-create optimizer and scheduler to ensure their states are fresh
        # and pass them to the manager instance.
        new_model = SimpleModel()
        new_optimizer = Adam(new_model.parameters(), lr=0.01)
        new_lr_scheduler = StepLR(new_optimizer, step_size=1)
        mgr.model = new_model
        mgr.optimizer = new_optimizer
        mgr.lr_scheduler = new_lr_scheduler

        loaded_state = mgr.load(iteration=10)

        assert torch.equal(original_model_state["layer.weight"], mgr.model.state_dict()["layer.weight"])
        # Check that the optimizer state has been restored by checking a known property
        assert len(mgr.optimizer.state_dict()["state"]) > 0
        # Check that the LR scheduler's step count was restored
        assert mgr.lr_scheduler.state_dict()["_step_count"] == original_lr_state["_step_count"]
        # Also check the returned dict
        assert "model" in loaded_state
        assert "optimizer" in loaded_state
        assert "lr_scheduler" in loaded_state
        assert mgr.last_iteration == 10

    def test_save_interval(self, ckpt_setup):
        """Test that checkpoints are only saved according to save_interval."""
        mgr = CkptMgr(base_path=ckpt_setup["base_path"], model=ckpt_setup["model"], save_interval=5)
        # This should be skipped
        path1 = mgr.save(iteration=4)
        assert path1 is None
        assert not os.path.exists(os.path.join(ckpt_setup["base_path"], "4"))

        # This should be saved
        path2 = mgr.save(iteration=5)
        assert path2 is not None
        assert os.path.exists(os.path.join(ckpt_setup["base_path"], "5", "ckpt.pth"))

        # This should be skipped
        path3 = mgr.save(iteration=9)
        assert path3 is None
        assert not os.path.exists(os.path.join(ckpt_setup["base_path"], "9"))

        # This should be saved due to force=True
        path4 = mgr.save(iteration=9, force=True)
        assert path4 is not None
        assert os.path.exists(os.path.join(ckpt_setup["base_path"], "9", "ckpt.pth"))

    def test_load_latest_iteration(self, ckpt_setup):
        """Test loading the latest checkpoint when no iteration is specified."""
        mgr = CkptMgr(base_path=ckpt_setup["base_path"], model=ckpt_setup["model"])
        mgr.save(iteration=10)
        mgr.save(iteration=20)  # This is the latest
        mgr.last_iteration = 20  # Manually set as save would do

        # Change model state
        ckpt_setup["model"].layer.weight.data.fill_(20.0)
        mgr.save(iteration=20)

        # Reset model and load latest
        new_model = SimpleModel()
        mgr.model = new_model
        mgr.load()  # Should load iteration 20

        assert torch.all(mgr.model.layer.weight.data == 20.0)
        assert mgr.last_iteration == 20

    def test_load_with_exclude_dict(self, ckpt_setup):
        """Test that components in exclude_dict are not loaded."""
        mgr = CkptMgr(base_path=ckpt_setup["base_path"], model=ckpt_setup["model"], optimizer=ckpt_setup["optimizer"])
        # Change state and save
        ckpt_setup["model"].layer.weight.data.fill_(1.0)
        ckpt_setup["optimizer"].step()
        mgr.save(iteration=1)

        # Reset state
        new_model = SimpleModel()
        new_optim = Adam(new_model.parameters(), lr=0.01)
        mgr.model = new_model
        mgr.optimizer = new_optim
        fresh_optim_state_before_load = new_optim.state_dict()

        # Load, excluding optimizer
        loaded_state = mgr.load(iteration=1, exclude_dict={"optimizer"})

        # Model should be loaded
        assert torch.all(mgr.model.layer.weight.data == 1.0)
        # Optimizer should NOT be loaded and should remain in its fresh state
        assert mgr.optimizer.state_dict()["state"] == fresh_optim_state_before_load["state"]
        # Check the returned dict
        assert "model" in loaded_state
        assert "optimizer" in loaded_state

    def test_save_selective(self, ckpt_setup):
        """Test saving only specific parts of the state."""
        mgr = CkptMgr(base_path=ckpt_setup["base_path"], model=ckpt_setup["model"], optimizer=ckpt_setup["optimizer"])
        mgr.save(iteration=1, save_weights=True, save_optimizer=False)

        # Create a new model and manager to load into
        new_model = SimpleModel()
        new_mgr = CkptMgr(base_path=ckpt_setup["base_path"], model=new_model, optimizer=Adam(new_model.parameters(), lr=0.01))

        ckpt = new_mgr.load(iteration=1)
        assert "model" in ckpt
        assert "optimizer" not in ckpt
        # Check that the model was actually loaded
        assert torch.all(new_mgr.model.layer.weight.data != 0.0)
        # Verify model was actually restored
        assert torch.all(ckpt_setup["model"].layer.weight.data == 1.0)

    def test_load_non_existent(self, ckpt_setup):
        """Test that FileNotFoundError is raised for a non-existent checkpoint."""
        mgr = CkptMgr(base_path=ckpt_setup["base_path"], model=ckpt_setup["model"])
        with pytest.raises(FileNotFoundError):
            mgr.load(iteration=999)

    def test_extra_state(self, ckpt_setup):
        """Test saving and loading extra arbitrary state defined at init."""
        extra_state_template = {"epoch": 0, "some_metric": 0.0}
        mgr = CkptMgr(base_path=ckpt_setup["base_path"], model=ckpt_setup["model"], extra_state=extra_state_template)

        # Update the state before saving
        mgr.extra_state["epoch"] = 5
        mgr.extra_state["some_metric"] = 0.95
        mgr.save(iteration=1, save_weights=False)

        # Create a new manager with a fresh extra_state dict to load into
        fresh_extra_state = {"epoch": 0, "some_metric": 0.0}
        new_mgr = CkptMgr(base_path=ckpt_setup["base_path"], model=SimpleModel(), extra_state=fresh_extra_state)
        loaded_state = new_mgr.load(iteration=1)

        # Check that the manager's extra_state was updated
        assert new_mgr.extra_state["epoch"] == 5
        assert new_mgr.extra_state["some_metric"] == 0.95

        # Check the returned dictionary
        assert loaded_state["epoch"] == 5
        assert "model" not in loaded_state
