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

"""Checkpoint manager (CkptMgr).

Clean, single-definition implementation of CkptMgr. It stores checkpoints
under numeric iteration subfolders and optionally prefers a resume_path
on first load.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
from torch.distributed import checkpoint as dcp

from nrl.common.logger import nrl_logger


class CkptMgr:
    """Checkpoint manager backed by torch.distributed.checkpoint.

    Members:
      - base_path: directory containing iteration subfolders
      - resume_path: optional path or directory to prefer on first load
      - model: torch.nn.Module managed by the manager
      - optimizer: optional torch.optim.Optimizer managed by the manager
      - lr_scheduler: optional learning rate scheduler managed by the manager
      - last_iteration: last saved/loaded iteration (int or None)
      - save_interval: save every N iterations (default 1)
    """

    def __init__(
        self,
        base_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        resume_path: Optional[str] = None,
        save_interval: int = 1,
    ) -> None:
        self.base_path = base_path
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.extra_state = extra_state
        self.resume_path = resume_path
        self.save_interval = int(save_interval)
        os.makedirs(self.base_path, exist_ok=True)

        if self.resume_path:
            self.last_iteration = 0
        else:
            try:
                entries = [
                    d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d)) and d.isdigit()
                ]
                self.last_iteration = max(int(d) for d in entries) if entries else 0
            except FileNotFoundError:
                self.last_iteration = 0

    def _make_path(self, iteration: int) -> str:
        return os.path.join(self.base_path, str(iteration), "ckpt.pth")

    def save(
        self,
        iteration: int,
        force: bool = False,
        save_weights: bool = True,
        save_optimizer: bool = True,
        save_lr_scheduler: bool = True,
    ) -> Optional[str]:
        """Save checkpoint state. Skip saving if iteration not aligned with save_interval."""
        if not force and self.save_interval > 1:
            if (int(iteration) % self.save_interval) != 0:
                nrl_logger.debug("skip saving iteration %s due to save_interval=%s", iteration, self.save_interval)
                return None

        state_to_save: Dict[str, Any] = {}
        if save_weights:
            state_to_save["model"] = self.model.state_dict()
        if save_optimizer and self.optimizer:
            state_to_save["optimizer"] = self.optimizer.state_dict()
        if save_lr_scheduler and self.lr_scheduler:
            state_to_save["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.extra_state:
            state_to_save.update(self.extra_state)

        if not state_to_save:
            nrl_logger.warning("save was called but nothing was saved.")
            return None

        path = self._make_path(iteration=iteration)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dcp.save(state_dict=state_to_save, checkpoint_id=path)
        self.last_iteration = int(iteration)
        nrl_logger.info("checkpoint saved via dcp to %s", path)
        return path

    def load(self, iteration: Optional[int] = None, exclude_dict: Optional[set[str]] = None) -> Dict[str, Any]:
        """Load a checkpoint and restore managed components using a load planner.

        Loading rules:
        - If iteration is provided, load from that iteration.
        - Otherwise, load from the latest iteration in base_path.
        """
        if exclude_dict is None:
            exclude_dict = set()

        path = None
        if iteration is not None:
            path = self._make_path(iteration)
            self.last_iteration = iteration
        else:
            if self.last_iteration is not None and self.last_iteration > 0:
                path = self._make_path(self.last_iteration)
            else:
                raise FileNotFoundError("no checkpoint path available to load")

        if not os.path.exists(path):
            raise FileNotFoundError(f"no checkpoint file found for iteration {iteration} or path {path}")

        # Step 1: Perform in-place loading for managed components.
        state_to_load: Dict[str, Any] = {}
        if "model" not in exclude_dict:
            state_to_load["model"] = self.model
        if self.optimizer and "optimizer" not in exclude_dict:
            state_to_load["optimizer"] = self.optimizer
        if self.lr_scheduler and "lr_scheduler" not in exclude_dict:
            state_to_load["lr_scheduler"] = self.lr_scheduler
        # Also include extra_state in the in-place loading template
        if self.extra_state:
            for k in self.extra_state:
                if k not in exclude_dict:
                    state_to_load[k] = None

        planner = dcp.DefaultLoadPlanner(allow_partial_load=True)
        dcp.load(state_dict=state_to_load, checkpoint_id=path, planner=planner)
        nrl_logger.info("In-place checkpoint loaded via dcp from %s", path)

        for k in state_to_load:
            if k in self.extra_state:
                self.extra_state[k] = state_to_load[k]
        return state_to_load


__all__ = ["CkptMgr"]
