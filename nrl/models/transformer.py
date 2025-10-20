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

"""Small Transformer model implementation.

Provides a simple TransformerEncoder-based model with a classification/regression head.

The implementation keeps external dependencies minimal and follows the project's
style for structure and imports.
"""

from typing import Optional

import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    """A lightweight Transformer encoder model.

    Args:
        input_dim: dimensionality of input features per token
        model_dim: transformer hidden dimension (d_model)
        nhead: number of attention heads
        num_layers: number of TransformerEncoder layers
        dim_feedforward: feedforward network hidden dim
        dropout: dropout probability
        num_classes: if provided, a classification head is added (Linear to num_classes).
        pooling: pooling over sequence: 'mean' or 'cls' (first token)

    Inputs/Outputs:
        - forward(x, src_key_padding_mask=None) where x: (B, T, input_dim)
        - returns logits or embedding depending on num_classes
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
        pooling: str = "mean",
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(input_dim, model_dim) if input_dim != model_dim else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = pooling
        self.num_classes = num_classes
        self.model_dim = model_dim

        if num_classes is not None:
            self.head = nn.Linear(model_dim, num_classes)
        else:
            self.head = None

        self._init_weights()

    def _init_weights(self):
        # follow common small init: xavier for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        """Forward pass.

        Args:
            x: (B, T, input_dim)
            src_key_padding_mask: (B, T) with True in positions that should be masked

        Returns:
            If num_classes is set, returns logits (B, num_classes), else returns sequence embedding (B, model_dim)
        """
        # project to model dim
        h = self.input_proj(x)

        # Transformer expects (B, T, C) when batch_first=True
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        if self.pooling == "cls":
            emb = h[:, 0, :]
        else:
            # mean pooling, account for padding mask
            if src_key_padding_mask is not None:
                mask = ~src_key_padding_mask
                mask = mask.unsqueeze(-1).to(h.dtype)
                summed = (h * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                emb = summed / denom
            else:
                emb = h.mean(dim=1)

        if self.head is not None:
            return self.head(emb)
        return emb


__all__ = ["SimpleTransformer"]
