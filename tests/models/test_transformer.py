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

import torch

from nrl.models.transformer import SimpleTransformer


def test_transformer_forward_shape():
    batch = 4
    seq_len = 10
    input_dim = 16
    num_classes = 3

    model = SimpleTransformer(input_dim=input_dim, model_dim=32, nhead=4, num_layers=2, num_classes=num_classes)
    x = torch.randn(batch, seq_len, input_dim)
    logits = model(x)

    assert logits.shape == (batch, num_classes)
