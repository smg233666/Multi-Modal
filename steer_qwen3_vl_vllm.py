# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py
# Copyright 2024 The Qwen team.
# Copyright 2024 The vLLM team.
# Copyright 2024 EleutherAI and the HuggingFace Inc. team.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate architectural differences compared to
# GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only steering-enabled Qwen3-VL model for vLLM."""

import os
from typing import Any, Iterable, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch import nn
from transformers import Qwen3Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
# Handle PoolingMetadata compatibility across vLLM versions.
from vllm.v1.pool.metadata import PoolingMetadata  # vLLM >= 0.10
# Handle SamplingMetadata import differences across versions.
from vllm.v1.sample.metadata import SamplingMetadata  # vLLM >= 0.10
from vllm.model_executor.models.qwen3 import SupportsEagle3, SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, PPMissingLayer, WeightsMapper, extract_layer_index,
    is_pp_missing_parameter, maybe_prefix)
from vllm.model_executor.models.qwen2 import (
    make_empty_intermediate_tensors_factory, make_layers)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration)
# Handle PoolerOutput availability.
try:
    from vllm.sequence import IntermediateTensors, PoolerOutput
except ImportError:  # pragma: no cover - older vLLM releases
    from vllm.sequence import IntermediateTensors
    PoolerOutput = Any  # type: ignore[misc,assignment]

logger = init_logger(__name__)


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        head_dim: Optional[int] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rope_scaling: Optional[Tuple] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        extra_attn_kwargs = {}
        if dual_chunk_attention_config is not None:
            extra_attn_kwargs = {
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=attn_type,
                              **extra_attn_kwargs)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_heads = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                         self.head_dim)
        q = self.q_norm(q_heads).view(*q.shape)

        k_heads = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                         self.head_dim)
        k = self.k_norm(k_heads).view(*k.shape)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class SteerQwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        steering_vector: Optional[torch.Tensor] = None,
        steering_strength: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)
        qkv_bias = getattr(config, "attention_bias", False)
        head_dim = getattr(config, "head_dim", None)

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=qkv_bias,
            head_dim=head_dim,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        self.register_buffer("steering_vector", steering_vector)
        self.register_buffer("steering_strength", steering_strength)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.steering_strength == self.steering_strength
        hidden_states = hidden_states + self.steering_vector * self.steering_strength

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class SteerQwen3Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3Config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                             "supported. This model uses sliding window "
                             "but `max_window_layers` = {} is less than "
                             "`num_hidden_layers` = {}. Please open an issue "
                             "to discuss this feature.".format(
                                 config.max_window_layers,
                                 config.num_hidden_layers,
                             ))

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        steering_vector_path = os.environ.get("steering_vector_path")
        if steering_vector_path is None:
            raise ValueError("Environment variable 'steering_vector_path' is not set.")
        if not os.path.exists(steering_vector_path):
            raise FileNotFoundError(
                f"Steering vector file not found: {steering_vector_path}")

        dtype = getattr(vllm_config, "dtype", torch.bfloat16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        steering_vectors = torch.from_numpy(
            np.load(steering_vector_path)).to(dtype=dtype, device=device)

        steering_strength_str = os.environ.get("steering_strength_list")
        if steering_strength_str is None:
            raise ValueError(
                "Environment variable 'steering_strength_list' is not set.")
        steering_strengths = torch.tensor(
            list(map(float, steering_strength_str.split(","))),
            dtype=dtype,
            device=device,
        )

        total_layers = config.num_hidden_layers
        if steering_vectors.shape[0] != total_layers:
            raise ValueError(
                f"Expected steering vectors for {total_layers} layers, "
                f"found {steering_vectors.shape[0]}.")
        if steering_strengths.numel() != total_layers:
            raise ValueError(
                f"Expected {total_layers} steering strengths, "
                f"found {steering_strengths.numel()}.")

        def layer_factory(layer_prefix: str) -> nn.Module:
            layer_idx = int(layer_prefix.split(".")[-1])
            return SteerQwen3DecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=layer_prefix,
                steering_vector=steering_vectors[layer_idx],
                steering_strength=steering_strengths[layer_idx],
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            layer_factory,
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class SteerQwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.model = SteerQwen3Model(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def set_aux_hidden_state_layers(self, layers: Tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> Tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


class Qwen3EmbeddingModel(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.model = SteerQwen3Model(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "model"))

        if pooler_config.pooling_type is None:
            logger.warning(
                "This embedding model will default to last-token pooling in "
                "an upcoming version. To avoid breaking changes, you should "
                "pass `--override-pooler-config '{\"pooling_type\": \"MEAN\"}'` "
                "explicitly.")

        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.MEAN,
            normalize=True,
            softmax=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = self.hf_to_vllm_mapper.apply(weights)
        weights = ((name, data) for name, data in weights
                   if not name.startswith("lm_head."))
        self.model.load_weights(weights)


class SteerQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Replace the base language model with the steering-enabled Qwen3 model.
        self.language_model = SteerQwen3ForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"))
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)
