import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import json
import os
from functools import partial
from einops import rearrange, repeat
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertLMHeadModel
from transformers.models.bert.configuration_bert import BertConfig
from timm.models.regnet import RegStage


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        patch_dropout = 0.
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.patch_dropout = patch_dropout

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t, patch_indices_keep=None):
        return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                norm_layer=nn.LayerNorm, subln=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, value_dim=1536, num_heads=16, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, xattn=False, rope=None, subln=True,
            norm_layer=nn.LayerNorm, rope_hw=37):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.subln = subln
        if self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(value_dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(value_dim, all_head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.window_size = None
        self.relative_position_bias_table = None
        self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

        self.rope = VisionRotaryEmbeddingFast(
            dim=64//2,
            pt_seq_len=rope_hw,
            ft_seq_len=rope_hw,
        )

    def forward(self, x, value, attn_mask=None):
        B, N, C = x.shape
        B, N_, C_ = value.shape
        if self.subln:
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            k = F.linear(input=value, weight=self.k_proj.weight, bias=None)
            v = F.linear(input=value, weight=self.v_proj.weight, bias=self.v_bias)

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)     # B, num_heads, N, C
            k = k.reshape(B, N_, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(B, N_, self.num_heads, -1).permute(0, 2, 1, 3)
            # B, num_heads, N, C
        else:
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope:
            # slightly fast impl
            q_t = q
            ro_q_t = self.rope(q_t)
            q = ro_q_t.type_as(v)

            k_t = k
            ro_k_t = self.rope(k_t)
            k = ro_k_t.type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.inner_attn_ln(x)
        x = self.proj(x)
        return x


class AdapterBlock(nn.Module):
    def __init__(self, in_channels, expert_channels, topk=3, rope_hw=48):
        super().__init__()
        self.topk = topk
        self.num_experts = len(expert_channels)

        self.cross_attn = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.mlp = SwiGLU(in_channels, int(in_channels*8/3), in_channels)
        self.self_attn = Attention(in_channels, in_channels, rope_hw=rope_hw)
        self.norm3 = nn.LayerNorm(in_channels)
        self.norm4 = nn.LayerNorm(in_channels)
        self.gate = nn.Sequential(
            nn.Linear(1024+in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, len(expert_channels))
        )
        nn.init.zeros_(self.self_attn.proj.weight)
        nn.init.zeros_(self.self_attn.proj.bias)
        nn.init.zeros_(self.mlp.w3.weight)
        nn.init.zeros_(self.mlp.w3.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        for value_dim in expert_channels:
            self.cross_attn.append(Attention(in_channels, value_dim, rope_hw=rope_hw))
            self.norm1.append(nn.LayerNorm(in_channels))
            self.norm2.append(nn.LayerNorm(value_dim))
            nn.init.zeros_(self.cross_attn[-1].proj.weight)
            nn.init.zeros_(self.cross_attn[-1].proj.bias)


    def forward(self, x, value, text_feat, routing_mask, routing_weight_mask, routing_binary_weight=None):
        B, L, C = x.shape
        result = []

        # Dynamic Gating Network
        router_logits = self.gate(torch.cat([text_feat, x.mean(1)], dim=-1)) # B, num_experts
        routing_weights = F.softmax(router_logits.float() + routing_weight_mask, dim=-1, dtype=torch.float)
        _, selected_experts = torch.topk(routing_weights, self.topk, dim=-1)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts)
        routing_weights = routing_weights * expert_mask.sum(1)
        routing_weights = routing_weights / (routing_weights.sum(1).unsqueeze(1)) # B, N
        routing_weights = routing_weights.to(x.dtype)

        result = []
        for expert_idx, expert_feat in enumerate(value):
            if (not self.training) and B==1 and routing_mask[0][expert_idx]==0:
                # We skip the cross-attention layer with non-relevant experts during inference
                current_expert_states = torch.zeros_like(x)
            else:
                current_expert_states = self.cross_attn[expert_idx](self.norm1[expert_idx](x), self.norm2[expert_idx](expert_feat)) # B, L, C
            result.append(current_expert_states)
        result = torch.stack(result, dim=1) # B, N, L, C
        if routing_weights is not None:
            result = torch.sum(result * routing_weights.unsqueeze(-1).unsqueeze(-1), dim=1)
        else:
            result = result.mean(dim=1)
        if routing_binary_weight is not None:
            result = result * routing_binary_weight.unsqueeze(-1).unsqueeze(-1)
        x = x + result.to(x.dtype)
        residual = x
        x = self.norm3(x)
        x = residual + self.self_attn(x, x)
        x = x + self.mlp(self.norm4(x))
        return x


class MoVAdapter(nn.Module):
    def __init__(self, config):
        # in_channels, channels, expert_channels, num_layers=3, topk=3, resolution=48, freeze_text_encoder=False
        super().__init__()
        self.in_channels = in_channels = config.mm_hidden_size
        self.channels = channels = config.hidden_size
        self.topk = topk = config.topk_experts
        self.resolution = resolution = config.image_feat_size
        self.freeze_text_encoder = config.mm_projector_freeze_text_encoder
        # expert_channels = [1536, 512, 256, 1536, 768, 768, 256]
        self.expert_channels = expert_channels = config.expert_channels
        self.num_experts = len(expert_channels)
        self.num_layers = num_layers = config.num_projector_layers
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(AdapterBlock(in_channels, expert_channels, topk, rope_hw=self.resolution))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

        # initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained(config.mm_projector_text_encoder, truncation_side="right")
        self.tokenizer = tokenizer
        # initialize BERT
        self.text_encoder = BertLMHeadModel.from_pretrained(config.mm_projector_text_encoder)
        self.text_encoder.resize_token_embeddings(len(tokenizer))
        if self.freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
        else:
            self.text_encoder.requires_grad_(True)
        self.text_encoder.cls = nn.Identity()

        # initialize downsampling blocks
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, self.resolution**2, in_channels))
        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            linear_out=True,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )
        s1 = RegBlock(
            1,
            in_channels,
            in_channels,
        )
        s1.b1.zero_init_last()
        sampler = nn.AdaptiveAvgPool2d((self.resolution//2, self.resolution//2))
        s2 = RegBlock(
            1,
            in_channels,
            in_channels,
        )
        s2.b1.zero_init_last()
        self.downsample = nn.Sequential(s1, sampler, s2)

    def forward(self, x, routing_weights=None, prompts=None):
        value = []
        prev_channel = self.in_channels
        for C in self.expert_channels:
            value.append(x[:, :, prev_channel:(prev_channel+C)])
            prev_channel += C
        x = x[:, :, :self.in_channels]
        residual = x

        x = x + self.pos_embed
        B, L, C = x.shape

        if routing_weights is not None:
            routing_weights = routing_weights[:, :self.num_experts]
            routing_mask = routing_weights
            routing_weight_mask = 1 - routing_weights
            routing_weight_mask = routing_weight_mask.float() * (-1e6)
            routing_binary_weight = routing_weights.mean(-1)
            routing_binary_weight = torch.where(routing_binary_weight>0, torch.ones_like(routing_binary_weight), torch.zeros_like(routing_binary_weight))
        else:
            routing_mask = None
            routing_weight_mask = None
            routing_binary_weight = None

        assert prompts is not None
        text = []
        for i in range(len(prompts)):
            assert len(prompts[i]) > 0
            questions = ''
            for j in range(len(prompts[i])):
                questions = questions + "question <i>: <q>\n\n".replace('<i>', str(j)).replace('<q>', prompts[i][j].lower())
            text.append(questions)
        text_tokens = self.tokenizer(
            text,
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(x.device)

        if self.freeze_text_encoder:
            with torch.no_grad():
                text_output = self.text_encoder.bert(
                    text_tokens.input_ids,
                    attention_mask=text_tokens.attention_mask,
                    return_dict=True,
                )
        else:
            text_output = self.text_encoder.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
        text_feat = text_output.last_hidden_state[:, 0, :].to(dtype=x.dtype, device=x.device)
        if x.shape[0] > text_feat.shape[0]:
            text_feat = text_feat.repeat(x.shape[0], 1)

        for blk in self.blocks:
            x = blk(x, value, text_feat, routing_mask, routing_weight_mask, routing_binary_weight)

        x = (x + residual)/2

        x = rearrange(x, "b (h w) d -> b d h w", h=self.resolution, w=self.resolution)
        x = self.downsample(x)
        x = rearrange(x, "b d h w -> b (h w) d", h=self.resolution//2, w=self.resolution//2)

        x = self.mlp(x)
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'mov_adapter'}


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == "mov_adapter":
        return MoVAdapter(config)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
