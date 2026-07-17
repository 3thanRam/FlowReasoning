from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

ExecutorMode = Literal["single", "paths"]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embedding")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, :, None, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        cos = self.cos_cached[:, :seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :seq_len].to(dtype=x.dtype, device=x.device)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim)
        q = self.rotary(q).transpose(1, 2)
        k = self.rotary(k).transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(bsz, seq_len, dim))


class StableResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden = hidden_mult * dim
        self.in_proj = nn.Linear(dim, 2 * hidden, bias=False)
        self.act = SwiGLU()
        self.out_proj = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.act(self.in_proj(x)))


class DiagonalUnitaryMemory(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even for diagonal unitary memory")
        self.half = dim // 2
        self.phases = nn.Parameter(torch.zeros(self.half))
        self.gate_raw = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_re, x_im = x[..., :self.half], x[..., self.half:]
        cos_p, sin_p = torch.cos(self.phases), torch.sin(self.phases)
        rot_re = x_re * cos_p - x_im * sin_p
        rot_im = x_re * sin_p + x_im * cos_p
        return x + torch.sigmoid(self.gate_raw) * torch.cat([rot_re, rot_im], dim=-1)


class FlowOperator(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalAttention(dim, num_heads)
        self.unitary = DiagonalUnitaryMemory(dim)
        self.mlp = StableResidualMLP(dim)
        self.out_norm = RMSNorm(dim)

        self.freqs = nn.Parameter(torch.linspace(0.5, max_seq_len / 4.0, dim).unsqueeze(0))
        self.decays = nn.Parameter(torch.linspace(0.01, 2.0, dim).unsqueeze(0))
        self.ssm_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.ssm_select = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def kernel_fft(self, seq_len: int, device: torch.device) -> torch.Tensor:
        with torch.amp.autocast(device_type=device.type, enabled=False):
            t = (torch.arange(seq_len, device=device, dtype=torch.float32) / max(float(seq_len), 1024.0)).unsqueeze(-1)
            decays = F.softplus(self.decays.float()) + 0.01
            kernel = torch.sin(2 * math.pi * self.freqs.float() * t) * torch.exp(-decays * t)
            kernel = 0.02 * kernel.unsqueeze(0)
            fft_len = 1 << max(1, (2 * seq_len - 1).bit_length())
            padded = F.pad(kernel, (0, 0, 0, fft_len - seq_len))
            return torch.fft.rfft(padded, dim=1) / float(max(seq_len, 1))

    def ssm_mix(self, x: torch.Tensor, kernel_fft: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        dtype = x.dtype
        gate = self.ssm_gate(x)
        select = self.ssm_select(x)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            fft_len = (kernel_fft.shape[1] - 1) * 2
            padded = F.pad(xf, (0, 0, 0, fft_len - seq_len))
            y = torch.fft.irfft(torch.fft.rfft(padded, dim=1) * kernel_fft, n=fft_len, dim=1)[:, :seq_len]
            out = xf + gate.float() * select.float() * y
        return out.to(dtype)

    def forward(
        self,
        z_original: torch.Tensor,
        z_curr: torch.Tensor,
        step_emb: torch.Tensor,
        kernel_fft: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm(z_curr + step_emb)
        x = self.ssm_mix(x, kernel_fft)
        x = x + self.attn(self.attn_norm(x))
        dz = self.mlp(self.unitary(x))
        return self.out_norm(dz + 0.01 * z_original)


@dataclass
class FlowResult:
    z: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


class FlowEvolver(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        num_heads: int,
        max_flow_steps: int = 16,
        max_step_size: float = 0.25,
        dz_clip: float = 3.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_flow_steps = max_flow_steps
        self.max_step_size = max_step_size
        self.dz_clip = dz_clip

        self.operator = FlowOperator(dim, max_seq_len, num_heads)
        self.step_embed = nn.Embedding(max_flow_steps + 1, dim)
        self.step_gate = nn.Linear(dim, 1)
        self.post_step_norm = RMSNorm(dim)

        self.path_embed = nn.Embedding(64, dim)
        self.path_score = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.path_noise_raw = nn.Parameter(torch.tensor(-5.0))


    def _step(
        self,
        z0: torch.Tensor,
        z: torch.Tensor,
        kernel_fft: torch.Tensor,
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_idx = min(step_idx, self.max_flow_steps)
        step_emb = self.step_embed.weight[step_idx].view(1, 1, -1)
        step_emb = step_emb.to(dtype=z.dtype, device=z.device)

        dz = self.operator(z0, z, step_emb, kernel_fft)
        dz = torch.clamp(dz, -self.dz_clip, self.dz_clip)

        step_size = torch.sigmoid(self.step_gate(z)) * self.max_step_size
        z_next = self.post_step_norm(z + step_size * dz)
        return z_next, dz


    def forward_single(
    self,
    z: torch.Tensor,
    flow_steps: int,
    ) -> FlowResult:
        z0 = z
        z_current = z
        kernel_fft = self.operator.kernel_fft(z.shape[1], z.device)
        dz_norms = []

        for step_index in range(flow_steps):
            z_current, dz = self._step(
                z0,
                z_current,
                kernel_fft,
                step_index,
            )
            dz_norms.append(dz.pow(2).mean().detach())

        mean_dz_norm = (
            torch.stack(dz_norms).mean()
            if dz_norms
            else torch.zeros((), device=z.device)
        )

        return FlowResult(
            z=z_current,
            diagnostics={
                "mode": "single",
                "mean_dz_norm": mean_dz_norm,
            },
        )

    def forward_paths(
        self,
        z: torch.Tensor,
        flow_steps: int,
        num_paths: int,
        eval_noise: bool = False,
    ) -> FlowResult:
        if num_paths > self.path_embed.num_embeddings:
            raise ValueError(f"num_paths must be <= {self.path_embed.num_embeddings}")

        bsz, seq_len, dim = z.shape
        kernel_fft = self.operator.kernel_fft(seq_len, z.device)

        z0 = z[:, None].expand(bsz, num_paths, seq_len, dim).reshape(bsz * num_paths, seq_len, dim)
        zc = z0.clone()

        path_ids = torch.arange(num_paths, device=z.device)
        path_bias = self.path_embed(path_ids).to(dtype=z.dtype).view(1, num_paths, 1, dim)
        zc = zc + path_bias.expand(bsz, num_paths, seq_len, dim).reshape(bsz * num_paths, seq_len, dim)

        logw = torch.zeros(bsz * num_paths, dtype=z.dtype, device=z.device)
        noise_scale = torch.exp(self.path_noise_raw)

        for i in range(flow_steps):
            z_prev = zc
            zc, dz = self._step(z0, zc, kernel_fft, i)
            if self.training or eval_noise:
                zc = zc + noise_scale * torch.randn_like(zc)
            score_in = torch.cat([z_prev.mean(1), zc.mean(1), dz.mean(1)], dim=-1)
            logw = logw + self.path_score(score_in).squeeze(-1)

        z_paths = zc.view(bsz, num_paths, seq_len, dim)
        logw = logw.view(bsz, num_paths)
        weights = torch.softmax(logw, dim=1)
        z_bar = (weights[:, :, None, None] * z_paths).sum(dim=1)

        var = (weights[:, :, None, None] * (z_paths - z_bar[:, None]).pow(2)).sum(dim=1).mean(dim=(1, 2))
        ess = 1.0 / weights.pow(2).sum(dim=1).clamp_min(1e-8)
        return FlowResult(
            z=z_bar,
            diagnostics={
                "mode": "paths",
                "path_log_weights": logw.detach(),
                "path_weights": weights.detach(),
                "latent_path_variance": var.detach(),
                "effective_sample_size": ess.detach(),
                "z_paths": z_paths.detach(),
            },
        )

    def forward(
        self,
        z: torch.Tensor,
        flow_steps: int,
        mode: ExecutorMode = "single",
        num_paths: int = 1,
        eval_noise: bool = False,
    ) -> FlowResult:
        flow_steps = max(0, min(int(flow_steps), self.max_flow_steps))

        if mode == "paths" and num_paths > 1:
            return self.forward_paths(
                z,
                flow_steps,
                num_paths,
                eval_noise=eval_noise,
            )

        return self.forward_single(z, flow_steps)


