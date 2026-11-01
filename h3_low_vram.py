"""H3 chunked inference extracted from the validated Linux/Windows patch.

SPDX-License-Identifier: GPL-3.0-or-later

Tokenwise projections and MLP are processed in 1024-row chunks. Attention
retains full K/V and recomputes Q in chunks; no local attention approximation.
Layer-local weight casts are reused. Copied embeddings are released after
packing. This version requires exact upstream source fingerprints.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.abc
import importlib.machinery
import logging
import sys

TARGET = "comfy.ldm.minimax.model"
ROWS = 1024
LAYERS = 50
EXPECTED = {
    ("Attention", "forward"): "e7110e547145f705582103df8238e6acad55d45ecb3d9c876223ca678a096d45",
    ("MLP", "forward"): "31e2a1103c41814e53121eed1c03af311109c8e75901f7fa89f5389b576e8b09",
    ("MiniMaxH3Model", "_forward"): "6d35c659d448e41e8c18217ccbe8224b25c0db2f881956f8e8b52d67c7b904c5",
}
RELEASABLE = ("video_embed", "audio_embed")
_held = {}
_calls = 0


def _rows_inplace(fn, x):
    for a in range(0, x.shape[0], ROWS):
        y = fn(x[a:a + ROWS])
        x[a:a + y.shape[0]] = y
        del y
    return x


def _release_embeds():
    frame = sys._getframe(2)
    while frame is not None and not (frame.f_code.co_name == "_forward" and frame.f_globals.get("__name__") == TARGET):
        frame = frame.f_back
    if frame is None:
        return
    for name in RELEASABLE:
        t = frame.f_locals.get(name)
        if t is not None and t.is_cuda and t._base is None and t.untyped_storage().size() > 0:
            t.untyped_storage().resize_(0)


def _verify(module):
    path = module.__file__
    with open(path, encoding="utf-8") as f:
        src = f.read()
    found = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and (node.name, item.name) in EXPECTED:
                    found[(node.name, item.name)] = hashlib.sha256(ast.get_source_segment(src, item).encode()).hexdigest()
    if found != EXPECTED:
        raise SystemExit("Unsupported H3 model source; stopped: " + path)


def _patch_casts(ops):
    if not callable(getattr(ops, "cast_bias_weight", None)) or not callable(getattr(ops, "uncast_bias_weight", None)):
        raise SystemExit("Unsupported ComfyUI weight-cast API; H3 LowVRAM stopped.")
    cast, uncast = ops.cast_bias_weight, ops.uncast_bias_weight

    def held_cast(s, *args, **kwargs):
        if s not in _held:
            return cast(s, *args, **kwargs)
        if _held[s] is None:
            _held[s] = cast(s, *args, **kwargs)
        return _held[s]

    def held_uncast(s, weight, bias, offload_stream):
        if s not in _held:
            uncast(s, weight, bias, offload_stream)

    ops.cast_bias_weight, ops.uncast_bias_weight = held_cast, held_uncast

    @contextlib.contextmanager
    def hold(*layers):
        for layer in layers:
            _held[layer] = None
        try:
            yield
        finally:
            for layer in layers:
                result = _held.pop(layer)
                if result is not None:
                    uncast(layer, *result)

    return hold


def _patch(module):
    _verify(module)
    comfy = sys.modules["comfy"]
    hold = _patch_casts(comfy.ops)
    optimized_attention = module.optimized_attention
    mlp_forward, attention_forward = module.MLP.forward, module.Attention.forward

    def mlp(self, x):
        if x.shape[0] <= ROWS or comfy.model_management.in_training:
            return mlp_forward(self, x)
        with hold(self.fc1, self.fc2):
            return _rows_inplace(lambda part: mlp_forward(self, part), x)

    def attention(self, x, rope_freqs=None, transformer_options={}):
        global _calls
        s = x.shape[0]
        if s <= ROWS or rope_freqs is None or comfy.model_management.in_training:
            return attention_forward(self, x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        if _calls % LAYERS == 0:
            _release_embeds()
        heads, dim = self.heads, self.head_dim
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2

        def project(a):
            q, k, v = self.qkv_proj(x[a:a + ROWS]).split(heads * dim, dim=-1)
            n = q.shape[0]
            q, k = q.view(1, n, heads, dim), k.view(1, n, heads, dim)
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs[:, a:a + n], qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            return q[0].transpose(0, 1), k[0].transpose(0, 1), v.view(n, heads, dim).transpose(0, 1)

        with hold(self.qkv_proj, self.out_proj):
            keys = values = None
            for a in range(0, s, ROWS):
                _, k, v = project(a)
                if keys is None:
                    keys = k.new_empty((1, heads, s, dim))
                    values = v.new_empty((1, heads, s, dim))
                keys[0, :, a:a + k.shape[1]] = k
                values[0, :, a:a + v.shape[1]] = v
                del k, v
            for a in range(0, s, ROWS):
                q = project(a)[0].unsqueeze(0)
                o = optimized_attention(q, keys, values, heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
                del q
                y = self.out_proj(o.squeeze(0))
                del o
                x[a:a + y.shape[0]] = y
                del y
            del keys, values
        if _calls % LAYERS == 0 and x.is_cuda:
            import torch
            free, total = torch.cuda.mem_get_info(x.device)
            logging.info("H3 LowVRAM: %d tokens, allocated %.0f MiB, allocator peak %.0f MiB, free %.0f/%.0f MiB",
                         s, torch.cuda.memory_allocated(x.device) / 2**20, torch.cuda.max_memory_allocated(x.device) / 2**20,
                         free / 2**20, total / 2**20)
        _calls += 1
        return x

    module.MLP.forward = mlp
    module.Attention.forward = attention
    logging.info("H3 LowVRAM enabled: %d tokens/chunk, full K/V attention, current process only", ROWS)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise SystemExit("Local native H3 model source was not found; stopped.")
        spec.loader = _Loader(spec.loader)
        return spec


def install():
    if TARGET in sys.modules:
        raise SystemExit("H3 was imported before the LowVRAM hook; stopped.")
    sys.meta_path.insert(0, _Finder())

