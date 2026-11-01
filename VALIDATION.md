# Validation status for v0.1.0

## Inherited computational checks

The patch was extracted from a locally tested H3 launcher. Local source fingerprints match the embedded checks. On Windows RTX 5060 Laptop with PyTorch 2.13.0+cu130, the original and chunked Attention/MLP were compared for:

- CPU FP32, CUDA FP32 and CUDA BF16.
- 1024, 1025 and 2051 tokens (boundary, partial final chunk and multiple chunks).
- Four weight casts per Attention+MLP combination, unchanged by chunk count.
- The unmodified no-RoPE attention path.
- Releasing disposable copied CUDA embedding storage without damaging the packed sequence.
- Rejecting changed source fingerprints and late hook installation.

All nine precision/shape combinations passed their dtype-specific tolerances. These are small-layer checks, not full checkpoint or final-video comparisons. The embedded patch is changed only in English-facing messages and its header during extraction.

## Original Linux observation

768 x 1344, 10 seconds, audio, INT8, Turbo 8 steps, RTX 5060 Laptop 8 GB: sampling proceeded after prior out-of-memory failures; the logged PyTorch allocator peak reached 4655 MiB. The owner reported successful larger-resolution use. The recorded first step took about 500 seconds. Do not interpret this as a total-device-memory figure, an identical-output guarantee, a speed gain or a test of all 8 GB GPUs.

## Distribution checks

The release includes standard-library contract tests, a source compatibility check, English-only text scanning, an explicit file allowlist and SHA256 hashes. Installer copying and generated launch paths are checked in a disposable local fixture. No private prompts, application databases, logs, images, videos, model weights or credentials are included.

The exported English-only patch was rechecked against the real local Attention/MLP using the same CPU/CUDA precision and shape combinations; it passed. Seven standard-library contract tests passed. Windows setup and reinstallation passed a disposable portable-layout fixture with spaces in its path, preserved original source sentinel files, and produced a recoverable previous-tool backup.

The standalone launcher also started the supported native ComfyUI backend successfully on a separate loopback port; `/system_stats` responded and `/object_info` included the native H3 conditioning and model loader nodes. No full checkpoint was loaded and no video was generated in that startup test.

The standalone launcher and installer remain an experimental integration. Full standalone video generation, other machines, other GPU architectures, cancellation under a loaded full checkpoint, every quantization backend, interactive file-picker user acceptance, and user acceptance remain unverified.

## Reproduce

Using your existing Python, run `python test_contract.py` from this extracted directory. With a supported local ComfyUI installation, run `python run.py --comfy-root /path/to/ComfyUI --check`. The check does not import torch, run inference or start a server.
