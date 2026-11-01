# MiniMax H3 LowVRAM

**A small, experimental, offline launcher that reduces native ComfyUI H3 activation memory with 1024-token chunks.**

**[Download the Windows setup ZIP](https://github.com/jiangzechunwnl-source/ComfyUI-H3-LowVRAM/releases/download/v0.1.0/ComfyUI-H3-LowVRAM-v0.1.0.zip)** | [All releases](https://github.com/jiangzechunwnl-source/ComfyUI-H3-LowVRAM/releases)

For people whose H3 sampling runs out of VRAM. Full K/V attention is preserved: this is not a local-window attention approximation. No extra Python packages, model downloads, account or cloud service are added.

> **Compatibility first:** v0.1.0 supports one verified H3 source baseline, not every ComfyUI release. It checks source fingerprints before launch and stops with an error on an unsupported version. This is an early experimental release, not a universal low-VRAM fix.

## Windows: simple setup

You need an **existing, working ComfyUI installation with local H3 models**. This download does not include ComfyUI, Python, PyTorch or model weights.

1. Download the release ZIP and extract it. Do not run it from inside the ZIP.
2. Double-click **`Install-Windows.cmd`**.
3. In the file picker, select **`main.py` inside your ComfyUI folder**.
4. Close your normal ComfyUI. Double-click the new **`Start-H3-LowVRAM.cmd`** in your installation folder.
5. When the server is ready, open **http://127.0.0.1:8188** and load your existing **native H3 workflow**.

Setup recognizes Windows portable Python and common local `.venv` / `venv` layouts. It only copies this tool and creates a separate launcher. It does not change your ComfyUI source, original launcher, models or workflows. Reinstalling backs up previous tool files.

If Windows refuses the setup script, do not disable system protections. Use the manual command below or review the script with someone you trust. Setup does not request administrator access or change execution policy.

**To undo:** stop this launcher and use your original ComfyUI launcher. Optionally remove only the added `H3-LowVRAM` folder and `Start-H3-LowVRAM.cmd`.

## Manual launch: Windows or Linux

Use the Python environment that already runs your ComfyUI. Replace the example paths with your own local paths:

```sh
python run.py --comfy-root /path/to/ComfyUI --check
python run.py --comfy-root /path/to/ComfyUI
```

On Windows portable, call your existing `python_embeded\python.exe` instead of `python`. The included `Start-Windows-Portable.cmd` also works when this extracted folder sits beside `ComfyUI` and `python_embeded`.

The server binds to **127.0.0.1:8188**. Use `--port 8189` if needed. `--extra-packages /local/folder` can select an existing H3 package overlay; nothing is installed or downloaded.

## Supported baseline

- Native ComfyUI MiniMax H3 implementation from commit **`dd79c643a95402136a75a28f6187d843bcf457ed`**.
- Exact `Attention.forward`, `MLP.forward` and `MiniMaxH3Model._forward` fingerprints are embedded in `h3_low_vram.py`.
- Extraction uses the same computational patch tested locally on an **RTX 5060 Laptop 8 GB**, with Linux PyTorch 2.9.1 / CUDA 12.8 and Windows PyTorch 2.13.0 / CUDA 13.0.
- Do not infer support for every quantization, older GPU, AMD/Intel GPU, training, concurrent inference, or other H3 patch.

If your source is unsupported, keep your working ComfyUI intact. This tool will not downgrade it or automatically fetch another checkout. New compatibility baselines need their own validation.

## What it does

1. Projects QKV and computes MLP in **1024-token chunks**, writing outputs back to consumed input rows.
2. Builds complete K/V first, then recomputes Q in chunks against **all** K/V. Full Q is never retained.
3. Reuses each layer's converted / LoRA-merged weights across its chunks.
4. Releases video/audio embedding storage after those embeddings have been copied into the packed sequence.

It preserves the full-attention computation graph mathematically. Different chunk shapes and backends may still produce floating-point differences; **bit-for-bit output and unchanged image quality are not guaranteed**. It recomputes projections, so a memory improvement can cost time. It is not advertised as a speed-up.

## How much VRAM can it save?

In the original Linux experiment, an H3 INT8 + Turbo request at **768 x 1344, 10 seconds, with audio** previously ran out of VRAM on an 8 GB GPU. With this patch, sampling proceeded and the logged PyTorch allocator peak was **4655 MiB**. The owner subsequently confirmed that larger-resolution generation worked.

That allocator statistic is **not total GPU memory**, a universal requirement, or a standalone-package end-to-end benchmark. Windows small-model numerical comparisons passed, but a complete Windows maximum-canvas video benchmark was not recorded at release. See [VALIDATION.md](VALIDATION.md).

Model weights, complete K/V, text encoding, VAE decoding, system RAM and the display driver still need resources. **A 4 GB card is not guaranteed to run H3.** Start with a shorter, smaller request and increase gradually. The patch's results depend on model, quantization, backend, frame count and available memory.

## Deliberately isolated

This first release runs native ComfyUI nodes with `--disable-all-custom-nodes`, `--disable-api-nodes`, `--disable-auto-launch`, `--lowvram`, `--cache-none` and a 1 GiB VRAM reservation. Existing third-party custom-node workflows may report missing nodes. Use a native H3 workflow for this release.

External connections, external DNS and child-process launches are denied by this launcher; common ML clients are configured offline and inherited proxies are neutralized. Required assets must already be local. Some unrelated workflows that need external processes will not work here. This is an application guard, not a replacement for OS-level network isolation.

Do **not** stack this with KJNodes H3 optimization, LongMedia, Sage/Sol patches or other model/weight-cast replacements. Normal and patched launches are separate processes. Restart normally to turn the patch off.

## Troubleshooting

| Message or symptom | What to do |
| --- | --- |
| Unsupported H3 source fingerprint | This ComfyUI version is not validated. Use your normal launcher; do not remove the fingerprint check. |
| Existing ComfyUI Python not found | Use your normal ComfyUI Python with the manual command. No Python was installed by Setup. |
| Port already in use | Close normal ComfyUI, or choose another `--port`. |
| Missing custom nodes | Use a native H3 workflow; custom nodes are intentionally disabled in v0.1.0. |
| Missing model or package | Supply the required files locally using your existing setup. This tool does not download them. |
| Still out of memory | Reduce resolution, duration or batch size. Other phases may be the limiting factor. |

For bug reports, include version numbers, GPU/VRAM, model variant, resolution, duration and a minimal sanitized error. Do not post private prompts, media, credentials or entire environment dumps.

## License and related work

GPL-3.0-or-later; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). No model weights or model license are included. Token chunking and query streaming are established techniques. Related H3 optimizers include [KJNodes](https://github.com/kijai/ComfyUI-KJNodes), [chunk-star7](https://github.com/star7code/minimax-h3-chunk-star7) and [LongMedia](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia). This package does not claim novelty or superiority over them.
