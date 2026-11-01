# Attribution

This package extracts the H3 chunked-inference patch developed in the CharacterStudioAI Linux experiment and subsequently ported to its Windows H3 runtime.

The attention and MLP replacement code adapts the native MiniMax H3 implementation in ComfyUI (GPL-3.0), verified against ComfyUI commit `dd79c643a95402136a75a28f6187d843bcf457ed`. See https://github.com/Comfy-Org/ComfyUI and the included GPL license. The source formulas, class and method contracts belong to their respective upstream contributors.

The package is distributed under GPL-3.0-or-later. It contains no model weights, model license grant, runtime environment, media, prompts, or private application source. The separate licenses and hardware requirements of ComfyUI, MiniMax H3, PyTorch, quantized model distributions and any LoRA still apply.

Chunking and query streaming are established techniques. No claim of inventing them, outperforming existing H3 optimizers, or guaranteeing identical floating-point bytes is made.

Related independent projects (listed as related work; their source is not included):

- https://github.com/kijai/ComfyUI-KJNodes
- https://github.com/star7code/minimax-h3-chunk-star7
- https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia

All URLs are documentation references only; this package never opens them or downloads resources.

