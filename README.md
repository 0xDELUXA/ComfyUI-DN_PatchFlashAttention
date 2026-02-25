# ComfyUI-DN_PatchFlashAttention

A ComfyUI custom node that patches the attention mechanism to use [Flash Attention 2](https://github.com/Dao-AILab/flash-attention), similar to how **Patch Sage Attention KJ** works in KJNodes.

## Requirements

`flash-attn` must be installed and working in your ComfyUI Python environment. Grab a prebuilt wheel matching your Python/PyTorch/CUDA versions from the [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases), or build from source (AMD):

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/
pip install ninja
$env:FLASH_ATTENTION_TRITON_AMD_ENABLE = "TRUE"
python setup.py install
```

## Installation

Clone into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/0xDELUXA/ComfyUI-DN_PatchFlashAttention
```

Then restart ComfyUI.

## Usage

Find the node under **DN > FlashAttention > Patch Flash Attention DN**.

Wire it between your model loader and sampler - the patched MODEL output is what you connect to KSampler:

```
Load Checkpoint → Patch Flash Attention DN → KSampler
```

Set `enabled` to `False` to bypass the patch and pass the model through unchanged.

## Notes

- The `--use-flash-attention` ComfyUI startup flag does not reliably force FA2 in all cases; this node guarantees it via the `optimized_attention_override` mechanism
- Requires fp16 or bf16 — fp32 inputs are automatically cast to fp16 and cast back
- Attention masks are not supported by `flash_attn_func` and will be ignored with a warning
- Tested with Flux, Qwen, and SDXL; should also work with SD1.5 and other models

## Credits

Inspired by the **Patch Sage Attention KJ** node from [KJNodes](https://github.com/kijai/ComfyUI-KJNodes).
