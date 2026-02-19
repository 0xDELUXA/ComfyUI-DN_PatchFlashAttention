import torch
import logging
from comfy.ldm.modules.attention import wrap_attn


def get_flash_attn_func():
    from flash_attn import flash_attn_func

    @wrap_attn
    def attention_flash(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        in_dtype = v.dtype

        # flash_attn requires fp16 or bf16; cast fp32 to fp16
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)

        if skip_reshape:
            # q/k/v already (b, heads, seq, dim_head)
            b, _, seq, dim_head = q.shape
            # flash_attn expects (b, seq, heads, dim_head)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        else:
            # q/k/v are (b, seq, heads*dim_head)
            b, seq, _ = q.shape
            dim_head = q.shape[-1] // heads
            q = q.view(b, seq, heads, dim_head)
            k = k.view(b, seq, heads, dim_head)
            v = v.view(b, seq, heads, dim_head)

        if mask is not None:
            logging.warning("PatchFlashAttentionDN: attention mask ignored (not natively supported by flash_attn_func).")

        # flash_attn_func: (b, seq, heads, dim_head) -> (b, seq, heads, dim_head)
        out = flash_attn_func(q, k, v, causal=False)

        if skip_reshape:
            if skip_output_reshape:
                # caller wants (b, heads, seq, dim_head)
                out = out.transpose(1, 2)
            else:
                out = out.reshape(b, seq, heads * dim_head)
        else:
            if skip_output_reshape:
                # caller wants (b, heads, seq, dim_head)
                out = out.transpose(1, 2)
            else:
                out = out.reshape(b, seq, heads * dim_head)

        return out.to(in_dtype)

    return attention_flash


class PatchFlashAttentionDN():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Flash Attention 2 patch. Set to False to pass the model through unchanged."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = (
        "Patches ComfyUI attention to use Flash Attention 2 (flash_attn). "
        "Requires the flash-attn package. Wire the output MODEL into your sampler."
    )
    EXPERIMENTAL = True
    CATEGORY = "DN/FlashAttention"

    def patch(self, model, enabled):
        if not enabled:
            return (model,)

        try:
            new_attention = get_flash_attn_func()
        except ImportError:
            raise RuntimeError(
                "flash_attn is not installed. "
                "Install it from https://github.com/Dao-AILab/flash-attention/releases "
                "or with: pip install flash-attn --no-build-isolation"
            )

        model_clone = model.clone()

        def attention_override_flash(func, *args, **kwargs):
            return new_attention.__wrapped__(*args, **kwargs)

        model_clone.model_options["transformer_options"]["optimized_attention_override"] = attention_override_flash

        logging.info("PatchFlashAttentionDN: Flash Attention 2 patch applied.")
        return (model_clone,)
