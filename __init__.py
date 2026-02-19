from .patch_flash_attention import PatchFlashAttentionDN

NODE_CLASS_MAPPINGS = {
    "PatchFlashAttentionDN": PatchFlashAttentionDN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PatchFlashAttentionDN": "Patch Flash Attention DN",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
