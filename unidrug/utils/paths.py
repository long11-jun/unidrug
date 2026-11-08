import os

ASSET_ROOT = os.environ.get("UNIDRUG_ASSET_ROOT", "")


def expand_asset_paths(obj):
    """Replace {asset_root} in loaded configs. Override with UNIDRUG_ASSET_ROOT."""
    if isinstance(obj, dict):
        return {k: expand_asset_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_asset_paths(v) for v in obj]
    if isinstance(obj, str) and "{asset_root}" in obj:
        if not ASSET_ROOT:
            raise RuntimeError(
                "Set UNIDRUG_ASSET_ROOT to the directory that holds the pretrained weights and BMKG."
            )
        return obj.replace("{asset_root}", ASSET_ROOT)
    return obj
