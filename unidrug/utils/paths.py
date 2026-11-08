import os

ASSET_ROOT = os.environ.get(
    "UNIDRUG_ASSET_ROOT",
    "/data-nxs/songlongjun/KEDD_temporal-master",
)


def expand_asset_paths(obj):
    """Replace {asset_root} in loaded configs. Override with UNIDRUG_ASSET_ROOT."""
    if isinstance(obj, dict):
        return {k: expand_asset_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_asset_paths(v) for v in obj]
    if isinstance(obj, str) and "{asset_root}" in obj:
        return obj.replace("{asset_root}", ASSET_ROOT)
    return obj
