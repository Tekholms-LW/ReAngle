"""comfyui-reshoot-nodes: drop this folder into ComfyUI/custom_nodes/."""
try:
    from .reshoot_nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # imported outside ComfyUI (e.g. by pytest rootdir collection)
    from reshoot_nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
