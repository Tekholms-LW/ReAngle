# Publishing the node pack to the Comfy Registry

The pack lives in `comfyui-reshoot-nodes/` (folder name kept so imports and tests keep working); its registry name is **comfyui-reangle**.

1. Create a publisher at https://registry.comfy.org (sign in with GitHub). Put its id in `comfyui-reshoot-nodes/pyproject.toml` → `[tool.comfy] PublisherId` (currently a placeholder `tekholms-lw`).
2. Create a registry API key for that publisher (keep it out of the repo).
3. `pip install comfy-cli`, then from `comfyui-reshoot-nodes/`: `comfy node publish` (paste the key when asked).
   Or for CI: add the key as the GitHub secret `REGISTRY_ACCESS_TOKEN` and use Comfy's `Comfy-Org/publish-node-action`.
4. Bump `version` in `pyproject.toml` for every publish.
5. Optional: add an `Icon` URL (square PNG) and screenshots.

Not published yet. Nothing here has been pushed.
