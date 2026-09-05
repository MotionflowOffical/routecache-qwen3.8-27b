# Maintainer release notes

The public source repository contains no GGUF weights or bundled CUDA binaries.

Release layout:
- source code -> GitHub repository, updated through normal Git commits;
- fallback/reproduction ZIPs -> GitHub Release assets (their SHA values are recorded in committed `RELEASE_ASSETS.json`);
- baked GGUF + model card + Ollama files -> Hugging Face model repository.

The private RouteCache Release Builder only prepares/validates staging. Remote publication is manual. Modified GitHub files are replaced through `git add -A` / `git commit` / `git push`; Hugging Face changes are made by an explicit `hf upload` Hub commit.
