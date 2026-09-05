# Maintainer release notes

This source repository is intentionally kept small and contains no GGUF weights or bundled CUDA binaries.

Release layout:

- source code -> GitHub repository;
- `RouteCache-win-x64-cuda-fallback.zip` -> GitHub Release asset;
- `RouteCache-win-x64-repro-tools.zip` -> GitHub Release asset for the advanced source reproducer;
- GGUF + model card + RouteCache profile + Ollama files -> Hugging Face model repository.

The maintainer's RouteCache Ultimate Publisher builds and validates these trees before upload. Do not publish the private optimizer workspace, calibration state, local manifests, absolute Windows paths, or access tokens.
