
# cmpext3 - PyTorch/ComfyUI CUDA Extension

This is a PyTorch/ComfUI extension to bypass FFMA/Tensor cores throttling on CMP mining cards based on Turing chips (TU10X).
Forked from [eastmoe/cmp_ext](https://github.com/eastmoe/cmp_ext) - the original code targeted CMP 170HX (Ampere GPU).
Supports FP16 and FP32, BF16 paths has been removed.

For additional info - go to the original repository.

# Warning

Shamelessly vibecoded (using Claude.ai) to make it run on Turing cards. Can contain errors. Can break the output.
Tested on CMP 50HX card and text-to-image models (SDXL & Anima), maxing out the power usage on this card.
Speed is at least tripled in comparison to the normal FP16 workloads.

# Compilation

For additional info - go to the original repository.
Release page contains a pre-compiled wheel for CUDA 12.8 + Python 3.12

# License

MIT
