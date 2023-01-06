# CUDASamples

[![Coverage](https://codecov.io/gh/loonatick-src/CUDASamples.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/loonatick-src/CUDASamples.jl)

There's a nifty set of example CUDA kernels, different kinds of optimizations etc over at the [cuda-samples](https://github.com/NVIDIA/cuda-samples) repository. Start off by porting them to `CUDA.jl`, and eventually branch out to other programming models and APIs like Metal, ROCm, KernelAbstractions etc. I realize that the repository name does not quite reflect that, but we will cross that bridge when we get there.

Before any of these extras are worked on, it might be a good idea to organize this whole project as literate programs. This combined the convenient profiling and benchmarking tools could possibly make for a great study. 

# TODO
- [ ] Start with the introductory kernels
- [ ] Set up documentation generation/literate programming frameworks

## Somewhat Advanced Goals
Replicating papers from literature on GPU algorithms, benchmarking, optimisations etc.
