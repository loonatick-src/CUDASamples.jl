using CUDASamples
using CUDA

export vector_add!

function vector_add!(z, x, y)
  N = length(x)
  tidx = threadIdx().x
  bidx = blockIdx().x
  bdim = blockDim().x
  idx = bdim * (bidx-1) + tidx
  # grid-strided loop
  stride = gridDim().x * bdim
  for i in idx:stride:N
    if i <= N
      z[i] = x[i] + y[i]
    end
  end
  nothing
end
