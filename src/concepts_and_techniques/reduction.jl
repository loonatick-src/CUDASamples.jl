using CUDA

export reduce0
# export reduce1

# need to be extra careful when indexing
function reduce0(g_idata, g_odata)
  n = length(g_idata)
  # shared memory allocated at launch-time in launch config
  sdata = CuDynamicSharedArray(eltype(g_idata), blockDim().x)
  tid = threadIdx().x
  # load data corresponding to block into shared memory
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  sdata[tid] = (i <= n) ? g_idata[i] : 0
  sync_threads()

  #= TODO: consider creating a device-compatible `LogRange`
  ## consult the custom device structs section of the CUDA.jl manual
  =#
  s = 1
  while s < blockDim().x
    if tid % (2 * s) == 1
      sdata[tid] += sdata[tid + s]
    end
    s *= 2
    sync_threads()
  end
  if tid == 1
    g_odata[blockIdx().x] = sdata[1]
  end
  nothing
end
