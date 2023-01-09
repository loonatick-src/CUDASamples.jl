module CUDASamples
using CUDA


include("Utilities.jl")
include("ConceptsAndTechniques.jl")
include("introduction/vector_add.jl")

# function dump_thread_block_info()
#   tidx = threadIdx()
#   bidx = blockIdx()
#   bdim = blockDim()

#   @cuprint bidx bdim tidx
#   return nothing
# end

# @cuda threads=(32,32) blocks=2 dump_thread_block_info()

end
