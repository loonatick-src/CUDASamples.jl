using CUDASamples
using CUDA
using Test
# TODO: use proper RNG instead of relying on default one
using Random: shuffle

@testset "CUDASamples.jl" begin
  # Write your tests here.
  device_attributes, _ = CUDASamples.Utilities.device_query(CUDA.device())
  # 1024 on my machine
  threads_per_block = device_attributes[CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK]
  # 16
  blocks_per_sm     = device_attributes[CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR]
  # 1024
  threads_per_sm    = device_attributes[CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR]
  # 16
  sm_count          = device_attributes[CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT]
  

  # TODO: move these tests to their own folder/file and include here as more samples will be added
  @testset "`vector_add` correctness" begin
    N = 2^12  # big-ass vector
    xs_d = CUDA.fill(1.0f0, N)
    ys_d = CUDA.fill(2.0f0, N)
    zs_d = similar(xs_d)
    block_size = threads_per_block
    nblocks = Int(ceil(N/block_size))
    CUDA.@sync begin
      @cuda threads=threads_per_block blocks=nblocks vector_add!(zs_d, xs_d, ys_d)
    end
    xs = Array(xs_d)
    ys = Array(ys_d)
    zs = Array(zs_d)
    for (x, y, z) in zip(xs, ys, zs)
      @test z == x + y
    end
  end

  @testset "Concepts and Techniques" begin
    @testset "Reduction" begin
      threads_per_block = 32
      nblocks = 3  # magic number
      N = nblocks * threads_per_block - 5
      xs = 1.0f0:N |> collect |> shuffle
      xs_d = CuArray(xs)
      out_d = similar(xs_d, nblocks)
      reduce0 = CUDASamples.ConceptsAndTechniques.reduce0
      CUDA.@sync begin
        @cuda threads=threads_per_block blocks=nblocks shmem=threads_per_block*sizeof(eltype(xs)) reduce0(xs_d, out_d)
      end
      out = Array(out_d)
      @test sum(out) == sum(xs)
    end
  end

end
