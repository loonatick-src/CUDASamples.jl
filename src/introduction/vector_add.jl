using CUDA

function vector_add!(z, x, y)
  tidx = threadIdx().x
  bidx = blockIdx().x
  bdim = blockDim().x
  n = min(length.(z, x, y))
  i = bdim * bidx + tidx
  if (i < n)
    z[i] = x[i] + y[i]
  end
  nothing
end



@testset "`vector_add` correctness" begin
  const N = 2^20
  xs = CUDA.fill(1.0f0, N)
  ys = CUDA.fill(2.0f0, N)
  zs = CuArray{eltype(xs)}(undef, N)
  CUDA.@sync begin
    @cuda threads=256 vector_add!(zs, xs, ys)
  end
  @test
end

