# Functions to replace BLAS kernels in gmres.jl and orthogonalize.jl for 
# counting flops with GFlops.jl

import LinearAlgebra: dot, mul!, norm
import Base: *


# gmres228, orthogonalize55, orthogonalize70
function dot(x::StridedVector{Float64}, y::StridedVector{Float64})
	z = zero(Float64)
	for i=1:length(x)
		z += x[i] * y[i]
	end
	z
end


# orthogonalize14, orthogonalize42
function mul!(y::StridedVector{Float64}, A::Adjoint{<:Any, <:StridedMatrix}, x::StridedVector{Float64})
	for i in 1:length(y)
		tmp = zero(Float64)
		for j in 1:length(x)
			tmp += A[i,j] * x[j]
		end
		y[i] = tmp
	end
end

# orthogonalize26
function *(A::Adjoint{<:Any, <:StridedMatrix{Float64}}, x::StridedVector{Float64})
	y = zeros(size(A)[1])
	for i in 1:length(y)
		tmp = zero(Float64)
		for j in 1:length(x)
			tmp += A[i,j] * x[j]
		end
		y[i] = tmp
	end
	y
end

# gmres267, gmres272, orthogonalize15, orthogonalize29, orthogonalize43
function mul!(y::StridedVector{Float64}, A::StridedMatrix{Float64}, x::StridedVector{Float64}, alpha::Float64, beta::Float64)
	for i in 1:length(y)
		tmp = zero(Float64)
		for j in 1:length(x)
			tmp += alpha * A[i,j] * x[j]
		end
		y[i] = tmp + beta * y[i]
	end
end

# gmres252, orthogonalize17, orthogonalize22, orthogonalize28, orthogonalize32, 
# orthogonalize45, orthogonalize61, orthogonalize75
function norm(x::StridedVector{Float64})
	result = 0.0
	for i = 1:length(x)
		result += x[i] * x[i]
	end
	sqrt(result)
end
