using BenchmarkTools
using ExtendableSparse
using GFlops
using IterativeSolvers
using LinearAlgebra
using Random
using SparseArrays
using Suppressor
@suppress using MatrixDepot

include("utils/myrand.jl")
include("utils/utils.jl")

import Base: sum


# Command line arguments
# @show ARGS
nARGS = length(ARGS)
csv = ARGS[1]
solver = ARGS[2]
bench = ARGS[3]
matrix = ARGS[4]
if bench == "nthreads" || bench == "matrix"
	sizes = [parse(Int, ARGS[5])]
elseif bench == "size"
	N0 = parse(Int, ARGS[5])
	ppomag = parse(Int, ARGS[6])
	nrun = parse(Int, ARGS[7])
	sizes = get_vsizes(N0, ppomag, nrun)
else
	throw(ArgumentError(bench))
end
kernel = ARGS[nARGS-7]
format = ARGS[nARGS-6]
orth = ARGS[nARGS-5]
precon = ARGS[nARGS-4]
nthreads = parse(Int, ARGS[nARGS-3])
if nthreads != Threads.nthreads()
	throw(ArgumentError(nthreads))
end
nsamples = parse(Int, ARGS[nARGS-2])
niter = ARGS[nARGS-1]
small_tol = parse(Bool, ARGS[nARGS])


# Use packages with spmv kernel
# (Did not find a better way)
if kernel == "par"
	using MtSpMV
elseif kernel == "mkl"
	using MKLSparse
elseif kernel == "test"
	include("mymul.jl")
elseif kernel == "hist"	
	using MtSpMV
elseif kernel == "flop"
	include("utils/custom_blas.jl")
end

# Benchmark parameters
BenchmarkTools.DEFAULT_PARAMETERS.samples = nsamples # 10000 by default
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 24*60*60 # 5 by default


# Returns solver function
function get_solver()
	usablesolvers = Dict(
		"cg" => cg, 
	    "minres" => minres, 
	    "gmres" => gmres, 
	    "bicgstabl" => bicgstabl
	)
	if haskey(usablesolvers, solver)
		solverfunc = usablesolvers[solver]
	else
		throw(ArgumentError(solver))
	end
	return solverfunc
end

# Returns type of orthogonalization method (for gmres)
function get_orth()
	usableorth = Dict(
		"mgs" => ModifiedGramSchmidt(),
	    "cgs" => ClassicalGramSchmidt(),
	    "dgks" => DGKS()
	)
	if haskey(usableorth, orth)
		orth_meth = usableorth[orth]
	else
		throw(ArgumentError(orth))
	end
	return orth_meth
end

# Returns preconditioner
function get_precon()
	usableprecon = Dict(
		"jacobi" => JacobiPreconditioner,
	    "ilu0" => ILU0Preconditioner,
	    "parjacobi" => ParallelJacobiPreconditioner,
	    "parilu0" => ParallelILU0Preconditioner
	)
	if haskey(usableprecon, precon)
		preconfunc = usableprecon[precon]
	else
		throw(ArgumentError(precon))
	end
	return preconfunc
end


# Sum all types of flop of a counter
function sum(cnt::GFlops.Counter)
	cnt32 = cnt.fma32 + cnt.add32 + cnt.sub32 + cnt.mul32 + cnt.div32 + cnt.sqrt32
	cnt64 = cnt.fma64 + cnt.add64 + cnt.sub64 + cnt.mul64 + cnt.div64 + cnt.sqrt64
	return cnt32 + cnt64
end

# Use costum BLAS kernels to compute gflop
function flop_count()

	# Solver
	solverfunc = get_solver()

	# Orthogonalization method (for gmres)
	if solver == "gmres"
		orth_meth = get_orth()
	end

	for n in sizes
		
		# Linear system
		A, transA, b = get_linsys(matrix, n)

		# Number of iterations and tolerance
		maxiter = size(A, 2)
		if niter != "default"
		 	maxiter = parse(Int, niter)
		end
		reltol = sqrt(eps(real(eltype(b))))
		if small_tol == true
			reltol = 1e-99
		end

		# Bench
		cnt_precon = 0
		cnt_reorder = 0
		cnt_solver = 0
		if precon == "none"
			if solver == "gmres"
				cnt_iter = @count_ops solverfunc($transA, $b, maxiter=$maxiter,
					reltol=$reltol, orth_meth=$orth_meth)[2]
			else
				cnt_iter = @count_ops solverfunc($transA, $b, maxiter=$maxiter,
					reltol=$reltol)[2]
			end
		else
			if precon == "parilu0"
				Random.seed!(1)
				cnt_reorder = @count_ops reorderlinsys(A, b)
				Random.seed!(1)
				A, b = reorderlinsys(A, b)
			end
			preconfunc = get_precon()
			P = preconfunc(A)
			cnt_precon = @count_ops preconfunc($A)
			if solver == "gmres"
				cnt_iter = @count_ops solverfunc($transA, $b, Pl=$P, 
					maxiter=$maxiter, reltol=$reltol, orth_meth=$orth_meth)[2]
			else
				cnt_iter = @count_ops solverfunc($transA, $b, Pl=$P, 
					maxiter=$maxiter, reltol=$reltol)[2]
			end
		end

		# Write flop of solver to csv file
		csvv = replace(csv, "flop" => "flop-solver")
		s = string(A.m, " ", A.n, " ", length(A.nzval), " ", flop_solver, "\n")
		open(csvv, "a") do io
			write(io, s)
		end

		# Write flop of preconditioner creation to csv file
		if precon != "none"
			csvv = replace(csv, "flop" => "flop-precon")
			s = string(A.m, " ", A.n, " ", length(A.nzval), " ", flop_precon, "\n")
			open(csvv, "a") do io
				write(io, s)
			end
		end

		# # Write flop of reordering to csv file
		# if precon == "parilu0"
		# 	csvv = replace(csv, "flop" => "flop-reorder")
		# 	s = string(A.m, " ", A.n, " ", length(A.nzval), " ", flop_reorder, "\n")
		# 	open(csvv, "a") do io
		# 		write(io, s)
		# 	end
		# end

	end
end


# Single solve for convergence history
function conv_hist()

	# Solver
	solverfunc = get_solver()

	# Orthogonalization method (for gmres)
	if solver == "gmres"
		orth_meth = get_orth()
	end

	for n in sizes
		
		# Linear system
		A, transA, b = get_linsys(matrix, n)

		# Number of iterations and tolerance
		maxiter = size(A, 2)
		if niter != "default"
		 	maxiter = parse(Int, niter)
		end
		reltol = sqrt(eps(real(eltype(b))))
		if small_tol == true
			reltol = 1e-99
		end

		# Bench
		if precon == "none"
			if solver == "gmres"
				ch = solverfunc(transA, b, maxiter=maxiter, reltol=reltol,
					orth_meth=orth_meth, log=true)[2]
			else
				ch = solverfunc(transA, b, maxiter=maxiter, reltol=reltol, 
					log=true)[2]
			end
		else
			if precon == "parilu0"
				Random.seed!(1)
				P = ParallelILU0Preconditioner(A)
				coloring = P.coloring
				c = collect(Iterators.flatten(coloring))
				A = A[c,:][:,c]
				b = b[c]
				transA = transpose(sparse(transpose(A)))
			else
				preconfunc = get_precon()
				P = preconfunc(A)
			end
			if solver == "gmres"
				ch = solverfunc(transA, b, Pl=P, maxiter=maxiter, reltol=reltol,
					orth_meth=orth_meth, log=true)[2]
			else
				ch = solverfunc(transA, b, Pl=P, maxiter=maxiter, reltol=reltol,
					log=true)[2]
			end
		end

		# Write results to csv file
		s = string(A.m, " ", A.n, " ", length(A.nzval), " ",
				   ch.isconverged, " ", ch.iters, " ", nrests(ch), " ", nprods(ch), 
				   " ", ch[:abstol], " ", ch[:reltol], "\n")
		open(csv, "a") do io
			write(io, s)
		end

	end

end


# Run benchmarks
function run_benchmark()

	# Write header to output file
	if bench == "size"
		write(csv, "nthreads nrows ncols nnz min med mean max\n")
	end

	# Terminal output
	if bench == "size"
		print("size = ")
	end

	# Run benchmarks for all sizes
	for n in sizes
		
		# Terminal output
		if bench == "nthreads"
			if kernel != "hist"
				print(nthreads, "..")
			end
		elseif bench == "size"
			print(n, "..")
		end

		# Get linear system
		A, transA, b = get_linsys(matrix, n)

		# Number of iterations and tolerance
		maxiter = size(A, 2)
		if niter != "default"
		 	maxiter = parse(Int, niter)
		end
		reltol = sqrt(eps(real(eltype(b))))
		if small_tol == true
			reltol = 1e-99
		end

		# Orthogonalization method (for gmres)
		if solver == "gmres"
			orth_meth = get_orth()
		end

		# Run benchmarks and get runtimes
		solverfunc = get_solver()

		# Bench
		btimes_solver = zeros(4)
		btimes_precon = zeros(4)
		btimes_reorder = zeros(4)
		if precon == "none"
			if solver == "gmres"
				if format == "csr"
					x, btimes_solver = @mybtimes2 $solverfunc($transA, $b,
						maxiter=$maxiter, reltol=$reltol, orth_meth=$orth_meth)
				elseif format == "csc"
					x, btimes_solver = @mybtimes2 $solverfunc($A, $b,
						maxiter=$maxiter, reltol=$reltol, orth_meth=$orth_meth)
				else
					throw(ArgumentError(format))
				end
			else
				if format == "csr"
					x, btimes_solver = @mybtimes2 $solverfunc($transA, $b,
						maxiter=$maxiter, reltol=$reltol)
				elseif format == "csc"
					x, btimes_solver = @mybtimes2 $solverfunc($A, $b,
						maxiter=$maxiter, reltol=$reltol)
				else
					throw(ArgumentError(format))
				end
			end
		else
			if precon == "parilu0"
				Random.seed!(1)
				P, btimes_precon = @mybtimes2 ParallelILU0Preconditioner($A)
				coloring = P.coloring
				c = collect(Iterators.flatten(coloring))
				A = A[c,:][:,c]
				b = b[c]
				transA = transpose(sparse(transpose(A)))
			else
				preconfunc = get_precon()
				P, btimes_precon = @mybtimes2 $preconfunc($A)
			end
			if solver == "gmres"
				if format == "csr"
					x, btimes_solver = @mybtimes2 $solverfunc($transA, $b, Pl=$P,
						maxiter=$maxiter, reltol=$reltol, orth_meth=$orth_meth)
				elseif format == "csc"
					x, btimes_solver = @mybtimes2 $solverfunc($A, $b, Pl=$P,
						maxiter=$maxiter, reltol=$reltol, orth_meth=$orth_meth)
				else
					throw(ArgumentError(format))
				end
			else
				if format == "csr"
					x, btimes_solver = @mybtimes2 $solverfunc($transA, $b, Pl=$P, 
						maxiter=$maxiter, reltol=$reltol)
				elseif format == "csc"
					x, btimes_solver = @mybtimes2 $solverfunc($A, $b, Pl=$P,
						maxiter=$maxiter, reltol=$reltol)
				else
					throw(ArgumentError(format))
				end
			end
		end

		# Set first column for csv files
		if bench == "matrix"
			firstcol = matrix
		else
			firstcol = nthreads
		end

		# Write runtimes of solver to csv file
		btimes = btimes_solver
		csvv = replace(csv, "runtimes" => "runtimes-solver")
		open(csvv, "a") do io
			s = string(firstcol, " ", A.m, " ", A.n, " ", length(A.nzval), 
				" ", btimes[1], " ", btimes[2], " ", btimes[3], " ", btimes[4], "\n")
			write(io, s)
		end

		# Write runtimes of preconditioner creation to csv file
		if precon != "none"
			btimes = btimes_precon
			csvv = replace(csv, "runtimes" => "runtimes-precon")
			open(csvv, "a") do io
				s = string(firstcol, " ", A.m, " ", A.n, " ", length(A.nzval), 
					" ", btimes[1], " ", btimes[2], " ", btimes[3], " ", btimes[4], "\n")
				write(io, s)
			end
		end

		# # Write runtimes of reordering to csv file
		# if precon == "parilu0"
		# 	btimes = btimes_reorder
		# 	csvv = replace(csv, "runtimes" => "runtimes-reorder")
		# 	open(csvv, "a") do io
		# 		s = string(nthreads, " ", A.m, " ", A.n, " ", length(A.nzval), 
		# 			" ", btimes[1], " ", btimes[2], " ", btimes[3], " ", btimes[4], "\n")
		# 		write(io, s)
		# 	end
		# end

	end

	# Terminal output
	if bench == "size"
		println()
	end
end


# Run from command line
function main()
	if kernel == "hist"
		conv_hist()
	elseif kernel == "flop"
		flop_count()
	elseif any(x->x==kernel, ["ser", "par", "mkl"]) 
		run_benchmark()
	else
		throw(ArgumentError(kernel))
	end
end
main()
