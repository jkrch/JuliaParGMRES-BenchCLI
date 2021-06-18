using BenchmarkTools
using ExtendableSparse
using LinearAlgebra
using Random
using SparseArrays
using Suppressor
@suppress using MatrixDepot

include("utils/myrand.jl")
include("utils/utils.jl")


# Command line arguments
nARGS = length(ARGS)
csv = ARGS[1]
bench = ARGS[2]
matrix = ARGS[3]
if bench == "nthreads"
	sizes = [parse(Int, ARGS[4])]
elseif bench == "size"
	prog = ARGS[4]
	N0 = parse(Int, ARGS[5])
	ppomag = parse(Int, ARGS[6])
	nrun = parse(Int, ARGS[7])
	sizes = get_sizes(prog, N0, ppomag, nrun)
else
	throw(ArgumentError(bench))
end	
kernel = ARGS[nARGS-3]
format = ARGS[nARGS-2]
nthreads = parse(Int, ARGS[nARGS-1])
nsamples = parse(Int, ARGS[nARGS])


# Use packages with spmv kernel
# (Did not find a better way)
if kernel == "par"
	using MtSpMV
elseif kernel == "mkl"
	using MKLSparse
elseif kernel == "test"
	include("mymul.jl")
end 


# Benchmark parameters
BenchmarkTools.DEFAULT_PARAMETERS.samples = nsamples # 10000 by default
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 24*60*60 # 5 by default


# Run benchmarks from command line
function main()

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
			print(nthreads, "..")
		else
			print(n, "..")
		end

		# Initialize matrix and vectors
		A, transA, b = get_linsys(matrix, n)
		x = rand(A.n)
		y = rand(A.m)

		# Run benchmarks and get runtimes
		if format == "csr"
		    benchoutput = @capture_out @mybtimes mul!($y, $transA, $x)
		elseif format == "csc"
		    benchoutput = @capture_out @mybtimes mul!($y, $A, $x)
		else
			throw(ArgumentError(format))
		end

		# Write number of threads, matrix properties and runtimes to output file
		open(csv, "a") do io
			s = string(nthreads, " ", A.m, " ", A.n, " ", length(A.nzval), 
				benchoutput)
			write(io, s)
		end

	end

	# Terminal output
	if bench == "size"
		println()
	end
	
end

main()
