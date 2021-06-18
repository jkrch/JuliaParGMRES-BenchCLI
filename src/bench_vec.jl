using BenchmarkTools
using LinearAlgebra
using Random
using Suppressor

include("utils/utils.jl")

# Command line arguments
nARGS = length(ARGS)
file = ARGS[1]
solver = ARGS[2]
bench = ARGS[3]
if bench == "nthreads"
	n = parse(Int, ARGS[4])
elseif bench == "size"
	N0 = parse(Int, ARGS[4])
	ppomag = parse(Int, ARGS[5])
	nrun = parse(Int, ARGS[6])
else
	throw(ArgumentError(bench))
end
nthreads = parse(Int, ARGS[nARGS])


# Benchmark parameters
BenchmarkTools.DEFAULT_PARAMETERS.samples = parse(Int, ARGS[length(ARGS)]) # 10000 by default
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60*60 # 5 by default


# Run benchmarks for increasing number of threads
function run_nthreads(file::String, solver::String, n::Int, nthreads::Int)

	# Initialize arrays
	Random.seed!(1)
	a = rand()
	x = rand(n)
	y = rand(n)

	# Run benchmarks and get runtimes
	if solver == "dot"
		benchoutput = @capture_out @mybtimes dot($x, $y)
	elseif solver == "axpy"
		benchoutput = @capture_out @mybtimes axpy!($a, $x, $y)
	end

	# Write benchmark runtimes to output file
	open(file, "a") do io
		write(io, string(nthreads, " ", n, benchoutput))
	end

end


# Run benchmarks for increasing vector sizes
function run_size(file::String, solver::String, N0::Int, ppomag::Int, 
	nrun::Int, nthreads::Int)
	
	# Get vector sizes
	vecsizes = get_vsizes(N0, ppomag, nrun)

	# Write header to output file
	write(file, "nthreads size min med mean max\n")

	# Terminal output
	print("size = ")

	# Run benchmarks for all vector sizes
	for n in vecsizes
		
		# Terminal output
		print(n, "..")

		# Initialize arrays
		Random.seed!(1)
		a = rand()
		x = rand(n)
		y = rand(n)

		# Run benchmarks and get runtimes
		if solver == "dot"
			benchoutput = @capture_out @mybtimes dot($x, $y)
		elseif solver == "axpy"
			benchoutput = @capture_out @mybtimes axpy!($a, $x, $y)
		end

		# Write vectorsize and benchmark runtimes to output file
		open(file, "a") do io
			write(io, string(nthreads, " ", n, benchoutput))
		end

	end

	# Terminal output
	println()

end


# Run from command line
function main()
	if bench == "nthreads"
		run_nthreads(file, solver, n, nthreads)
	elseif bench == "size"
		run_size(file, solver, N0, ppomag, nrun, nthreads)
	else
		throw(ArgumentError(bench))
	end
end

main()
