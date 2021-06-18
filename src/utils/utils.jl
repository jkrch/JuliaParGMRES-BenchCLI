using BenchmarkTools, Suppressor

import BenchmarkTools: prunekwargs, hasevals

# Adapted from BenchmarkTools.btime() https://github.com/JuliaCI/BenchmarkTools.jl/blob/5f8b7f2880b68e019d8cd65317f275b1b1d1ca03/src/execution.jl#L474
# Returns the *median* (instead of the *minimum* in btime) elapsed time (in 
# seconds instead in pretty time in btime).
# No output about memory and no tuning phase.
macro mybtimes(args...)
    _, params = prunekwargs(args...)
    bench, trial, result = gensym(), gensym(), gensym()
    trialmin = gensym()
    trialmed = gensym()
    trialmean = gensym()
    trialmax = gensym()
    return esc(quote
        local $bench = $BenchmarkTools.@benchmarkable $(args...)
        $BenchmarkTools.warmup($bench)
        local $trial, $result = $BenchmarkTools.run_result($bench)
        local $trialmin = $BenchmarkTools.minimum($trial)
        local $trialmed = $BenchmarkTools.median($trial)
        local $trialmean = $BenchmarkTools.mean($trial)
        local $trialmax = $BenchmarkTools.maximum($trial)
        println(" ", $BenchmarkTools.time($trialmin)/1e9,
        		" ", $BenchmarkTools.time($trialmed)/1e9,
        		" ", $BenchmarkTools.time($trialmean)/1e9,
        		" ", $BenchmarkTools.time($trialmax)/1e9)
        $result
    end)
end

# Returns benchmark results as array of floats instead of a string 
macro mybtimes2(args...)
    _, params = prunekwargs(args...)
    bench, trial, result = gensym(), gensym(), gensym()
    trialmin = gensym()
    trialmed = gensym()
    trialmean = gensym()
    trialmax = gensym()
    btimes = gensym()
    return esc(quote
        local $bench = $BenchmarkTools.@benchmarkable $(args...)
        $BenchmarkTools.warmup($bench)
        local $trial, $result = $BenchmarkTools.run_result($bench)
        local $trialmin = $BenchmarkTools.minimum($trial)
        local $trialmed = $BenchmarkTools.median($trial)
        local $trialmean = $BenchmarkTools.mean($trial)
        local $trialmax = $BenchmarkTools.maximum($trial)
        local $btimes = [
        	$BenchmarkTools.time($trialmin)/1e9,
        	$BenchmarkTools.time($trialmed)/1e9,
        	$BenchmarkTools.time($trialmean)/1e9,
        	$BenchmarkTools.time($trialmax)/1e9
        ]
        $result, $btimes
    end)
end


# Create an array of vector lengths of size nrun
# starting with N0 and increasing in geometrica progression.
# ppomag denotes the number of 
# elements per order of magnitude.
# Try to keep full powers of 10 at once.
# From: https://github.com/j-fu/julia-tests/blob/3a3d3646e41e13b1ec23f2dea7df7eee5967c551/parallel/jlvtriad.jl#L60
function get_sizes_gp(N0::Int, ppomag::Int, nrun::Int)
    vsz = [N0]
    N = N0
    N0 *= 10
    for irun=1:nrun-1
        N = N * 10^(1.0 / ppomag)
        if (irun%ppomag == 0)
            N = N0
            N0 *= 10
        end
        push!(vsz, Int(ceil(N)))
    end
    return vsz
end

# Create an array of vector lengths of size nrun
# starting with N0 and increasing in arithmetic progression.
# ppomag denotes the number of 
# elements per order of magnitude.
# Try to keep full powers of 10 at once.
function get_sizes_ap(N0::Int, ppomag::Int, nrun::Int)
	NN = Int[]
	for i in 1:ceil(nrun/ppomag)+1
		push!(NN, Int(N0 * 10^i))
	end
    sizes = []
    s = N0
    push!(sizes, s)
    irun = 0
	for N in NN
		commdiff = N / ppomag
		for j in 1:ppomag
			if irun == nrun
				break
			end
		 	s = j*commdiff
		 	if sizes[length(sizes)] != s
				push!(sizes, Int(round(s)))
				irun += 1
			end
		end
	end
    return sizes
end

# Return sizes in geometric or arithmetic progression
function get_sizes(prog::String, N0::Int, ppomag::Int, nrun::Int)
	if prog == "gp"
		sizes = get_sizes_gp(N0::Int, ppomag::Int, nrun::Int)
	elseif prog == "ap"
		sizes = get_sizes_ap(N0::Int, ppomag::Int, nrun::Int)
	end
	return sizes
end


# Initialize the matrix, transpose the matrix and compute rhs
function get_linsys(matrix::String, n::Int)
	if n <= 0
		# SuiteSparse/MatrixMarket without rhs
		if n == 0 
		    Random.seed!(1)
		    if matrix == "sprand_sdd!/1e6"
		        A = fem2d(1000000)
		    elseif matrix == "sprand_sdd!/1e7"
		        A = fem2d(10000000)
		    elseif matrix == "sprand/1e6"
		        n = 1000000
		        A = sprand(n, n, 5/n)
		    elseif matrix == "sprand/1e7"
		        n = 10000000
		        A = sprand(n, n, 5/n)
		    else
			    A = @suppress matrixdepot(matrix)
			end
			b = A * ones(A.n)
		# SuiteSparse/MatrixMarket including rhs
		else 
			md = @suppress mdopen(matrix)
			A = md.A
			b = md.b[:, -n]
		end
	else 
		Random.seed!(1)
		# myrand.jl
		if matrix == "fem2d" 
			A = fem2d(n)
		# MatrixDepot matrix generator	
		else 
			A = @suppress matrixdepot(matrix, n)
		end
		b = A * ones(A.n)
	end
	transA = transpose(sparse(transpose(A)))
	return A, transA, b
end


# Get benchmark results from CSV file and return as arrays
function get_results(filename::String, pos::Int)
    # Read in CSV file as dataframe
    df = DataFrame!(CSV.File(filename, header=true))
    # Select nthreads/matsize and runtime columns
    select!(df, [1, pos+1])
    # Convert columns to arrays
    nthreads_or_matsize = convert(Vector{Int}, df[:, 1])
    seconds = convert(Vector{Float64}, df[:, 2])
    # Return nthreads/matsize and runtime
    nthreads_or_matsize, seconds
end
