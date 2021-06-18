using CSV
using DataFrames
using Glob
using Suppressor

include("utils/utils.jl")
include("utils/myrand.jl")

# Solvergroup types
abstract type Solvertype end
struct SolvertypeDot <: Solvertype end # vec
struct SolvertypeGeMV <: Solvertype end # gemv
struct SolvertypeSpMV <: Solvertype end # spmv
struct SolvertypeGMRES <: Solvertype end # gmres
struct SolvertypeCG <: Solvertype end # cg


# Compute gflops and save as csv files for dot
function compute_gflops(folder::String, bench::String, ::Type{SolvertypeDot})

	# For all files with runtimes 
	for f_runtimes in glob("runtimes*", joinpath(folder, "csv"))
		
		# Get runtimes
		df_runtimes = CSV.read(f_runtimes, DataFrame, header=true, copycols=true)
	    
	    # Compute gflops
	    df_gflops = deepcopy(df_runtimes)
	    lrow, lcol = size(df_runtimes)
	   	for i in 1:lrow
	   		n = df_runtimes[i, lcol-4]
	   		gflop = n * 2.0 / 1.0e9
	    	for j in lcol-3:lcol
	    		df_gflops[i, j] = gflop / df_runtimes[i, j]
	    	end
	    end

	    # Save as csv
	    f_gflops = replace(f_runtimes, "runtimes" => "gflops")
	    CSV.write(f_gflops, df_gflops, delim=' ')
	end
end

# Compute gflops and save as csv files for spmv
function compute_gflops(folder::String, bench::String, ::Type{SolvertypeSpMV})

	# For all files with runtimes 
	for f_runtimes in glob("runtimes*", joinpath(folder, "csv"))
		
		# Get runtimes
		df_runtimes = CSV.read(f_runtimes, DataFrame, header=true, copycols=true)
	    
	    # Compute gflops
	    df_gflops = deepcopy(df_runtimes)
	    lrow, lcol = size(df_runtimes)
	   	for i in 1:lrow
	   		nnz = df_runtimes[i, lcol-4]
	   		gflop = nnz * 2.0 / 1.0e9
	    	for j in lcol-3:lcol
	    		df_gflops[i, j] = gflop / df_runtimes[i, j]
	    	end
	    end

	    # Save as csv
	    f_gflops = replace(f_runtimes, "runtimes" => "gflops")
	    CSV.write(f_gflops, df_gflops, delim=' ')
	end
end

# Compute gflops and save as csv files for gmres
function compute_gflops(folder::String, bench::String, ::Type{SolvertypeGMRES})

	# For all files with runtimes
	for f_runtimes in glob("*runtimes*", joinpath(folder, "csv"))

		# Get runtimes
		df_runtimes = CSV.read(f_runtimes, DataFrame, header=true, copycols=true)

		# Get info from filename
		info = f_runtimes[findlast('/', f_runtimes) + 1 : findlast(
			'.', f_runtimes) - 1]
		info = info[findfirst('_', info) + 1 : length(info)]
		info = split(info, '_')
		orth = info[3]
		precon = info[4]
		
		# Get file with flop
		f_flop = joinpath(folder, "csv", string("flop_", orth, "_", precon, ".csv"))
		
		# Check if flop count exists
		if isfile(f_flop)
			
		    # Compute gflops
		    df_flop = CSV.read(f_flop, DataFrame, header=true, copycols=true)
		    df_gflops = deepcopy(df_runtimes)
		    lrow, lcol = size(df_runtimes)
		    for i in 1:lrow
		    	if bench == "nthreads"
		    		g = df_flop[1, 4] / 1.0e9
		    	elseif bench == "size"
		    		g = df_flop[i, 4] / 1.0e9
		    	else
		    		throw(ArgumentError(bench))
		    	end
		    	for j in lcol-3:lcol
		    		df_gflops[i, j] = g / df_runtimes[i, j]
		    	end
		    end

		    # Save as csv
		    f_gflops = replace(f_runtimes, "runtimes" => "gflops")
		    CSV.write(f_gflops, df_gflops, delim=' ')
		end

	end

end


# Compute parallel speedups and save as csv files
function compute_speedups(folder::String, bench::String, ::Type{SolvertypeDot})

	# Distinguish between bench
	if bench == "nthreads"

		# Compute parallel speedups and save as csv files
		for f in glob("*runtimes*", joinpath(folder, "csv"))
			
			# Get runtimes
			df_runtimes = CSV.read(f, DataFrame, header=true, 
				copycols=true)
		    
		    # Compute speedups
		    df_speedups = deepcopy(df_runtimes)
		    for i in 2:size(df_runtimes)[1]
		    	for j in 3:size(df_runtimes)[2]
		    		df_speedups[i, j] = df_runtimes[1, j] / df_runtimes[i, j]
		    	end
		    end

		    # Save as csv
		    f_speedups = replace(f, "runtimes" => "speedups")
		    CSV.write(f_speedups, df_speedups, delim=' ')
		end
	elseif bench == "size"
		
		for kernel in ["par", "mkl"]

			# Check if serial runtimes exists
			f = glob(string("runtimes_", kernel, "_1.csv"), folder)
			if !isempty(f)

				# Get serial results
				df_runtimes_ser = CSV.read(f[1], DataFrame, header=true)

				# Iterate through all files
				for f in glob(string("runtimes_", kernel, "*"),
					joinpath(folder, "csv"))

					# Not for 1 thread
					if f != glob(string("runtimes_", kernel, "_1.csv"), 
						joinpath(folder, "csv"))[1]
						
						# Get results
						df_runtimes = CSV.read(f, DataFrame, header=true, 
							copycols=true)

						# Compute parallel speedups
						df_speedups = deepcopy(df_runtimes)
						for i in 1:size(df_runtimes)[1]
							for j in 3:size(df_runtimes)[2]
								df_speedups[i, j] = 
									df_runtimes_ser[i, j] / df_runtimes[i, j]
							end
						end

						# Save as csv
						filepath_speedups = replace(f, "runtimes" => "speedups")
						CSV.write(filepath_speedups, df_speedups, delim=' ')
					end
				end # for
			end # if
		end # for
	end # if
end
function compute_speedups(folder::String, bench::String, ::Type{SolvertypeSpMV})

	# Iterate through matrix formats
	for format in ["csr", "csc"]

		# Check if serial runtimes exists
		f_runtimes_ser = glob(string("runtimes_ser_", format, "*"), 
			joinpath(folder, "csv"))

		if !isempty(f_runtimes_ser)

			# Get serial runtimes
			df_runtimes_ser = CSV.read(f_runtimes_ser[1], DataFrame, header=true)

			# Get filenames of the csv files with parallel runtimes
			f_runtimes = vcat(
				glob(string("*runtimes_par_", format, "*"), joinpath(folder, "csv")),
				glob(string("*runtimes_mkl_", format, "*"), joinpath(folder, "csv"))
			)
			
			# Compute parallel speedups and save as csv files
			if !isempty(f_runtimes)
				for f in f_runtimes
					
					# Get runtimes
					df_runtimes = CSV.read(f, DataFrame, header=true, copycols=true)
				    
				    # Compute speedups
				    df_speedups = deepcopy(df_runtimes)
				    nrows, ncols = size(df_runtimes)
				    for i in 1:nrows
				    	for j in ncols-3:ncols
				    		df_speedups[i, j] = df_runtimes_ser[1, j] / df_runtimes[i, j]
				    	end
				    end

				    # Save as csv
				    f_speedups = replace(f, "runtimes" => "speedups")
				    CSV.write(f_speedups, df_speedups, delim=' ')
				end # for
			end # if
		end # if
	end # for
end


function compute_speedups(folder::String, bench::String, ::Type{SolvertypeGMRES})

	# Iterate through matrix formats
	for format in ["csr", "csc"]

		# Iterate through preconditioners
		for precon in ["none", "jacobi", "ilu0", "parjacobi", "parilu0"]

			# Check if serial runtimes exists (orth=mgs)
			f_runtimes_ser = string("runtimes_ser_", format, "*mgs*")
			f_runtimes_ser = glob(f_runtimes_ser, joinpath(folder, "csv"))
			if !isempty(f_runtimes_ser)

				# Get serial runtimes
				df_runtimes_ser = CSV.read(f_runtimes_ser[1], DataFrame, header=true)

				# Get filenames of the csv files with parallel runtimes
				f_runtimes_par = string("*runtimes_par_", format, "*", precon, "*")
				f_runtimes_par = glob(f_runtimes_par, joinpath(folder, "csv"))
				f_runtimes_mkl = string("*runtimes_mkl_", format, "*", precon, "*")
				f_runtimes_mkl = glob(f_runtimes_mkl, joinpath(folder, "csv"))
				
				# Check if parallel runtimes exists
				if !isempty(vcat(f_runtimes_par, f_runtimes_mkl))

					# Iterate through files with parallel runtimes
					for f in f_runtimes
						
						# Get runtimes
						df_runtimes = CSV.read(f, DataFrame, header=true, copycols=true)
					    
					    # Compute speedups
					    df_speedups = deepcopy(df_runtimes)
					    nrows, ncols = size(df_runtimes)
					    for i in 1:nrows
					    	for j in ncols-3:ncols
					    		df_speedups[i, j] = df_runtimes_ser[1, j] / df_runtimes[i, j]
					    	end
					    end

					    # Save as csv
					    f_speedups = replace(f, "runtimes" => "speedups")
					    CSV.write(f_speedups, df_speedups, delim=' ')

					end # for

				end # if

			end # if

		end # for

	end # for

end


# Run from command line
function main(folder::String)

	# Get solver and bench
	info = readlines(joinpath(folder, "info.txt"))
	solver = string(split(info[1], " = ")[2])
	bench = string(split(info[2], " = ")[2])

	# Distinguish between solver groups
	d = Dict(
		# vector
		"dot" => SolvertypeDot, "axpy" => SolvertypeDot,
		 # spmv
		"spmv" => SolvertypeSpMV,
		# gmres
		"gmres" => SolvertypeGMRES,
		# cg
		"cg" => SolvertypeCG
	)
	solvertype = get(d, solver, 0)
	if solvertype == 0
		throw(ArgumentError(solver))
	end

	# Compute GFlops
	compute_gflops(folder, bench, solvertype)

	# # Compute parallel speedup
	compute_speedups(folder, bench, solvertype)
end

main(ARGS[1])
