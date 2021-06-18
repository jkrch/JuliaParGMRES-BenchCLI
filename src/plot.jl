# Solves problems with Plots.jl on ssh servers
# (https://github.com/JuliaPlots/Plots.jl/issues/1905#issuecomment-458778817)
ENV["GKSwstype"]="100"

using CSV
using DataFrames
using DelimitedFiles
using Glob
using Plots


# Command line arguments
nARGS = length(ARGS)
folder = ARGS[1]
if nARGS == 1
	backend == gr
else
	backend = ARGS[2]
end


# Choose backend for Plots (GR by default)
outputformats = ["png"]
if nARGS == 1
	gr()
else
	if backend == "gr"
		gr()
	elseif backend == "plotly"
		plotly()
	elseif backend == "pyplot"
		pyplot()
	elseif backend == "pgfplotsx"
		outputformats = ["png", "tex", "tikz"]
		pgfplotsx()
	end
end


# Solvergroup types
abstract type Solvergroup end
struct Solvergroup1 <: Solvergroup end # vec
struct Solvergroup2 <: Solvergroup end # spmv
struct Solvergroup3 <: Solvergroup end # iter


# Create subfolders for all benchmark estimates
function create_subfolders(folder::String)
	for i in outputformats	
		if !isdir(joinpath(folder, i))
			mkdir(joinpath(folder, i))
			# for j in ["min", "med", "mean", "max"]
			# 	mkdir(joinpath(folder, i, j))
			# end
		end
	end
end


# Create and return plot title
function get_plottitle(folder::String, solver::String, bench::String, 
					   ::Type{Solvergroup1})
	# Create title with solver
	title = solver

	# Add vector size to title
	if bench == "nthreads"
		df = DataFrame!(CSV.File(glob("runtimes*", joinpath(folder, "csv"))[1], 
			header=true))
		title = string(title, ", n=", df[1,2])
	end

	return title
end
function get_plottitle(folder::String, solver::String, bench::String,
					   ::Type{Solvergroup2})
	# Create title with solver
	title = solver

	# Add matrix to title
	info = readlines(joinpath(folder, "info.txt"))
	matrix = string(split(info[3], " = ")[2])
	title = string(title, ", ", matrix)

	# Add matrix size to title for non SuiteSparse/MatrixMarket
	if bench == "nthreads"
		if !occursin("/", matrix)
			df = DataFrame!(CSV.File(glob("runtimes*", joinpath(folder, "csv"))[1], 
				header=true))
			title = string(title, ", n=", df[1,2])
		end
	end

	return title
end
function get_plottitle(folder::String, solver::String, bench::String,
					   ::Type{Solvergroup3})
	# Create title with solver
	title = solver

	# Add additional info inf bench is not matrix
	if bench != "matrix"

		# Add matrix to title
		info = readlines(joinpath(folder, "info.txt"))[3]
		matrix = split(info, " = ")[2]
		title = string(title, ", ", matrix)

		# Add matrix size to title for non SuiteSparse/MatrixMarket
		if bench == "nthreads"
			if !occursin("/", matrix)
				df = DataFrame!(CSV.File(glob("runtimes*", joinpath(folder, "csv"))[1], 
					header=true))
				title = string(title, ", n=", df[1,2])
			end
		end

	end

	return title
end


# Create plots
function create_plots(folder::String)

	# Get solver and bench
	info = readlines(joinpath(folder, "info.txt"))
	solver = string(split(info[1], " = ")[2])
	bench = string(split(info[2], " = ")[2])

	# Get progession for bench == size
	if bench == "size"
		prog = split(split(info[4], " = ")[2], " ")[1]
	end

	# Distinguish between solver groups
	d = Dict(
		# vector
		"dot" => Solvergroup1, "axpy" => Solvergroup1,
		 # spmv
		"spmv" => Solvergroup2,
		# iter
		"cg" => Solvergroup3, "minres" => Solvergroup3, 
		"gmres" => Solvergroup3, "bicgstabl" => Solvergroup3
	)
	solvergroup = get(d, solver, 0)
	if solvergroup == 0
		throw(ArgumentError(solver))
	end

	# Dictionary to convert command line arguments to label parts
	# for all solvergroups
	labels1 = Dict(
		# kernels
		"par" =>  "OpenBLAS", "mkl" => "MKL.jl"
	)
	labels2 = Dict(
		# kernels
		"ser" => "SparseArrays", "par" =>  "MtSpMV.jl", "mkl" => "MKLSparse.jl",
		# formats
		"csr" => ", CSR", "csc" => ", CSC"
	)
	labels3 = Dict(
		# kernels
		"ser" => "SparseArrays & OpenBLAS", "par" =>  "MtSpMV.jl & OpenBLAS", 
		"mkl" => "MKLSparse.jl & MKL.jl",
		# formats
		"csr" => ", CSR", "csc" => ", CSC",	
		# orth
		"mgs" => ", MGS", "cgs" => ", CGS", "dgks" => ", DGKS",
		# precon
		"none" => "", "jacobi" => ", Jacobi", "ilu0" => ", ILU0",
		"parjacobi" => ", Parallel Jacobi", "parilu0" => ", Parallel ILU0",
	)
	labels = Dict(Solvergroup1 => labels1, Solvergroup2 => labels2, 
				  Solvergroup3 => labels3)

   	# General plot parameter
   	metrics = ["runtimes", "gflops", "speedups"]
   	ylabels = ["Runtime in seconds", "GFlops", "Parallel speedup"]
   	if bench == "nthreads"
   		xaxis = :none
   		xlabel = "Number of threads"
   	elseif bench == "size"
   		if prog == "gp"
   			xaxis = :log10
   		elseif prog == "ap"
   			xaxis = :none
   		end
   		if solvergroup == Solvergroup1
			xlabel = "Vector size"
		else
			xlabel = "Matrix size"
		end
   	end

   	# Position of runtime in csv file
   	ix = Dict("nthreads"=>1, "size"=>2, "matrix"=>1)
   	iy = Dict(Solvergroup1=>3, Solvergroup2=>5, Solvergroup3=>5)

   	# Create subfolders for plots
   	create_subfolders(folder)

   	# Get plottitle
   	title = get_plottitle(folder, solver, bench, solvergroup)

   	# Plot for all benchmark estimates
   	for (i, benchestimate) in enumerate(["min", "med"])

   		# Plot for all benchmark types
   		for (metric, ylabel) in zip(metrics, ylabels)

			# Get all csv files for metric in sorted order
			files_ser = glob(string(metric, "*ser*"), joinpath(folder, "csv"))
			files_par = glob(string(metric, "*par*"), joinpath(folder, "csv"))
			files_mkl = glob(string(metric, "*mkl*"), joinpath(folder, "csv"))
			files = vcat(files_ser, files_par, files_mkl)

			# Check if results for metric exists
			if !isempty(files)

				# Create plot
				if bench != "matrix"
					plot(
						title=title,
						xaxis=xaxis,
						xlabel=xlabel,
						ylabel=ylabel,
						dpi=300,
						legend=:outertopright,
						framestyle=:box			
					)
				else
					matx = String[]
					maty = Float64[]
				end

				# Add all results to plot
				for (ifile, file) in enumerate(files)

					# Get benchmark results
				    df = DataFrame!(CSV.File(file, header=true))
				    x = convert(Vector, df[!, ix[bench]])
				    y = convert(Vector, df[!, iy[solvergroup]+i-1])

				    # Create label from filename
				    infos = file[findlast('/', file) + 1 : findlast('.', file) - 1]
				    infos = infos[findfirst('_', infos) + 1 : length(infos)]
				    infos = split(infos, "_")
				    label = ""
				    for (j, info) in enumerate(infos)
				    	if bench == "nthreads"
				    		label = string(label, labels[solvergroup][info])
				    	elseif bench == "size"
					    	if j == length(infos)
				    			label = string(label, ", ", info, " thread(s)")
					    	else
								label = string(label, labels[solvergroup][info])
							end
						elseif bench == "matrix"
							if j == 1
								title = string(solver, ", ")
								info = replace(info, "--" => "/")
								# info = replace(info, "-" => "\_")
								push!(matx, info)
								if typeof(y[1]) == Float64 
									push!(maty, y[1])
								else 
									push!(maty, parse(Float64, y[1]))
								end
							elseif j == length(infos)
							    title = string(title, ", ", info, " thread(s)")
					    	else
								title = string(title, labels[solvergroup][info])
							end   
						end
					end

					# Add to plot
					if bench != "matrix"
						if length(x) == 1
							scatter!(x, y, label=label, markershape=:xcross)
						else
							plot!(x, y, label=label)
						end
					end

				end # for

				# Create bar plot if bench is matrix
				if bench == "matrix"
					println(matx)
					println(maty)
					bar(matx, maty, xrotation=60, ylabel=ylabel, legend=:none)
				end

				# Save in all given outputformats
				for outputformat in outputformats
					outputfile = string(metric, "_", benchestimate, ".", outputformat)
					savefig(joinpath(folder, outputformat, outputfile))
				end

			end # if

		end # for

	end # for

end # function

# Run from command line
create_plots(folder)
