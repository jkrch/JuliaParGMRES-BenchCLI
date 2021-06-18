"""
Matrix of a 2D piecewise linear FEM discretization
=======================

*Input options:*

+ n: size of the matrix
""" 
function fem2d(n::Int; nnzrow=4)
    A = ExtendableSparseMatrix(n, n)
    sprand_sdd!(A, nnzrow=nnzrow)
    flush!(A)
    return A.cscmatrix
end


"""
Matrix for a mock finite difference operator for a diffusion
problem with random coefficients on a unit hypercube
=======================

*Input options:*

+ n: number of unknowns in x, y, z direction 
""" 
function fdm3d(n::Int)
	nx = floor(Int, n^(1/3))
	ny = floor(Int, n^(1/3))
	nz = floor(Int, n^(1/3))
    A = ExtendableSparseMatrix(nx*ny*nz, nx*ny*nz)
    fdrand!(A, nx, ny, nz)
    flush!(A)
    return  A.cscmatrix
end




