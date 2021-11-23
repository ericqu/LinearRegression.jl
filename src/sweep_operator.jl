using LinearAlgebra
import LinearAlgebra:checksquare

"""
    function sweep_op_internal!(A::AbstractMatrix{Float64}, k::Int64)

    Implement the naive sweep operator described by Goodnight, J. (1979). "A Tutorial on the SWEEP Operator." The American Statistician.
"""
function sweep_op_internal!(A::AbstractMatrix{Float64}, k::Int64)
    p = checksquare(A)
    if k >= p 
        throw(ArgumentError("Incorrect k value"))
    end
    
    @inbounds D = A[k,k] # step 1 D = Aₖₖ
    
    if D == zero(eltype(A))
        throw(ArgumentError("sweep_op_internal!: the element $k,$k of the matrix is zero. Is the Design Matrix symmetric positive definite?"))
    end
    
    @inbounds for i in 1:p
        A[k,i] = A[k,i] / D   # step 2 divide row k by D
    end
    
    @inbounds for i in 1:(k - 1) # step 3: for every other row  i != k 
        B = A[i,k]   # let B = Aᵢₖ
        for j in 1:p
            A[i,j] = A[i,j] - B * A[k, j]  # Subtract B * row k from row i
        end
        A[i,k] = - B / D # set Aᵢₖ = -B/D    
    end

    @inbounds for i in k + 1:p # step 3: for every other row  i != k 
        B = A[i,k]   # let B = Aᵢₖ
        for j in 1:p
            A[i,j] = A[i,j] - B * A[k, j]  # Subtract B * row k from row i
        end
        A[i,k] = - B / D # set Aᵢₖ = -B/D    
    end
    
    @inbounds A[k,k] = 1 / D # step 4 Set Aₖₖ = 1/D
    
    return A
end

function sweep_op!(A::AbstractMatrix{Float64}, k::Int64)
    sweep_op_internal!(A, k)
end

function sweep_op!(A::AbstractMatrix{Float64}, ks::AbstractVector{Int64})
    for k in ks
        sweep_op_internal!(A, k)
    end
    return A
end

"""
    function sweep_op_full!(A::AbstractMatrix{Float64})

    (internal) Get SSE, error sum of squares, for all p-1 models. The last one is the SSE for the full model.
"""
function sweep_op_full!(A::AbstractMatrix{Float64})
    n , p = size(A)

    SSEv = Vector{Float64}(undef, p-1)

    for k in 1:p-1
        sweep_op_internal!(A, k)
        SSEv[k] = A[p,p]
    end
    return SSEv
end

