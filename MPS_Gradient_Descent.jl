using LinearAlgebra: norm
using Random: Random
using Optimization
using OptimizationOptimJL
using HDF5
include("Itdvp_GD.jl")



function lossF(params::AbstractVector{R}, p::Tuple) where R
    t = 1;
    A = reshape([params[q] + im*params[q+1] for q in 1:2:dH*dBond*dBond*2], (dBond, dH,dBond));
    time_step = p[1]
    h1 = p[2]
    h2 = p[3]
    T = p[4]
    # println("h")
    AL, AC, AR, C = integrate_l(h1, A, time_step, t/2)
    AL0 = AL[1]

    AL2, AC2, AR2, C2 = integrate_l(h2, AL[end], time_step, t/2)
    AL1 = AL2[end]
    for i in 2:T
        AL, AC, AR, C = integrate_l(h1, AL1, time_step, t/2)
        AL2, AC2, AR2, C2 = integrate_l(h2, AL[end], time_step, t/2)
        AL1 = AL2[end]
    end

    @tensor TC[i,j,k,l] := conj(AL0)[i,m,k]*AL1[j,m,l]
    res = maximum(abs.(LinearAlgebra.eigvals(reshape(TC, (dBond^2, dBond^2)))))

    return res
end

# Tangent basis construction for A in a mixed gauge
function Tbasis(A)
    V = reshape(nullspace(transpose(reshape(conj(A), (dBond*dH,dBond)))), (dBond,dH,dBond))

    basisM = Array{Matrix{Float64}}(undef, dBond^2)
    index = 1
    for i in 1:dBond
        for j in 1:dBond
            M = zeros(dBond, dBond)
            M[i, j] = 1
            basisM[index] = M
            index += 1
        end
    end
    basis_B = Array{Array{ComplexF64,3}}(undef, dBond^2)
    for i in 1:dBond^2
        @tensor B_matrix[j, m, n] := V[j, m, k] *  basisM[i][k, n]
        basis_B[i] = B_matrix
    end
    return basis_B
end

# Transform MPS tensor into vector of parameters

function flatA(A)
    A_flat = vec(A)
    par2 = Vector{Float64}(undef, 2 * length(A_flat))
    for i in 1:length(A_flat)
        par2[2*i-1] = real(A_flat[i]) 
        par2[2*i] = imag(A_flat[i])  
    end
    return par2
end

# Get the PID of the current Julia process
pid = getpid()
println("The PID of this Julia process is $pid")

dH = 2

num = ARGS[1]
maxit = parse(Int, ARGS[2])
kT = parse(Int, ARGS[3])
dt = parse(Float64, ARGS[4]) 
delta = parse(Float64, ARGS[5])
tol = parse(Float64,ARGS[6])
atol = parse(Float64,ARGS[7])
rho = parse(Float64,ARGS[8])
tau = parse(Float64,ARGS[9])
J = g = parse(Float64,ARGS[10])
h = parse(Float64,ARGS[11])
dBond = parse(Int,ARGS[12])
c = parse(Float64,ARGS[13])
maxc = parse(Float64,ARGS[14])

h1 = hamiltonian(jx = 0., jy = 0., jz = J, hx = 0., hy = 0., hz = h); 
h2 = hamiltonian(jx = 0., jy = 0., jz = 0., hx = g, hy = 0., hz = 0.); 

println("maxit = ", maxit)
println("kT = ", kT)
println("dt = ", dt)
println("delta = ", delta)
println("tol = ", tol)
println("atol = ", atol)
println("rho = ", rho)
println("tau = ", tau)
println("J,g = ", J)
println("h = ", h)
println("dBond = ", dBond)
println("c = ", c)
println("maxc = ", maxc)

function gradA( AL, delta, p, c, loss_f)

    # Compute a finite-difference gradient step for tensor `AL` using a complex-step–style
    # symmetric difference on a basis `Tbasis(AL)`. Normalizes the direction, performs a
    # backtracking/forward-tracking line search to update step size `c`, and returns:

    # - `AL_res` : updated left-canonical tensor
    # - `c`      : possibly adjusted step size
    # - `res`    : overlap diagnostic between old `AL` and `AL_res`
    # - `gnorm`  : `norm(g + im* gi)` (difference between real/imag FD gradients)

    # Notes:
    # - Uses globals you already had (`dBond`, `dH`, `ws`, `Tbasis`, `mixedCanonical`,
    #   `rho`, `tau`, `maxc`).
    # - `loss_f(ws, vec(AL))` ireturn a scalar loss (real), we square it inside.

    # Pre-compute baseline objective
    g = Vector{Float64}(undef, dBond^2)
    g0 = lossF(flatA(AL), p)^2
    # g0 = (loss_f(ws, reshape(AL, (dBond^2*dH))))^2  # Cpp version
    # Build basis at current point
    basis_B = Tbasis(AL)
    for i in 1:dBond^2
        g[i] = (lossF(flatA(AL + basis_B[i]*delta), p)^2 - lossF(flatA(AL - basis_B[i]*delta), p)^2) / delta / 2
	    # g[i] = ((loss_f(ws, reshape(AL + basis_B[i]*delta, (dBond^2*dH))))^2 - (loss_f(ws, reshape(AL - basis_B[i]*delta, (dBond^2*dH))))^2)/delta/2 # Cpp version
    end
    gi = Vector{ComplexF64}(undef, dBond^2)
    for i in 1:dBond^2
        gi[i] = (lossF(flatA(AL + im*basis_B[i]*delta), p)^2 - lossF(flatA(AL - im*basis_B[i]*delta), p)^2) / delta / 2
	    # gi[i] = ((loss_f(ws, reshape(AL + im*basis_B[i]*delta, (dBond^2*dH))))^2 - (loss_f(ws, reshape(AL - im*basis_B[i]*delta, (dBond^2*dH))))^2)/(delta)/2 # Cpp version
    end
    # Gradient in tensor space via basis expansion: (g + im*gi) ⋅ basis
    dA = zero(AL)
    for i in 1:dBond^2
        dA .+= (g[i] - gi[i]) .* basis_B[i]
    end
    # Normalize dA
    @tensor TC[i,j,k,l] := conj(dA)[i,m,k]* dA[j,m,l]
    rr = maximum(abs.(LinearAlgebra.eigvals(reshape(TC, (dBond^2, dBond^2)))))
    dA /= sqrt(rr)

    # ---- back/forward tracking line search on c ----
    AL_new = AL + c * dA
    F = lossF(flatA(AL_new), p)^2
    # F = (loss_f(ws, reshape(AL_new, (dBond^2*dH))))^2 # Cpp version

    # If better than baseline → multiply by `rho` once
    if F > g0
        c  *= rho  # progressively more aggressive
    end

    # Try to improve while (F <= g0) and counter small
    coun = 1
    while (F <= g0 && coun <=20)
        c /= tau  
        AL_new = AL + c * dA
        F = lossF(flatA(AL_new), p)^2
	    # F = (loss_f(ws, reshape(AL_new, (dBond^2*dH))))^2 # Cpp version
        if c < 10^(-14)
	        AL_new = AL
            break
        end
	    coun += 1
    end

    # If still stuck, push to `maxc` and try dividing by `tau` until improvement stops
    if coun >=20
        c = maxc
        while (F <= g0)
            c /= tau
            AL_new = AL + c * dA
            F = lossF(flatA(AL_new), p)^2
	        # F = (loss_f(ws, reshape(AL_new, (dBond^2*dH))))^2 # Cpp version
            if c < 10^(-14)
		        AL_new = AL
                break
            end
        end
    end
    # Project back to mixed-canonical and compute overlap diagnostic
    AL_res = mixedCanonical(AL_new)[1]
    @tensor TC[i,j,k,l] := conj(AL)[i,m,k]*AL_res[j,m,l]
    res = maximum(abs.(LinearAlgebra.eigvals(reshape(TC, (dBond^2, dBond^2)))))

    return AL_res, c, res, norm(g + im*gi) 
end

function alg(par2, c, maxc, maxit, loss_f)

    # Run iterative updates starting from parameters `par2`,
    # using `gradA` steps with line search. Returns:

    # - `flatA(AL_new)` : flattened optimized tensor
    # - `cc`            : history of step sizes
    # - `change`        : history of line-search change metric (`ch`)
    # - `dif`           : history of objective diffs (`res2 - res1`)
    # - `curr`          : history of objective values
    # - `it_done`       : number of iterations performed
    # - `grr_`          : history of gradient norms

    # Stopping criteria (all must hold to continue):
    # - improvement `res2 - res1` > `tol`
    # - `1 - res2` > `atol`
    # - `real(grr_[it]) / dBond^2 > 1e-10`
    # Notes:
    # - Uses globals: `dBond`, `dH`, `ws`, `mixedCanonical`, `flatA`, `delta`, `p`, `tol`, `atol`.

    cc = Vector{ComplexF64}(undef, maxit+2)
    change = Vector{ComplexF64}(undef, maxit)
    dif = Vector{ComplexF64}(undef, maxit)
    curr = Vector{ComplexF64}(undef, maxit+2)
    grr_ = Vector{ComplexF64}(undef, maxit+2)
    cc[1] = c

    # Rebuild complex tensor A from interleaved real/imag in `par2`, then left-canonicalize
    A = reshape([par2[q] + im*par2[q+1] for q in 1:2:dH*dBond*dBond*2], (dBond, dH,dBond));
    AL = mixedCanonical(A)[1]
    # Initial objective and first step
    curr[1] =  lossF(flatA(AL), p)^2
    # curr[1] = loss_f(ws, reshape(AL, (dBond^2*dH)))^2 # Cpp version
    AL_new, c, ch, grr = gradA(AL, delta, p, c, loss_f)
    # res2 = (loss_f(ws, reshape(AL_new, (dBond^2*dH))))^2 # Cpp version
    res2 = lossF(flatA(AL_new), p)^2
    curr[2] = res2
    grr_[1] = grr
    cc[2] = c
    res1 = 0
    it = 1
    while (((res2-res1)> tol) && (1 - res2 > atol) &&(real(grr_[it])/(dBond^2) > 10^(-10)))
        res1  = res2
        @time AL_new, c, ch, grr = gradA(AL_new, delta, p, c, loss_f)
        c = min(c, maxc)
        # res2 = (loss_f(ws, reshape(AL_new, (dBond^2*dH))))^2 # Cpp version
        res2 = lossF(flatA(AL_new), p)^2
        println("New loss new F - old F = ", res2 - res1)
        dif[it] = res2-res1
	    grr_[it+1] = grr
        curr[it+2] = res2
        change[it] = ch
        cc[it+2] = c
        it +=1
        if it >= maxit 
            break
        end
        if c <= 10^(-12) 
            break
        end
    end
    return flatA(AL_new), cc, change, dif, curr, it - 1, grr_
end

# Connection to the Cpp code:

LIB_PATH = "./mylib_up$(dBond).so"

mutable struct Workspace
    ptr::Ptr{Cvoid}
end

function Workspace(sz::Int, tolerance::Float64, tolerance_itdvp::Float64, tolerance_exp::Float64, tolerance_rq::Float64, t::Float64, dt::Float64)
    ptr = ccall((:init_workspace, LIB_PATH), Ptr{Cvoid},
                (UInt64, Float64, Float64, Float64, Float64, Float64, Float64),
                sz, tolerance, tolerance_itdvp, tolerance_exp, tolerance_rq, t, dt)
    return Workspace(ptr)
end

function get_dH(ws::Workspace)
    ccall((:get_dH, LIB_PATH), Int32, (Ptr{Cvoid},), ws.ptr)
end

ws = Workspace(10000, 1e-13, 1e-14,1e-20,1e-16, 0.5, dt)
println("Workspace initialized")

function loss_cpp(ws::Workspace, arr::Vector{ComplexF64})
    ccall((:init_julia, LIB_PATH), Float64,
          (Ptr{Cvoid}, Ptr{ComplexF64}),
          ws.ptr, arr)
end

function loss2_cpp(ws::Workspace, arr::Vector{ComplexF64})
	ccall((:init_julia_loss2, LIB_PATH), Float64,
	      (Ptr{Cvoid},Ptr{ComplexF64}),
	      ws.ptr, arr)
end

function loss_double_cpp(ws::Workspace, arr::Vector{ComplexF64})
    ccall((:init_julia_loss_double, LIB_PATH), Float64,
          (Ptr{Cvoid}, Ptr{ComplexF64}),
          ws.ptr, arr)
end

function hamiltonian_c(ws::Workspace, jx::Float64, jy::Float64, jz::Float64,
    hx::Float64, hy::Float64, hz::Float64, whichh::String)
    ccall((:hamiltonian_c, LIB_PATH), 
    Cvoid,                       
    (Ptr{Cvoid}, Cdouble, Cdouble, Cdouble, 
    Cdouble, Cdouble, Cdouble, Cstring), 
    ws.ptr, jx, jy, jz, hx, hy, hz, whichh)
end

function set_paulis(ws::Workspace)
    ccall((:set_paulis_c, LIB_PATH),
          Cvoid,          
          (Ptr{Cvoid},), 
          ws.ptr)
end

function change_dt_in_julia(ws::Workspace, dt::Float64)
    ccall(
        (:change_dt, LIB_PATH ),
        Cvoid,
        (Ptr{Cvoid}, Cdouble),
        ws.ptr, dt
    )
end

# Initialize Cpp envinroment and 
set_paulis(ws)
hamiltonian_c(ws, 0.0, 0.0, J, 0.0, 0.0, h, "h1")
hamiltonian_c(ws, 0.0, 0.0, 0., J, 0.0, 0.0, "h2")

par = rand([-1,1],(dH*dBond*dBond*2)).*rand((dH*dBond*dBond*2))
p = (dt, h1, h2, kT)
@time println("julia loss ", lossF(par, p))
p = (dt, h1, h2, kT)

# sol, cc, change, dif, curr, it, grads_ = alg(par, c, maxc, maxit, loss_cpp) # Cpp version
sol, cc, change, dif, curr, it, grads_ = alg(par, c, maxc, maxit, lossF)

file_path = "./files/BD$(dBond)/50_loss0_J$(J)_dB$(dBond)_dt$(dt).h5"

h5write(file_path,"par", sol)
h5write(file_path, "grads", grads_)
h5write(file_path,"dif", dif)
h5write(file_path,"curr", curr)
h5write(file_path,"cc", cc)
h5write(file_path,"change", change)
h5write(file_path,"it", it)
h5write(file_path,"kT", kT)
h5write(file_path,"dt", dt)
h5write(file_path,"delta", delta)
h5write(file_path,"tol", tol)
h5write(file_path,"atol", atol)
h5write(file_path,"maxit", maxit)
h5write(file_path,"rho", rho)
h5write(file_path,"tau", tau)
h5write(file_path,"Jg", J)
h5write(file_path,"h", h)
h5write(file_path,"dBond", dBond)
h5write(file_path,"c", c)
h5write(file_path,"maxc", maxc)
