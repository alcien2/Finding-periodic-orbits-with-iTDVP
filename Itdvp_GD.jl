# Packages used in this code
using LinearAlgebra # Used for svd, qr.
using TensorOperations # Used for @tensor
using KrylovKit # Used for eigsolve, linsolve and exponentiate
using EllipsisNotation # Used for ... slicing of arrays

using Optimization
using OptimizationOptimJL
using ForwardDiff
using LineSearches
using CSV
using DataFrames


function leftPack(A)
    dBond, dH = size(A)[1:2];
    Amatrix = reshape(A, (dBond*dH, dBond));
    return Amatrix;
end;

function leftUnpack(Amatrix)
    dC, dBond = size(Amatrix);
    dH = dC ÷ dBond;
    A = reshape(Amatrix, (dBond, dH, dBond));
    return A;
end;

function rightPack(A)
    dBond, dH = size(A)[1:2];
    Amatrix = reshape(A, (dBond, dBond*dH));
end;

function rightUnpack(Amatrix)
    dBond, dC = size(Amatrix);
    dH = dC ÷ dBond;
    A = reshape(Amatrix, (dBond, dH, dBond));
end;

function qrGauged(a; tol=eps(Float64))
    f = qr(a)
    q = Matrix(f.Q);
    r = f.R;
    D = size(r)[1];
    rDiag = zeros((D,D));
    for i=1:D
        check = r[i, i];
        # If an item on the diagonal is negative,
        # record that to flip signs later
        if abs(check) > tol && real(check) < 0
            rDiag[i, i] = -1;
        else
            rDiag[i,i] = 1;
        end;
    end;
    # Flip signs now
    r = rDiag * r;
    q = q * rDiag;

    return q, r;
end;

function rqGauged(a; tol=eps(Float64))
    
    # RQ from QR
    # https://leohart.wordpress.com/2010/07/23/rq-decomposition-from-qr-decomposition/
    reversed_a = reverse(a, dims=1);

    f = qr(transpose(reversed_a));
    q = Matrix(f.Q);
    r = f.R;

    r = reverse(transpose(r), dims=1);
    r = reverse(r, dims=2);
    q = reverse(transpose(q), dims=1);
    
    D = size(r)[2];
    rDiag = zeros((D,D));
    for i=1:D
        check = r[i, i];
        if abs(check) > tol && real(check) < 0
            rDiag[i, i] = -1;
        else
            rDiag[i,i] = 1;
        end;
    end;
    r = r * rDiag;
    q = rDiag * q;
    return r, q;
end;

function svdGauged(a; tol=eps(Float64))
    F = svd(a);
    u = F.U;
    s = F.S;
    vh = F.Vt;
    # Form singular values as matrix
    sMatrix = zeros((size(u)[2], size(vh)[1]));
    for i=1:length(s)
        sMatrix[i, i] = s[i];
    end;
    D = min(size(a)[1], size(a)[2]);
    # Scan through left singular vectors
    # Make first non-zero component real and positive
    for i=1:D
        ui = u[:, i];
        mags = abs.(ui);
        # Find the first component with magnitude larger than tol
        idx = findfirst(mags .> tol);
        val = ui[idx];
        # Find its phase
        angle = atan(imag(val), real(val));
        # If the phase is not zero
        if abs(angle) > tol
            phase = exp(1im * angle);
            # Multiply the columns of u with the opposite phase
            u[:, i] *= conj(phase);
            # Multiply the rows of vh with the phase
            vh[i, :] *= phase;
        end;

        # Drop the imaginary part of the first non-zero element of u completely
        u[idx, i] = real(u[idx, i]);
    end;

    return u, sMatrix, vh;
end;

function polarGauged(a; side="right")
    m, n = size(a);
    k = min(m, n);
    us, ss, vhs = svdGauged(a);
    u = us[:, 1:k] * vhs[1:k, :];
    if side == "right"
        p = vhs'[:, 1:k] * ss[1:k, 1:k] * vhs[1:k, :];
    elseif side == "left"
        p = us[:, 1:k] * ss[1:k, 1:k] * us'[1:k, :];
    end;

    return u, p;
end;

function leftOrthonormalize(A; L0=nothing, tol=10^(-12), resDiv=10)
# function leftOrthonormalize(A; L0=nothing, tol=10^(-15), resDiv=10)
    @label beg
    dBond, dH = size(A)[1:2];
    # If L0 is provided, take that as the initial guess
    # Otherwise create a random matrix as L0
    if L0 == nothing
        # Get a random matrix
        L = rand(ComplexF64, (dBond, dBond));
        L /= norm(L);
    else
        # Take the provided guess but normalize it
        L = copy(L0) / norm(L0);
    end;
    Lold = copy(L);
    # First guess for A_L and L
    @tensor LtimesA[i,j,k] := L[i,l] * A[l,j,k];
    ALmatrix, L = qrGauged(leftPack(LtimesA))
    AL = leftUnpack(ALmatrix);
    normL = norm(L);
    L /= normL;
    res = maximum(abs.(L-Lold));
    coun = 1
    while (res > tol) 
        # Transfer map for the left fixed point
        function leftFixedPointMap(Lguess)
            @tensor contraction[k,l] := Lguess[i,j]*conj(AL)[i,m,k]*A[j,m,l];
            return contraction;
        end;
        if coun >=20
            @goto beg
        end
        L = eigsolve(leftFixedPointMap, L, tol=res/resDiv)[2][..,1];
        # Refine with QR
        L = qrGauged(L)[2];
        L /= norm(L);
        Lold = copy(L)
        # Iterate with QR
        @tensor LtimesA[i,j,k] = L[i,l] * A[l,j,k];
        ALmatrix, L = qrGauged(leftPack(LtimesA))
        AL = leftUnpack(ALmatrix);
        normL = norm(L);
        L /= normL;
        res = maximum(abs.(L-Lold));
        coun +=1
    end;
    return AL, L, normL;
end;

function rightOrthonormalize(A; R0=nothing, tol=10^(-12), resDiv=10)
# function rightOrthonormalize(A; R0=nothing, tol=10^(-15), resDiv=10)
    @label beg
    dBond, dH = size(A)[1:2];

    if R0 == nothing
        # Get a random matrix
        R = rand(ComplexF64, (dBond, dBond));
        R /= norm(R);
    else
        # Take the provided guess but normalize it
        R = copy(R0) / norm(R0);
    end;
    Rold = copy(R);
    # First guess for A_R and R
    @tensor AtimesR[i,j,k] := A[i,j,l] * R[l,k];
    R, ARmatrix = rqGauged(rightPack(AtimesR))
    AR = rightUnpack(ARmatrix);
    normR = norm(R);
    R /= normR;
    res = maximum(abs.(R-Rold));
    coun = 1
    while (res > tol) 
        # Transfer map for the right fixed point
        function rightFixedPointMap(Rguess)
            @tensor contraction[i,j] := conj(AR)[j,m,l] * A[i,m,k] * Rguess[k,l];
            return contraction;
        end;
        if coun >=20
            @goto beg
        end

        R = eigsolve(rightFixedPointMap, R, tol=res/resDiv)[2][..,1];
        R = rqGauged(R)[1];
        R /= norm(R);
        Rold = copy(R)
        @tensor AtimesR[i,j,k] = A[i,j,l] * R[l,k];
        R, ARmatrix = rqGauged(rightPack(AtimesR))
        AR = rightUnpack(ARmatrix);
        normR = norm(R);
        R /= normR;
        res = maximum(abs.(R-Rold));
        coun+=1
    end;
    return AR, R, normR;
end;

function mixedCanonical(A, tol=10^(-13), resDiv=10)
    dBond, dH = size(A)[1:2];
    # Compute left and right orthonormal forms
    AL, _, normA = leftOrthonormalize(A, tol=tol, resDiv=resDiv);
    AR, C, _ = rightOrthonormalize(AL,  tol=tol, resDiv=resDiv);
    # Diagonalize C
    u, C, vh = svdGauged(C);

    # Absorb u and vh to AL and AR
    @tensor AL[i,j,k] = u'[i,l]*AL[l,j,m]*u[m,k];
    @tensor AR[i,j,k] = vh[i,l]*AR[l,j,m]*vh'[m,k];
    # Compute AC
    @tensor AC[i,j,k] := AL[i,j,l]*C[l,k];

    return AL, AC, AR, C, normA;
end;

function minACC(AC, C)
    # See "Variational optimization algorithms for uniform matrix product states"
    # by Zauner-Stauber et al.
    # for an explanation.
    # arXiv: https://arxiv.org/abs/1701.07035
    # PhysRevB: https://doi.org/10.1103/PhysRevB.97.045145

    dBond, dH = size(AC)[1:2];
    ULAC = leftUnpack(polarGauged(leftPack(AC), side="right")[1])
    ULC = polarGauged(C, side="right")[1]
    @tensor AL[i,j,k] := ULAC[i,j,l] * ULC'[l,k];
    URAC = rightUnpack(polarGauged(rightPack(AC), side="left")[1])
    URC = polarGauged(C, side="left")[1]
    @tensor AR[i,j,k] := URC'[i,l] * URAC[l,j,k];

    return AL, AR;
end;

function ERRp(AR; l0=nothing, tol=eps(Float64))
    """Computes (1 - T_R') where T_R' is the transfer tensor in the right
    canonical form with the dominant component subtracted out.
    
    Part of the rhs of eq. (169)-2.
    """

    dBond, dH = size(AR)[1:2];
    # Compute the left fixed point of AR's transfer tensor
    @tensor TR[i,j,k,l] := conj(AR)[i,m,k] * AR[j,m,l];
    # Its action on the left fixed point
    function leftFixedPointMap(lGuess)
        @tensor contraction[k,l] := lGuess[i,j]*TR[i,j,k,l];
        return contraction;
    end;

    if l0 == nothing
        # Random Hermitian matrix
        l = Matrix(Hermitian(rand(ComplexF64, (dBond, dBond))));
    else
        l = copy(l0);
    end;
    
    # Find the fixed point
    l = eigsolve(leftFixedPointMap, l, tol=tol)[2][..,1];
    # Normalize it with its trace
    l /= tr(l);
    # Make it exactly Hermitian
    l = Matrix(Hermitian(l));
    # Negate and remove the fixed point projector
    @tensor domP[i,j,k,l] := Matrix{ComplexF64}(I, dBond, dBond)[i,j] * l[k,l];
    TRPP = -TR + domP;
    # Add identity
    @tensor idP[i,j,k,l] := Matrix{ComplexF64}(I, dBond, dBond)[i,k] * Matrix{ComplexF64}(I, dBond, dBond)[j,l];
    TRP = idP + TRPP;
    return TRP, l;
end;

function getRh(AR, h; l0=nothing, Rh0=nothing, tol=eps(Float64))
    """Computes Rh in eq. (169), a contribution to the tangent space projector.
        h is the local two site Hamiltonian.
    """
    dBond, dH = size(AR)[1:2];
    TRP, l = ERRp(AR, l0=l0);
    # Contraction around the local Hamiltonian
    @tensor hc[k,l] := conj(AR)[k,i,p] * conj(AR)[p,j,q] * h[i,j,m,n] * AR[l,m,o] * AR[o,n,q];
    @tensor energy = hc[i,j]*l[i,j]
    @tensor hct[i,j] := hc[i,j] - Matrix{ComplexF64}(I, dBond, dBond)[i,j]*energy
    function RhEq(Rhguess)
        @tensor preres[i,j] := TRP[i,j,k,l] * Rhguess[l,k];
        return preres - hct;
    end;

    if Rh0 == nothing
        # Compute the inverse
        Rh = Matrix(transpose(reshape(pinv(reshape(TRP, (dBond^2, dBond^2)))*reshape(hct, (dBond^2)),(dBond, dBond))));
    else
        Rh = copy(Rh0);
    end;
    # println(TRP)
    Rh = linsolve(RhEq, zeros(ComplexF64, size(Rh)), Rh, tol=tol)[1];
    sol = Matrix(Hermitian(Rh));
    return sol, l;
end;

function ELLp(AL; r0=nothing, tol=eps(Float64))
    """Computes (1 - T_L') where T_L' is the transfer tensor in the left
    canonical form with the dominant component subtracted out.
    
    Part of the rhs of eq. (169)-1.
    """
    dBond, dH = size(AL)[1:2];
    @tensor TL[i,j,k,l] :=  conj(AL)[j,m,l] * AL[i,m,k];
    # println(TL)
    function rightFixedPointMap(rGuess)
        @tensor contraction[i,j] := TL[i,j,k,l]*rGuess[k,l];
        return contraction;
    end;

    if r0 == nothing
        r = Matrix(Hermitian(rand(ComplexF64, (dBond, dBond))));
    else
        r = copy(r0);
    end;
    # Find the fixed point
    r = eigsolve(rightFixedPointMap, r, tol=tol)[2][..,1];
    r /= tr(r);
    r = Matrix(Hermitian(r));
    # println(r)
    # Negate and remove the fixed point projector
    @tensor domP[i,j,k,l] := r[i,j] * Matrix{ComplexF64}(I, dBond, dBond)[k,l];
    
    TLPP = -TL + domP;
    # Add identity

    @tensor idP[i,j,k,l] := Matrix{ComplexF64}(I, dBond, dBond)[i,k] * Matrix{ComplexF64}(I, dBond, dBond)[j,l];
    TLP = idP + TLPP;
    return TLP, r;
end;

function getLh(AL, h; r0=nothing, Lh0=nothing, tol=eps(Float64))
    """Computes Lh in eq. (169), a contribution to the tangent space projector.
        h is the local two site Hamiltonian.
    """

    dBond, dH = size(AL)[1:2];
    TLP, r = ELLp(AL, r0=r0);
    # Contraction around the local Hamiltonian
    @tensor hc[i,j] := conj(AL)[k,l,q] * conj(AL)[q,m,j] * h[l,m,n,o] * AL[k,n,p] * AL[p,o,i];
    @tensor energy = hc[i,j]*r[i,j]
    @tensor hct[i,j] := hc[i,j] - Matrix{ComplexF64}(I, dBond, dBond)[i,j]*energy
    # Equation to be solved
    function LhEq(Lhguess)
        @tensor preres[k,l] := Lhguess[j,i] * TLP[i,j,k,l];
        return preres - hct;
    end;
    if Lh0 == nothing
        Lh = Matrix(transpose(reshape(transpose(reshape(hct, (dBond^2)))*pinv(reshape(TLP, (dBond^2, dBond^2))),(dBond, dBond))));
    else
        Lh = copy(Lh0);
    end;
    Lh = linsolve(LhEq, zeros(ComplexF64, size(Lh)), Lh, tol=tol)[1];
    sol = Matrix(Hermitian(reshape(Lh, (dBond, dBond))));
    return sol, r;
end;

function getG1L(AL, h)
    """Computes acting-from-left part of G_1 in eq. (170) (second term).
        So this would hit on A_C to the right.
        Part of the tangent space projector.
    """
    dBond, dH = size(AL)[1:2];
    @tensor G1L[i,j,k,l] := conj(AL)[m,n,i]*h[n,j,p,l]*AL[m,p,k];
    # This is supposed to be Hermitian.
    G1Lmatrix = Matrix(Hermitian(reshape(G1L, (dH * dBond, dH * dBond))));
    G1L = reshape(G1Lmatrix, (dBond, dH, dBond, dH));

    return G1L;
end;

function getG1R(AR, h)
    """Computes acting-from-right part of G_1 in eq. (170) (first term).
        So this would hit on A_C to the left.
        Part of the tangent space projector.
#     """

    dBond, dH = size(AR)[1:2];
    @tensor G1R[i,j,k,l] := conj(AR)[l,o,m] * h[k,o,i,p] * AR[j,p,m];

    # This is supposed to be Hermitian.
    G1Rmatrix = Matrix(Hermitian(reshape(G1R, (dH * dBond, dH * dBond))));
    G1R = reshape(G1Rmatrix,(dH,dBond,dH,dBond));

    return G1R;
end;


# Pauli operators
Sx = [0 1; 1 0];
Sy = 1im * [0 -1; 1 0];
Sz = [1 0; 0 -1];
identity = Matrix{ComplexF64}(I, 2, 2);

function leftrightPackT(A)
    dBond = size(A)[1];
    Amatrix = reshape(A, (dBond*dBond, dBond*dBond));
    return Amatrix;
end;

function leftrightUnpackT(Amatrix)
    dC1, dC2 = size(Amatrix);
    dBond = convert(Integer,sqrt(dC1));
    A = reshape(Amatrix, (dBond, dBond, dBond, dBond));
    return A;
end;
function leftrightPackLR(A)
    dBond = size(A)[1];
    Amatrix = reshape(A, (dBond*dBond));
    return Amatrix;
end;

function leftrightUnpackLR(Amatrix)
    dC1 = size(Amatrix)[1];
    dBond = convert(Integer,sqrt(dC1));
    A = reshape(Amatrix, (dBond, dBond));
    return A;
end;


function getG2M(AL, AR, h)
    """Computes part of G_2 eq. (172).
        This would take C in the "middle".
        This doesn't exist in the article directly, to understand,
        see the tangent space projector in eq. 91.
    """

    dBond, dH = size(AL)[1:2];
    @tensor G2M[i,j,k,l] := conj(AL)[m,n,i] * conj(AR)[j,o,r] * h[n,o,p,q] * AL[m,p,k] * AR[l,q,r];

    # This is supposed to be Hermitian.
    G2Mmatrix = Matrix(Hermitian(reshape(G2M, (dBond^2, dBond^2))));
    G2M = reshape(G2Mmatrix,(dBond, dBond, dBond, dBond));
    
    return G2M;
end;

function getG2M2(AL, AR, h) 
    dBond, dH = size(AL)[1:2];
    @tensor G2M[r, l] := conj(AL)[m,n,i] * conj(AL)[i,o,r] * h[n,o,p,q] * AL[m,p,k] * AL[k,q,l];

    G2Mmatrix = Matrix(Hermitian(reshape(G2M, (dBond, dBond))));
    G2M = reshape(G2Mmatrix,(dBond, dBond));
    
    return G2M;
end;


function eulerStepExp(AL, AC, AR, C, h; tstep=0.01, l0=nothing, r0=nothing, Lh0=nothing, Rh0=nothing, tol=eps(Float64))
    """Time evolve AC and C, use them to find evolved AL, AR.
    """
    
    # The four parts of G_1
    G1L = getG1L(AL, h);
    G1R = getG1R(AR, h);
    Lh, r = getLh(AL, h, r0=r0, Lh0=Lh0);
    Rh, l = getRh(AR, h, l0=l0, Rh0=Rh0);

    function hamAC(AC_)
        res = zeros(ComplexF64,(dBond,dH,dBond));
        @tensor res[i,j,m] += G1L[i,j,k,l] * AC_[k,l,m];
        @tensor res[i,l,m] += AC_[i,j,k] * G1R[j,k,l,m];
        @tensor res[i,k,l] += Lh[i,j]  * AC_[j,k,l];
        @tensor res[i,j,l] += AC_[i,j,k] * Rh[k,l];
        
        return res;
    end;

    # Forward AC
    AC_ = exponentiate(hamAC,-1im * tstep, AC,ishermitian=true,tol=tol / abs(tstep))[1];
    
    # The three parts of G_2
    G2M = getG2M(AL, AR, h);
    G2M2 = getG2M2(AL, AR, h);
    function hamC(C_)
        res = zeros(ComplexF64,(dBond,dBond));
        @tensor res[i,j] += G2M2[i,k] * C_[k,j];
        @tensor res[i,k] += conj(AL)[w,m,i] *Lh[w,j]*AL[j,m,l] * C_[l,k];
        @tensor res[i,k] += C_[i,j] * Rh[j,k];
        @tensor res[i,j] += G2M[i,j,k,l] * C_[k,l];
        return res;
        
    end;

    C_ = exponentiate(hamC,-1im * tstep,C,ishermitian=true,tol=tol / abs(tstep))[1];

    # Find evolved A_L and A_R using evolved A_C and C
    AL_, AR_ = minACC(AC_, C_);

    return AL_, AC_, AR_, C_, l, r, Lh, Rh;
end;

function hamiltonian(;jx=1., jy=2., jz=3.0, hx=0.0, hy=0., hz=1.0)
    """Hamiltonian for
        j_i S_i S_{i+1} + h_i S_i
        type interactions.

        ham[ijkl] = <ij|H|kl>.
    """

    ham = zeros(Complex{Float64},(2, 2, 2, 2));
    function addTerm(op1, op2)
        @tensor res[i,j,k,l] := op1[i,k]*op2[j,l];
        return res;
    end;

    ham += jx * addTerm(Sx,Sx);
    ham += jy * addTerm(Sy,Sy);
    ham += jz * addTerm(Sz,Sz);
    
    ham += (hx * addTerm(Sx,identity) + hx * addTerm(identity, Sx))/2;
    ham += (hy * addTerm(Sy,identity) + hy * addTerm(identity, Sy))/2;
    ham += (hz * addTerm(Sz,identity) + hz * addTerm(identity, Sz))/2;

    return ham;
end;

function integrate_l(h, A, time_step, t)
    l = Int(floor(t /time_step))
    AL = Vector{Array{ComplexF64, 3}}(undef, 2)
    AC = Vector{Array{ComplexF64, 3}}(undef, 2)
    AR = Vector{Array{ComplexF64, 3}}(undef, 2)
    C = Vector{Matrix{ComplexF64}}(undef, 2)
    AL[1], AC[1], AR[1], C[1], normA = mixedCanonical(A);
    AL_, AC_, AR_, C_, normA = mixedCanonical(A);

    ti = 0
    for i in 1:Int(l)
        ti += time_step
        AL_ , AC_ , AR_, C_ , _, _, _, _ = eulerStepExp(AL_ , AC_ , AR_ , C_ , h, tstep=time_step);
    end
    t_last = t - l*time_step
    
    ti+=t_last
    AL[2], AC[2], AR[2], C[2], _, _, _, _ = eulerStepExp(AL_, AC_, AR_, C_, h, tstep = t_last);
    return AL, AC, AR, C
end

# Returns array with tensors at all time steps
function integrate(h, A, time_step, t)
    l = Int(floor(t /time_step))
    AL = Vector{Array{ComplexF64, 3}}(undef, Int(l)+2)
    AC = Vector{Array{ComplexF64, 3}}(undef, Int(l)+2)
    AR = Vector{Array{ComplexF64, 3}}(undef, Int(l)+2)
    C = Vector{Matrix{ComplexF64}}(undef, Int(l)+2)
    AL[1], AC[1], AR[1], C[1], normA = mixedCanonical(A);
    ti = 0
    for i in 1:Int(l)
        ti += time_step
        AL[i+1] , AC[i+1] , AR[i+1] , C[i+1] , _, _, _, _ = eulerStepExp(AL[i] , AC[i] , AR[i] , C[i] , h, tstep=time_step);
        
    end
    t_last = t - l*time_step
    ti+=t_last
    AL[l + 2], AC[l + 2], AR[l + 2], C[l + 2], _, _, _, _ = eulerStepExp(AL[l + 1], AC[l + 1], AR[l + 1], C[l + 1], h, tstep = t_last);
    return AL, AC, AR, C
end
