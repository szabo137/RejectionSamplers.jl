# N ... number of photons -> assume all have the layout (omega, 0, 0, omega) with one omega for all photons
abstract type AbstractNPhotonRestSystem{N} <: AbstractTwoBodyInPhaseSpaceLayout end

QEDbase.phase_space_dimension(proc, model, ::AbstractNPhotonRestSystem) = 1 # omega is fix


### in-psl: multi photon, same omega for all imcoming photon

struct NPhotonRestSystem{N} <: AbstractNPhotonRestSystem{N}

    function NPhotonRestSystem(
            n_photons::Int
        )
        return new{n_photons}()
    end

end

Base.broadcastable(psl::NPhotonRestSystem) = Ref(psl)

function QEDbase._build_momenta(
        proc::AbstractProcessDefinition,
        ::AbstractPerturbativeModel,
        ::NPhotonRestSystem{N},
        in_coords,
    ) where {N}
    T = eltype(in_coords)

    mass_rest = mass(T, incoming_particles(proc)[1])
    P_rest = SFourMomentum{T}(mass_rest, 0, 0, 0)

    omega = @inbounds in_coords[1]

    Ks = ntuple(x -> SFourMomentum{T}(omega, 0, 0, omega), N)

    return (P_rest, Ks...)
end

@inline function _pert_OPE_omega_prime(pt, cth)
    Et = getE(pt)
    rho2_t = getMag2(pt)
    s = Et^2 - rho2_t

    return (s - 1) / (2 * (Et - sqrt(rho2_t) * cth))
end

struct OPESphericalLayout{INPSL <: NPhotonRestSystem} <:
    AbstractOutPhaseSpaceLayout{INPSL}
    in_psl::INPSL
end

function QEDbase.phase_space_dimension(
        proc::AbstractProcessDefinition, model::PerturbativeQED, psl::OPESphericalLayout
    )
    # cth, phi
    return 2
end

QEDbase.in_phase_space_layout(psl::OPESphericalLayout) = psl.in_psl

function QEDbase._build_momenta(
        proc::AbstractProcessDefinition,
        model::PerturbativeQED,
        psl::OPESphericalLayout,
        in_coords::NTuple{1, T},
        out_coords::NTuple{2, T},
    ) where {T <: Real}
    P, Ks... = QEDbase._build_momenta(proc, model, in_phase_space_layout(psl), in_coords)
    sumK = sum(Ks)
    Pt = P + sumK
    cth, phi = @inbounds out_coords
    omega_prime = _pert_OPE_omega_prime(Pt, cth)
    sth = sqrt(1 - cth^2)
    sphi, cphi = sincos(phi)

    Kp = SFourMomentum{T}(
        omega_prime, omega_prime * sth * cphi, omega_prime * sth * sphi, omega_prime * cth
    )
    Pp = Pt - Kp

    return (P, Ks...), (Pp, Kp)
end
