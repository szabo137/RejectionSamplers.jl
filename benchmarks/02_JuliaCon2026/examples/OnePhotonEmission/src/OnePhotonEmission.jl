module OnePhotonEmission
using RejectionSamplers
using QEDFeynmanDiagrams
using QEDcore
using QEDprocesses
using StaticArrays

export OnePhotonEmission

include("psl.jl")

include(joinpath("procs", "ek_ek.jl"))
include(joinpath("procs", "ekk_ek.jl"))
include(joinpath("procs", "ekkk_ek.jl"))
include(joinpath("procs", "ekkkk_ek.jl"))
include(joinpath("procs", "ekkkkk_ek.jl"))

### General perturbative Compton scattering

# fixed omega version

struct OnePhotonEmission{
        T,
        NPHOTONS,
        DOF,
        C <: AbstractProcessDefinition,
        M <: AbstractModelDefinition,
        PSL <: AbstractOutPhaseSpaceLayout,
    } <: RejectionSamplers.AbstractMultivariateTarget{DOF}
    proc::C
    model::M
    psl::PSL
    omega::T

    function OnePhotonEmission(
            nphotons::Int,
            omega::T,
        ) where {T <: Real}
        proc = ScatteringProcess((Electron(), ntuple(x -> Photon(), nphotons)...), (Electron(), Photon()))
        model = PerturbativeQED()
        in_psl = NPhotonRestSystem(nphotons)
        psl = OPESphericalLayout(in_psl)
        DOF = phase_space_dimension(proc, model, psl)
        return new{T, nphotons, DOF, typeof(proc), typeof(model), typeof(psl)}(proc, model, psl, omega)
    end

end

function RejectionSamplers._compute(
        target::OnePhotonEmission{T, N, DOF},
        coords::SVector{DOF, T},
    ) where {N, DOF, T <: Real}

    psp = PhaseSpacePoint(target.proc, target.model, target.psl, SVector(target.omega), coords)

    return scatter(psp)
end


end # module OnePhotonEmission
