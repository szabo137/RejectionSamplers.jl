#=
function Base.findmax(
        rng::AbstractRNG,
        out_dtype::Type{T},
        target::AbstractTargetDistribution,
        proposal::AbstractProposalDistribution,
        method::AbstractSampleBasedMaxFinder;
        dtype = out_dtype,
    ) where {T <: Real}

    # Stages:
    # - build samples (momenta)
    # - build psps
    # - calculate weights==dcs
    # - perform maxfinder on weight array

    N = _nsamples(method)

    coords = Vector{dtype}(undef, N)
    weights = Vector{out_dtype}(undef, N)
    propose!(rng, proposal, coords, weights)
    weights = _compute.(target, coords)

    return _findmax(method, weights)

end

=#

# CPU version only
function Base.findmax(
        out_dtype::Type{T},
        in_type::Type{Ti},
        target::AbstractTargetDistribution,
        proposal::AbstractSampler,
        method::AbstractSampleBasedMaxFinder
    ) where {T <: Real, Ti}

    N = _nsamples(method)
    backend = CPU()

    batch = SampleBuffer(backend, Ti, T, N)
    rand!(proposal, batch)
    _compute_update_kernel(backend, 32)(batch, target; ndrange = N)
    weights = batch.samples.weight

    return _findmax(method, weights)

end
