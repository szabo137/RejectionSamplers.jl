function _assert_correct_boundaries(::Tuple{}, ::Tuple{}) end

function _assert_correct_boundaries(
        lower::Tuple{Vararg{T, N}},
        upper::Tuple{Vararg{T, N}},
    ) where {T <: Real, N}
    first(lower) <= first(upper) || throw(
        ArgumentError(
            "lower boundary need to be smaller or equal to the respective upper boundary",
        ),
    )
    return _assert_correct_boundaries(lower[2:end], upper[2:end])
end

# generic transformation of x in (0,1) to (low,high)
_transform_uniform_val(x::Real, low::Real, high::Real) = (high - low) * x + low
_transform_uniform_val(x::SVector, low::Tuple, high::Tuple) =
    _transform_uniform_val.(x, low, high)


struct UniformSampler{T, N, Ts} <: AbstractSampler{Ts, T}
    lower::NTuple{N, T}
    upper::NTuple{N, T}
    function UniformSampler(
            lower::NTuple{N, T},
            upper::NTuple{N, T};
            sample_type::Type{Ts} = SVector{N, T}
        ) where {T, N, Ts}
        _assert_correct_boundaries(lower, upper)
        return new{T, N, Ts}(lower, upper)
    end
end

UniformSampler(lower::T, upper::T) where {T <: Real} = UniformSampler((lower,), (upper,); sample_type = T)

UniformSampler(lower::AbstractVector, upper::AbstractVector) =
    UniformSampler(Tuple(lower), Tuple(upper))

degrees_of_freedom(::UniformSampler{T, N}) where {T, N} = N

Base.extrema(p::UniformSampler) = (minimum(p), maximum(p))
Base.minimum(p::UniformSampler) = p.lower
Base.maximum(p::UniformSampler) = p.upper

@inline _uniform_weight(lower, upper) = inv(prod(upper .- lower))
_weight(s::UniformSampler) = _uniform_weight(minimum(s), maximum(s))

function _transform(
        s::UniformSampler{T, N, Ts},
        v::Ts,
    ) where {T, N, Ts}

    return Sample(
        constructorof(Ts)(
            ntuple(
                x -> _transform_uniform_val(
                    getindex(v, x),
                    getindex(s.lower, x),
                    getindex(s.upper, x),
                ),
                N,
            ),
        ), _weight(s)
    )
end
function _transform(
        s::UniformSampler{T, 1, Ts},
        v::Ts,
    ) where {T, Ts}

    return Sample(
        constructorof(Ts)(
            _transform_uniform_val(v, s.lower[1], s.upper[1]),
        ),
        _weight(s)
    )
end

function RejectionSamplers._rand_single(
        rng::AbstractRNG,
        s::UniformSampler{T, N, Ts}
    ) where {T, N, Ts}

    u01 = rand(rng, Ts)
    return _transform(s, u01)
end

# TODO:
# - implement `allocate_buffer(rng, uniform_sampler, backend, batch_size)`
# - consider host-side rng sampling
