function _generate_gaussian_setup(dtype, dim)

    mu = zeros(dtype, dim)

    sig = ones(dtype, dim)
    l = [dtype(0.0), fill(dtype(-5), dim - 1)...]
    u = fill(dtype(5), dim)
    target = TruncatedGaussians.TruncatedGaussian(mu, sig, l, u)

    proposal = UniformSampler(
        fill(dtype(-6), dim),
        fill(dtype(6), dim),
    )

    max_val = dtype(maximum_value(target)) * RejectionSamplers._weight(proposal)

    in_type = SVector{dim, dtype}

    return target, proposal, max_val, in_type
end

function _generate_one_photon_emission_setup(dtype, nphotons)

    omega = one(dtype)

    target = OnePhotonEmission.OnePhotonEmission(nphotons, omega)

    proposal = UniformSampler(
        dtype.([-1, 0]),
        dtype.([1, 2pi])
    )

    max_val = findmax(dtype, SVector{2, dtype}, target, proposal, QuantileReductionMethod(dtype(0.01), 10_000))

    in_type = SVector{2, dtype}

    return target, proposal, max_val, in_type
end

function generate_setup(problem_name, dtype, problem_size)

    if problem_name == "one-photon-emission"
        return _generate_one_photon_emission_setup(dtype, problem_size)
    end

    if problem_name == "truncated-gaussian"
        return _generate_gaussian_setup(dtype, problem_size)
    end
end
