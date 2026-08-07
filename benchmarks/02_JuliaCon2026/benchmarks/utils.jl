function generate_setup(dtype, dim)

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

    return target, proposal, max_val
end
