bench_name = "multi_stage"
@info "Adding benchmark: $bench_name"
nevent_vec = 2 .^ (5:6)
@info "used nevents: $nevent_vec"
batch_size_vec = 2 .^ (5:6)
@info "used batch sizes: $batch_size_vec"

RNG_CPU = Xoshiro(137)
@info "used rng: $RNG_CPU"

group = addgroup!(SUITE, bench_name)
for out_type in DTYPES

    local _group = addgroup!(group, "$out_type")

    for dim in DIMS

        local __group = addgroup!(_group, dim)

        in_type = SVector{dim, out_type}

        taget, proposal, max_val = generation_setup(RNG, in_type, out_type, mod, psl)

        SAMPLER = RejectionSampler(target, proposal, max_val; backend = BACKEND, in_type = in_type, out_type = out_type)

        for N in nevent_vec
            for batch_size in batch_size_vec
                @info "Adding benchmark problem: dtype=$dtype, dim=$dim, Neve=$N, Nbatch=$batch_size"

                __group[batch_size][N] = @benchmarkable @sb(
                    begin
                        sample_multi_stage(RNG, eg, RS, BS)
                    end
                )
            end
        end
    end
end
