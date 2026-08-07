bench_name = "single_stage"
@info "Adding benchmark: $bench_name"
nevent_vec = 2 .^ (5:6)
@info "used nevents: $nevent_vec"
batch_size_vec = 2 .^ (5:6)
@info "used batch sizes: $batch_size_vec"

group = addgroup!(SUITE, bench_name)
for out_type in DTYPES

    local _group = addgroup!(group, "$out_type")

    for dim in DIMS

        local __group = addgroup!(_group, dim)

        in_type = SVector{dim, out_type}

        target, proposal, max_val = generate_setup(out_type, dim)

        SAMPLER = RejectionSampler(target, proposal, max_val; backend = BACKEND, in_type = in_type, out_type = out_type)

        for N in nevent_vec
            for batch_size in batch_size_vec
                @info "Adding benchmark problem: dtype=$out_type, dim=$dim, Neve=$N, Nbatch=$batch_size"

                __group[batch_size][N] = @benchmarkable @sb(
                    begin
                        sample_single_stage($SAMPLER, $N, $batch_size)
                    end
                )
            end
        end
    end
end
