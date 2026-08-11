bench_name = "single_stage_naive"
@info "Adding benchmark: $bench_name"
nevent_vec = 2 .^ (5:6)
@info "used nevents: $nevent_vec"

group = addgroup!(SUITE, bench_name)
for out_type in DTYPES

    local _group = addgroup!(group, "$out_type")


    local __group = addgroup!(_group, problem_size_arg)

    in_type = SVector{problem_size_arg, out_type}

    target, proposal, max_val, in_type = generate_setup(problem_arg, out_type, problem_size_arg)

    SAMPLER = RejectionSampler(target, proposal, max_val; backend = BACKEND, in_type = in_type, out_type = out_type)

    for N in nevent_vec
        @info "Adding benchmark problem: dtype=$out_type, problem_size_arg=$problem_size_arg, Neve=$N"

        __group[N] = @benchmarkable @sb(
            begin
                RejectionSamplers.sample_naive_single_stage!($SAMPLER, output, $N)
                output.level .= zero(UInt32)
            end
        ) setup = begin

            # Allocate output buffers
            output = OutBuffer(
                get_backend($SAMPLER),
                input_type($SAMPLER),
                output_type($SAMPLER),
                $N
            )
        end
    end
end
