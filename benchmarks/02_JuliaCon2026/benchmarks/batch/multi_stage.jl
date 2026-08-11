bench_name = "multi_stage"
@info "Adding benchmark: $bench_name"
nevent_vec = 2 .^ (5:6)
@info "used nevents: $nevent_vec"
batch_size_vec = 2 .^ (5:6)
@info "used batch sizes: $batch_size_vec"

group = addgroup!(SUITE, bench_name)
for out_type in DTYPES

    local _group = addgroup!(group, "$out_type")


    local __group = addgroup!(_group, problem_size_arg)


    target, proposal, max_val, in_type = generate_setup(problem_arg, out_type, problem_size_arg)

    SAMPLER = RejectionSampler(target, proposal, max_val; backend = BACKEND, in_type = in_type, out_type = out_type)

    for N in nevent_vec
        for batch_size in batch_size_vec
            @info "Adding benchmark problem: dtype=$out_type, problem-size=$problem_size_arg, Neve=$N, Nbatch=$batch_size"

            __group[batch_size][N] = @benchmarkable @sb(
                begin
                    RejectionSamplers.sample_batch_multi_stage!($SAMPLER, batch, output)
                    output.level .= zero(UInt32)
                end
            ) setup = begin
                # Allocate batch buffers
                batch = BatchBuffer(
                    get_backend($SAMPLER),
                    input_type($SAMPLER),
                    output_type($SAMPLER),
                    $batch_size
                )

                # Allocate output buffers
                output = OutBuffer(
                    get_backend($SAMPLER),
                    input_type($SAMPLER),
                    output_type($SAMPLER),
                    $N + $batch_size # one additional batch for safety
                )
            end
        end
    end
end
