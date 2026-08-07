# TODO:
# - rename in-type and out-type to value type and weight type
# - fully implement the sampler interface for this!


SUPPORTED_OUT_TYPES = Union{Float16, Float32, Float64}
IN_TYPES_F64 = Union{Float64, SVector{N, Float64}, NTuple{N, Float64}} where {N}
IN_TYPES_F32 = Union{Float32, SVector{N, Float32}, NTuple{N, Float32}} where {N}
IN_TYPES_F16 = Union{Float16, SVector{N, Float16}, NTuple{N, Float16}} where {N}
SUPPORTED_IN_TYPES = Union{IN_TYPES_F16, IN_TYPES_F32, IN_TYPES_F64}

function _assert_compat_io_types(::Type{IN_T}, ::Type{OUT_T}) where {IN_T, OUT_T}
    throw(
        ArgumentError(
            "input type and output type must be compatible"
        )
    )
end
_assert_compat_io_types(
    ::Type{IN_T},
    ::Type{OUT_T}
) where {
    N,
    OUT_T <: SUPPORTED_OUT_TYPES,
    IN_T <: Union{OUT_T, SVector{N, OUT_T}, NTuple{N, OUT_T}},
} = nothing

# TODO: implement proper compat assert
function _assert_compat_target(target, proposal, in_type, out_type) end

_assert_compat_backend(backend::Backend, ::Type{IN_T}, ::Type{OUT_T}) where {IN_T <: SUPPORTED_IN_TYPES, OUT_T <: SUPPORTED_OUT_TYPES} = nothing
function _assert_compat_backend(backend::Backend, ::Type{IN_T}, ::Type{Float64}) where {IN_T <: IN_TYPES_F64}
    return KernelAbstractions.supports_float64(backend)
end


abstract type AbstractRejectionSampler end

struct RejectionSampler{IN_T, OUT_T, TARGET, PROPOSAL, BACKEND} <: AbstractRejectionSampler
    target::TARGET
    proposal::PROPOSAL
    max_value::OUT_T
    backend::BACKEND

    function RejectionSampler(
            target::TARGET,
            proposal::PROPOSAL,
            max_val::OUT_T;
            backend::BACKEND = CPU(), # default backend
            in_type::Type{IN_T},
            out_type::Type{OUT_T}
        ) where {
            IN_T <: SUPPORTED_IN_TYPES,
            OUT_T <: SUPPORTED_OUT_TYPES,
            TARGET <: RejectionSamplers.AbstractTargetDistribution,
            PROPOSAL <: RejectionSamplers.AbstractSampler,
            BACKEND <: KernelAbstractions.Backend,
        }

        _assert_compat_io_types(in_type, out_type)
        _assert_compat_target(target, proposal, in_type, out_type)
        _assert_compat_backend(backend, in_type, out_type)

        return new{IN_T, OUT_T, TARGET, PROPOSAL, BACKEND}(target, proposal, max_val, backend)
    end
end

input_type(::RejectionSampler{IN_T}) where {IN_T} = IN_T
output_type(::RejectionSampler{IN_T, OUT_T}) where {IN_T, OUT_T} = OUT_T
proposal_distribution(eg::RejectionSampler) = eg.proposal
target_distribution(eg::RejectionSampler) = eg.target
RejectionSamplers.maximum_value(eg::RejectionSampler) = eg.max_value
KernelAbstractions.get_backend(eg::RejectionSampler) = eg.backend

### filter scan (directly on buffers)

@kernel inbounds = true function _filter_select(
        max_val,
        batch::BatchBuffer,
        output::OutBuffer,
    )
    local_accepted_count = @localmem Int32 (1,)
    global_accepted_idx = @localmem Int32 (1,)

    global_idx = @index(Global, Linear)
    thread_idx = @index(Local, Linear)

    if thread_idx == 1
        local_accepted_count[1] = 0
    end
    @synchronize

    # filter using randoms
    weight = getweight(batch, global_idx)
    random = batch.u01[global_idx]

    local_accepted_idx = @private Int32 (1,)
    local_accepted_idx[1] = -one(Int32)

    if weight >= max_val * random
        local_accepted_idx[1] = Atomix.@atomic local_accepted_count[1] += 1
    end
    @synchronize

    # increase global output buffer index
    if thread_idx == 1
        temp = local_accepted_count[1]
        global_accepted_idx[1] = Atomix.@atomic output.level[1] += temp
        # this seems pointless but there doesn't seem to be an atomicadd that returns
        # the previous value in Atomix currently
        global_accepted_idx[1] -= local_accepted_count[1]
    end
    @synchronize

    # flush to global output
    if local_accepted_idx[1] != -one(Int32)
        idx1 = global_accepted_idx[1] + local_accepted_idx[1]
        setvalue!(output, getvalue(batch, global_idx), idx1)

        # update weights
        # TODO: consider putting this in a separate function
        setweight!(
            output,
            max(one(weight_type(output)), getweight(batch, global_idx) / max_val),
            idx1
        )
    end
end

### multi stage implementation

function generate_proposals!(
        eg::RejectionSampler,
        buf::BatchBuffer
    )
    rand!(proposal_distribution(eg), buf)
    return nothing
end

function generate_probabilities!(eg::RejectionSampler, batch::BatchBuffer)
    backend = get_backend(eg)
    _gen_prob_kernel!(backend, 32)(
        batch;
        ndrange = size(batch)
    )
    return nothing
end

@kernel inbounds = true function _gen_prob_kernel!(batch)
    I = @index(Global)
    batch.u01[I] = rand(weight_type(batch))
end


# replace the weight
@kernel inbounds = true function _compute_kernel(buf::AbstractSampleBuffer, @Const(dist))
    I = @index(Global)

    @inbounds begin
        x = getvalue!(buf, I)
        setweight!(buf, RejectionSamplers._compute(dist, x), I)
    end
end

# update the weight
@kernel inbounds = true function _compute_update_kernel(buf::AbstractSampleBuffer, @Const(dist))
    I = @index(Global)

    @inbounds begin
        x = getvalue(buf, I)

        proposal_weight = getweight(buf, I)
        setweight!(buf, proposal_weight * RejectionSamplers._compute(dist, x), I)
    end
end


function compute_update!(eg::RejectionSampler, batch::BatchBuffer)
    target = target_distribution(eg)
    backend = get_backend(eg)
    _compute_update_kernel(backend, 32)(
        batch,
        target;
        ndrange = size(batch)
    )

    return nothing
end

function rejection_filter!(
        eg::RejectionSampler,
        batch::BatchBuffer,
        output::OutBuffer,
    )

    max_val = maximum_value(eg)

    backend = get_backend(eg)
    _filter_select(backend, 32)(
        max_val,
        batch,
        output;
        ndrange = length(batch),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function sample_batch_multi_stage!(
        eg::RejectionSampler,
        batch::BatchBuffer,
        output::OutBuffer,
    )

    backend = get_backend(eg)

    @inline generate_proposals!(eg, batch)
    #    KernelAbstractions.synchronize(backend)

    @inline generate_probabilities!(eg, batch)
    #   KernelAbstractions.synchronize(backend)

    @inline compute_update!(eg, batch)
    #  KernelAbstractions.synchronize(backend)

    rejection_filter!(eg, batch, output)
    return nothing
end

function sample_multi_stage!(
        eg::RejectionSampler,
        batch::BatchBuffer,
        output::OutBuffer,
        res_size
    )

    # Main loop
    while true
        sample_batch_multi_stage!(eg, batch, output)

        if Vector(output.level)[1] >= res_size
            break
        end
    end


    return nothing
end

function sample_multi_stage(
        eg::RejectionSampler,
        res_size,
        batch_size,
    )

    # Allocate batch buffers
    batch = BatchBuffer(
        get_backend(eg),
        input_type(eg),
        output_type(eg),
        batch_size
    )

    # Allocate output buffers
    output = OutBuffer(
        get_backend(eg),
        input_type(eg),
        output_type(eg),
        res_size + batch_size # one additional batch for safety
    )

    # in-place sampling
    sample_multi_stage!(eg, batch, output, res_size)

    return output
end


### single kernel

@kernel inbounds = true function sample_batch_kernel(
        target, proposal, max_val, batch, output
    )

    ### 1. generate trials
    batch_idx = @index(Global, Linear) # reuse this! -> BATCH_INDEX
    sample = RejectionSamplers._rand_single(proposal)
    setsample!(batch, sample, batch_idx) # must this be done here?

    ### 2. generate probabilities
    batch.u01[batch_idx] = rand(weight_type(batch)) # must this be written?

    ### 3. compute target and update weight
    I = @index(Global) # reuse from above! -> BATCH_INDEX

    @inbounds begin
        # gets the proposed value from above
        x = getvalue(batch, batch_idx)

        # gets the weight from the proposal above
        proposal_weight = getweight(batch, batch_idx)

        # multiplies the proposal weight with the target weight
        setweight!(batch, proposal_weight * RejectionSamplers._compute(target, x), batch_idx)
    end

    ### 4. filter scan
    local_accepted_count = @localmem Int32 (1,)
    global_accepted_idx = @localmem Int32 (1,)

    thread_idx = @index(Local, Linear)

    if thread_idx == 1
        local_accepted_count[1] = 0
    end
    @synchronize

    # filter using randoms

    # uses updated weights from above
    weight = getweight(batch, batch_idx)

    # uses probabilities from above
    random = batch.u01[batch_idx]

    local_accepted_idx = @private Int32 (1,)
    local_accepted_idx[1] = -one(Int32)

    if weight >= max_val * random
        local_accepted_idx[1] = Atomix.@atomic local_accepted_count[1] += 1
    end
    @synchronize

    # increase global output buffer index
    if thread_idx == 1
        temp = local_accepted_count[1]
        global_accepted_idx[1] = Atomix.@atomic output.level[1] += temp
        # this seems pointless but there doesn't seem to be an atomicadd that returns
        # the previous value in Atomix currently
        global_accepted_idx[1] -= local_accepted_count[1]
    end
    @synchronize

    # flush to global output
    if local_accepted_idx[1] != -one(Int32)
        idx1 = global_accepted_idx[1] + local_accepted_idx[1]

        # copies the batch value to output if excepted
        setvalue!(output, getvalue(batch, batch_idx), idx1)

        # update weights
        # update batch weight and copy to output
        setweight!(
            output,
            max(one(weight_type(output)), getweight(batch, batch_idx) / max_val),
            idx1
        )
    end
end

function sample_single_stage!(
        eg::RejectionSampler,
        batch::BatchBuffer,
        output::OutBuffer,
        res_size
    )

    backend = get_backend(eg)
    target = target_distribution(eg)
    proposal = proposal_distribution(eg)
    max_val = maximum_value(eg)

    sample_kernel = sample_batch_kernel(backend, 32)

    # Main loop
    while true
        sample_kernel(
            target,
            proposal,
            max_val,
            batch,
            output;
            ndrange = length(batch)
        )

        if Vector(output.level)[1] >= res_size
            break
        end
    end


    return nothing
end

function sample_single_stage(
        eg::RejectionSampler,
        res_size,
        batch_size,
    )

    # Allocate batch buffers
    batch = BatchBuffer(
        get_backend(eg),
        input_type(eg),
        output_type(eg),
        batch_size
    )

    # Allocate output buffers
    output = OutBuffer(
        get_backend(eg),
        input_type(eg),
        output_type(eg),
        res_size + batch_size # one additional batch for safety
    )

    # in-place sampling
    sample_single_stage!(eg, batch, output, res_size)

    return output
end

### single kernel (batch less)

@kernel inbounds = true function sample_batchless_kernel(
        target, proposal, max_val, output
    )

    ### 1. generate trials
    trial_sample = RejectionSamplers._rand_single(proposal)

    ### 2. generate probabilities
    u01 = @private weight_type(output) (1,)
    u01[1] = rand(weight_type(output))

    ### 3. compute target and update weight
    # gets the proposed value from above
    proposal_value = @private value_type(output) (1,)
    proposal_value[1] = trial_sample.value
    # gets the weight from the proposal above
    proposal_weight = @private weight_type(output) (1,)
    proposal_weight[1] = trial_sample.weight
    # multiplies the proposal weight with the target weight

    target_weight = @private weight_type(output) (1,)
    target_weight[1] = proposal_weight[1] * RejectionSamplers._compute(target, proposal_value[1])

    ### 4. filter scan
    local_accepted_count = @localmem Int32 (1,)
    global_accepted_idx = @localmem Int32 (1,)

    thread_idx = @index(Local, Linear)

    if thread_idx == 1
        local_accepted_count[1] = 0
    end
    @synchronize

    # rejection filter

    local_accepted_idx = @private Int32 (1,)
    local_accepted_idx[1] = -one(Int32)

    if target_weight[1] >= max_val * u01[1]
        local_accepted_idx[1] = Atomix.@atomic local_accepted_count[1] += 1
    end
    @synchronize

    # increase global output buffer index
    if thread_idx == 1
        temp = local_accepted_count[1]
        global_accepted_idx[1] = Atomix.@atomic output.level[1] += temp
        # this seems pointless but there doesn't seem to be an atomicadd that returns
        # the previous value in Atomix currently
        global_accepted_idx[1] -= local_accepted_count[1]
    end
    @synchronize

    # flush to global output
    if local_accepted_idx[1] != -one(Int32)
        idx1 = global_accepted_idx[1] + local_accepted_idx[1]

        # copies the batch value to output if excepted
        setvalue!(output, proposal_value[1], idx1)

        # update weights
        # update batch weight and copy to output
        setweight!(
            output,
            max(one(weight_type(output)), target_weight[1] / max_val),
            idx1
        )
    end
end

function sample_single_stage_batchless!(
        eg::RejectionSampler,
        output::OutBuffer,
        res_size,
        batch_size
    )

    backend = get_backend(eg)
    target = target_distribution(eg)
    proposal = proposal_distribution(eg)
    max_val = maximum_value(eg)

    sample_kernel = sample_batchless_kernel(backend, 32)

    # Main loop
    while true
        sample_kernel(
            target,
            proposal,
            max_val,
            output;
            ndrange = batch_size
        )

        if Vector(output.level)[1] >= res_size
            break
        end
    end


    return nothing
end

function sample_single_stage_batchless(
        eg::RejectionSampler,
        res_size,
        batch_size
    )

    # Allocate output buffers
    output = OutBuffer(
        get_backend(eg),
        input_type(eg),
        output_type(eg),
        res_size + batch_size # one additional batch for safety
    )

    # in-place sampling
    sample_single_stage_batchless!(eg, output, res_size, batch_size)

    return output
end
