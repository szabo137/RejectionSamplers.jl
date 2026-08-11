module RejectionSamplers

# samples
export Sample, SampleVector, value_type, weight_type
export allocate_samples, rand_single

# rng strategies
export HostSide, DeviceSide

# buffers
export AbstractBuffer
export AbstractSampleBuffer
export SampleBuffer
export OutBuffer, BatchBuffer
export value_type, weight_type
export getsample, getsamples, setsample!, setsamples!
export getvalue, getvalues, setvalue!, setvalues!
export getweight, getweights, setweight!, setweights!


# utils
export filter_scan

# abstract sampler
export AbstractSampler, allocate_buffer

# maximum finding
export NaiveMaxFinder, QuantileReductionMethod

# Rejection Sampler
export RejectionSampler
export UniformSampler
export input_type, output_type, proposal_distribution, target_distribution, maximum_value

export sample_multi_stage, sample_single_stage, sample_single_stage_batchless, sample_naive_single_stage


using Distributions
using KernelAbstractions
using Atomix
using Adapt
using Random
using GPUArrays
using StaticArrays
using StaticArrays: sacollect
using StructArrays
using ConstructionBase

include("patches/gpuarrays.jl")

include("samples/interface.jl")
include("samples/generic.jl")
include("samples/impl.jl")

include("buffers/interface.jl")
include("buffers/samplebuffer.jl")

include("filter_scan.jl")

include("target.jl")

include("sampler/interface.jl")
include("sampler/random.jl")
include("sampler/utils.jl")
include("sampler/rejection_sampler.jl")
include("sampler/uniform.jl")


# max finding
# FIXME: remove ProposalDist from MaxFinder
include("max_finder/types.jl")
include("max_finder/findmax.jl")
include("max_finder/naive.jl")
include("max_finder/quantile_reduction.jl")


include("plotting.jl")

include("testutils/TestUtils.jl")


# FIXME: remove ProposalDist from MaxFinder
#include("mocks/Mocks.jl")

end
