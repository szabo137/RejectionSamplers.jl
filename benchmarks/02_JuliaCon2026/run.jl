import Pkg
using BenchmarkTools
using StaticArrays
using KernelAbstractions
using ArgParse

using RejectionSamplers
using TruncatedGaussians
using OnePhotonEmission

nevent_vec = 2 .^ (5:6)
batch_size_vec = 2 .^ (5:6)

DATADIR = "talk_data"

SAVE_RESULTS = true

PROBLEMS = [
    "truncated-gaussian",
    "one-photon-emission",
]

BACKENDS = [
    "CUDA",
    "oneAPI",
    "AMDGPU",
    "Metal",
    #"OpenCL",
    "CPU",
]

BENCHMARKS = [
    "full",
    "hotloop",
    "batch",
]

IMPLEMENTATIONS = [
    "multi",
    "single-batchful",
    "single-batchless",
    "single-naive",
]
IMPL_FILES = Dict(
    "multi" => "multi_stage.jl",
    "single-batchful" => "single_stage.jl",
    "single-batchless" => "single_stage_batchless.jl",
    "single-naive" => "single_stage_naive.jl"
)

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin

        "--backend", "-b"
        help = "backend used for the benchmarks. Available options: $(join(BACKENDS, ", "))"
        arg_type = String
        default = "CPU"

        "--benchmark", "--bench"
        help = "specify which benchmark to perform. Availabe options are: $(join(BENCHMARKS, ", "))"
        arg_type = String
        default = "full"

        "--implementation", "--impl", "-i"
        help = "specify which implementation to use. Availabe options are: $(join(IMPLEMENTATIONS, ", "))"
        arg_type = String
        default = "multi"

        "--problem-size", "--size", "-s"
        help = "specify the problem size, e.g. the number of dimensions, or the number of particles"
        arg_type = Int
        default = 1

        "--problem", "-p"
        help = "specify the problem to benchmark. Currently, only truncated-gaussian, and one-photon-emission are supported"
        arg_type = String

        "--tune", "-t"
        help = "enable tuning before benchmarking"
        action = :store_true
    end

    return parse_args(s)
end

NOINCLUDE = ["utils.jl"]

### parsing arguments
parsed_args = parse_commandline()

problem_arg = parsed_args["problem"]
problem_arg in PROBLEMS || throw(ArgumentError("\"$problem_arg\" unrecognized as problem! Supported options are $PROBLEMS"))

problem_size_arg = parsed_args["problem-size"]
if problem_arg == "one-photon-emission"
    problem_size_arg <= 4 || throw(ArgumentError("currently, only up to 5 photons are supported."))
end

bench_arg = parsed_args["benchmark"]
bench_arg in BENCHMARKS || throw(ArgumentError("\"$bench_arg\" unrecognized as a benchmark! Supported options are $BENCHMARKS"))

impl_arg = parsed_args["implementation"]
impl_arg in IMPLEMENTATIONS|| throw(ArgumentError("\"$impl_arg\" unrecognized as a implementation! Supported options are $IMPLEMENTATIONS"))

backend_arg = parsed_args["backend"]
backend_arg in BACKENDS || throw(ArgumentError("\"$backend_arg\" unrecognized as a backend! Supported options are $BACKENDS"))

tune_arg = parsed_args["tune"]

println("Parsed args:")

for (arg, val) in parsed_args
    println("  $arg  =>  $val")
end


### Select benchmark and implementation to run
benchmark_path = joinpath("benchmarks", bench_arg, IMPL_FILES[impl_arg])
@info "run benchmark from: $benchmark_path"

### select backend
if backend_arg == "CUDA"
    @info "Try using CUDA backend."
    "CUDA" in keys(Pkg.project().dependencies) ? nothing : Pkg.add("CUDA")

    using CUDA
    CUDA.functional() || throw("CUDA is not functional")
    CUDA.versioninfo()

    const BACKEND = CUDABackend()
    const DTYPES = (Float32, Float64)
    const DEVICE = replace(lowercase(CUDA.name(d)), " " => "-")

    macro sb(ex...)
        return quote
            (CUDA.@sync blocking = true $(esc.(ex)...))
        end
    end

elseif backend_arg == "--oneAPI"
    @info "Try using oneAPI backend."
    throw(ArgumentError("oneAPI is currently not supported, because it has no device side RNG"))

    #=
    "oneAPI" in keys(Pkg.project().dependencies) ? nothing : Pkg.add("oneAPI")

    using oneAPI
    oneAPI.functional() || throw("oneAPI is not functional")
    oneAPI.versioninfo()

    const BACKEND = oneAPIBackend()
    const DTYPES = (Float32,)

    macro sb(ex...)
        return quote
            oneAPI.@sync($(esc.(ex)...))
        end
    end
    =#
elseif backend_arg == "AMDGPU"
    @info "Try using AMDGPU backend."
    "AMDGPU" in keys(Pkg.project().dependencies) ? nothing : Pkg.add("AMDGPU")

    using AMDGPU
    AMDGPU.functional() || throw("AMDGPU is not functional")
    AMDGPU.versioninfo()


    const BACKEND = AMDGPUBackend()
    const DTYPES = (Float32, Float64)
    const DEVICE = replace(lowercase(AMDGPU.HIP.name(AMDGPU.device())), " " => "-")

    macro sb(ex...)
        return quote
            AMDGPU.@sync($(esc.(ex)...))
        end
    end
elseif backend_arg == "Metal"
    @info "Try using Metal backend."
    "Metal" in keys(Pkg.project().dependencies) ? nothing : Pkg.add("Metal")

    using Metal

    Metal.functional() || throw("Metal is not functional")

    Metal.versioninfo()

    const BACKEND = MetalBackend()
    const DTYPES = (Float32,)
    const DEVICE = replace(lowercase(string(Metal.device().name)), " " => "-")

    macro sb(ex...)
        return quote
            Metal.@sync($(esc.(ex)...))
        end
    end

    #=
# TODO: add OpenCL to supported backends
elseif backend_arg == "--OpenCL"
    using OpenCL
    OpenCL.versioninfo()
    const ArrayType = CLArray
    macro sb(ex...) # Not sure how to sync
        quote
            $(esc.(ex)...)
        end
    end
    =#

elseif backend_arg == "CPU"
    @info "Try using CPU backend."

    "InteractiveUtils" in keys(Pkg.project().dependencies) ? nothing : Pkg.add("InteractiveUtils")
    using InteractiveUtils
    InteractiveUtils.versioninfo()

    const BACKEND = CPU()
    const DTYPES = (Float32, Float64)
    const DEVICE = Base.Sys.CPU_NAME

    macro sb(ex...)
        return quote
            $(esc.(ex)...)
        end
    end
end


if backend_arg == "CUDA"
    function reclaim_mem()
        GC.gc(true)
        return CUDA.reclaim()
    end
else
    function reclaim_mem()
        GC.gc(true)
        GC.gc(true)
        return GC.gc(true)
    end
end


include("benchmarks/utils.jl")

SUITE = BenchmarkGroup()
include(benchmark_path)

@info "Preparing benchmarks"

@info "Performing warmup"
warmup(SUITE; verbose = true)

#=
### read "tune" flag from commandline
if tune
    @info "Performing tuning"
    tune!(SUITE, verbose = true)
    ### save tuning parameters
else
    ### load pre-tuned parameters
    ### update parameters of SUITE
end
=#
if tune_arg
    @info "Performing tuning"
    tune!(SUITE, verbose = true)
end

reclaim_mem()

@info "Running benchmarks"
results = run(SUITE, verbose = true)

if SAVE_RESULTS
    data_path = joinpath(DATADIR, problem_arg, backend_arg, bench_arg)
    mkpath(data_path)  # ensure output directory exists

    data_filepath = joinpath(data_path, "bench_$(impl_arg)_size=$(problem_size_arg).json")
    BenchmarkTools.save(data_filepath, results)
    @info "Save results to $data_filepath"
else
    @warn "Results of the benchmark are not saved."
    println(results)
end

# TODO:
# - implement BenchInfo holding at least (backend,dtypes,device)
# - implement a way of pre-tuning and loading pre-tuned parameters (use a flag --tune in
# ArgParse)
