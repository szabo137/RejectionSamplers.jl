"""
    SampleBuffer{Tv, Tw, S} <: AbstractSampleBuffer

Concrete implementation of `AbstractSampleBuffer` that stores weighted
samples in a structure-of-arrays layout.

`SampleBuffer` represents a one-dimensional buffer of `Sample{Tv,Tw}`
elements, where values and weights are stored in separate contiguous
arrays via a `StructVector`. This layout enables efficient
field-wise access and mutation, and is suitable for both CPU and GPU
backends.

### Type parameters
- `Tv`: value type of the samples
- `Tw`: weight type of the samples
- `S`: underlying storage type (typically `StructVector{Sample{Tv,Tw}}`)

### Constructors

- `SampleBuffer(samples)`

    Wrap an existing `StructVector{Sample}` as a `SampleBuffer`. The input
    storage is used directly without copying and must already reside on
    the desired backend.


- `SampleBuffer(values, weights)`

    Construct a `SampleBuffer` from separate value and weight vectors.
    The two vectors must have the same length and compatible backends.
    Internally, the data is stored in a structure-of-arrays representation.


- `SampleBuffer(backend, valuetype, weighttype, size)`

    Allocate a new `SampleBuffer` of the given `size` on the specified
    `backend`, with sample values of type `valuetype` and weights of type
    `weighttype`. Storage allocation is backend-aware.

### Notes

`SampleBuffer` satisfies the full `AbstractSampleBuffer` interface and
can be used interchangeably with other sample buffer implementations in
samplers and kernel-based algorithms.
"""
struct SampleBuffer{Tv, Tw, S} <: AbstractSampleBuffer
    samples::S

    function SampleBuffer(samples::S) where {Tv, Tw, S <: StructVector{Sample{Tv, Tw}}}
        return new{Tv, Tw, S}(samples)
    end

    function SampleBuffer(vals::V, ws::W) where {Tv, Tw, V <: AbstractVector{Tv}, W <: AbstractVector{Tw}}
        S = StructArray{Sample{Tv, Tw}}((vals, ws))
        return new{Tv, Tw, typeof(S)}(S)
    end

    function SampleBuffer(backend, valuetype, weighttype, size)
        struct_arr = allocate_samples(backend, valuetype, weighttype, size)
        return new{valuetype, weighttype, typeof(struct_arr)}(struct_arr)
    end
end

Adapt.@adapt_structure SampleBuffer

@inline value_type(buf::SampleBuffer{Tv}) where {Tv} = Tv
@inline weight_type(buf::SampleBuffer{Tv, Tw}) where {Tv, Tw} = Tw

KernelAbstractions.get_backend(buf::SampleBuffer) = get_backend(buf.samples.value)

Base.length(buf::SampleBuffer) = length(buf.samples)
Base.eltype(buf::SampleBuffer{Tv, Tw}) where {Tv, Tw} = Sample{Tv, Tw}
Base.getindex(buf::SampleBuffer, idx) = buf.samples[idx]
function Base.setindex!(
        buf::SampleBuffer{Tv, Tw},
        sample::Sample{Tv, Tw},
        idx,
    ) where {Tv, Tw}
    return buf.samples[idx] = sample
end

getsample(buf::SampleBuffer, idx) = buf[idx]
setsample!(buf::SampleBuffer, sample, idx) = (buf[idx] = sample)

getvalue(buf::SampleBuffer, idx) = buf.samples.value[idx]
setvalue!(buf::SampleBuffer, value, idx) = (buf.samples.value[idx] = value)

getweight(buf::SampleBuffer, idx) = buf.samples.weight[idx]
setweight!(buf::SampleBuffer, weight, idx) = (buf.samples.weight[idx] = weight)


### Output samples

struct OutBuffer{Tv, Tw, L, S} <: AbstractSampleBuffer
    samples::S
    level::L

    function OutBuffer(samples::S, level::L) where {Tv, Tw, L, S <: StructVector{Sample{Tv, Tw}}}
        return new{Tv, Tw, L, S}(samples, level)
    end

    function OutBuffer(vals::V, ws::W, level::L) where {Tv, Tw, L, V <: AbstractVector{Tv}, W <: AbstractVector{Tw}}
        S = StructArray{Sample{Tv, Tw}}((vals, ws))
        return new{Tv, Tw, L, typeof(S)}(S, level)
    end

    function OutBuffer(backend, valuetype, weighttype, size)
        struct_arr = allocate_samples(backend, valuetype, weighttype, size)
        level = KernelAbstractions.zeros(backend, UInt32, 1)
        return new{valuetype, weighttype, typeof(level), typeof(struct_arr)}(struct_arr, level)
    end
end

Adapt.@adapt_structure OutBuffer

@inline value_type(buf::OutBuffer{Tv}) where {Tv} = Tv
@inline weight_type(buf::OutBuffer{Tv, Tw}) where {Tv, Tw} = Tw

KernelAbstractions.get_backend(buf::OutBuffer) = get_backend(buf.samples.value)

Base.length(buf::OutBuffer) = length(buf.samples)
Base.eltype(buf::OutBuffer{Tv, Tw}) where {Tv, Tw} = Sample{Tv, Tw}
Base.getindex(buf::OutBuffer, idx) = buf.samples[idx]
function Base.setindex!(
        buf::OutBuffer{Tv, Tw},
        sample::Sample{Tv, Tw},
        idx,
    ) where {Tv, Tw}
    return buf.samples[idx] = sample
end

getsample(buf::OutBuffer, idx) = buf[idx]
setsample!(buf::OutBuffer, sample, idx) = (buf[idx] = sample)

getvalue(buf::OutBuffer, idx) = buf.samples.value[idx]
setvalue!(buf::OutBuffer, value, idx) = (buf.samples.value[idx] = value)

getweight(buf::OutBuffer, idx) = buf.samples.weight[idx]
setweight!(buf::OutBuffer, weight, idx) = (buf.samples.weight[idx] = weight)


### Batch samples

struct BatchBuffer{Tv, Tw, U, S} <: AbstractSampleBuffer
    samples::S
    u01::U

    function BatchBuffer(samples::S, u01::U) where {Tv, Tw, S <: StructVector{Sample{Tv, Tw}}, U <: AbstractVector{Tw}}
        return new{Tv, Tw, U, S}(samples, u01)
    end

    function BatchBuffer(vals::V, ws::W, u01::U) where {Tv, Tw, V <: AbstractVector{Tv}, W <: AbstractVector{Tw}, U <: AbstractVector{Tw}}
        S = StructArray{Sample{Tv, Tw}}((vals, ws))
        return new{Tv, Tw, U, typeof(S)}(S, u01)
    end

    function BatchBuffer(backend, valuetype, weighttype, size)
        struct_arr = allocate_samples(backend, valuetype, weighttype, size)
        u01 = allocate(backend, weighttype, (size,))
        return new{valuetype, weighttype, typeof(u01), typeof(struct_arr)}(struct_arr, u01)
    end
end

Adapt.@adapt_structure BatchBuffer

@inline value_type(buf::BatchBuffer{Tv}) where {Tv} = Tv
@inline weight_type(buf::BatchBuffer{Tv, Tw}) where {Tv, Tw} = Tw

KernelAbstractions.get_backend(buf::BatchBuffer) = get_backend(buf.samples.value)

Base.length(buf::BatchBuffer) = length(buf.samples)
Base.eltype(buf::BatchBuffer{Tv, Tw}) where {Tv, Tw} = Sample{Tv, Tw}
Base.getindex(buf::BatchBuffer, idx) = buf.samples[idx]
function Base.setindex!(
        buf::BatchBuffer{Tv, Tw},
        sample::Sample{Tv, Tw},
        idx,
    ) where {Tv, Tw}
    return buf.samples[idx] = sample
end

getsample(buf::BatchBuffer, idx) = buf[idx]
setsample!(buf::BatchBuffer, sample, idx) = (buf[idx] = sample)

getvalue(buf::BatchBuffer, idx) = buf.samples.value[idx]
setvalue!(buf::BatchBuffer, value, idx) = (buf.samples.value[idx] = value)

getweight(buf::BatchBuffer, idx) = buf.samples.weight[idx]
setweight!(buf::BatchBuffer, weight, idx) = (buf.samples.weight[idx] = weight)
