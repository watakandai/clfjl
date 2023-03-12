module clfjl

using LinearAlgebra
using JuMP


struct Parameters
    config::Dict{Any, Any}
    execPath::String
    pathFilePath::String
    imgFileDir::String
    startPoint::Vector{Real}
    maxXNorm::Real
    maxIteration::Real
    maxLyapunovGapForGenerator::Real
    maxLyapunovGapForVerifier::Real
    thresholdLyapunovGapForGenerator::Real
    thresholdLyapunovGapForVerifier::Real
    print::Bool
end

struct HyperRectangle
    lb::Vector{Real}
    ub::Vector{Real}
end

struct Dynamics
    A::Matrix{Float64}
    b::Vector{Float64}
    numDim::Integer
end

struct HybridSystem
    dynamics::Dict{Integer, Dynamics}
    numMode::Integer
    numDim::Integer
end

struct Env
    # For generality, we'll keep I & T set as vectors,
    # but generally they should be a single hyperrectagle.
    # initSet::Vector{HyperRectangle}
    # termSet::Vector{HyperRectangle}
    initSet::HyperRectangle
    termSet::HyperRectangle
    workspace::HyperRectangle
    obstacles::Vector{HyperRectangle}
    hybridSystem::HybridSystem
end

struct CounterExample
    x::Vector{Real}
    α::Real
    dynamics::Dynamics
    y::Vector{Real}
    isTerminal::Bool
    isUnsafe::Bool
end

struct LyapunovFunction
    a::Vector{Real}
    b::Real
end

struct JuMPLyapunovFunction
    a::Vector{VariableRef}
    b::VariableRef
end

LyapunovFunctions = Vector{LyapunovFunction}


include("sample.jl")
include("generator.jl")
include("verifier.jl")
include("learner.jl")
include("controlLyapunovFunctions.jl")

end # module
