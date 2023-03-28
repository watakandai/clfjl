module clfjl

using LinearAlgebra
using JuMP
import Base.@kwdef

@kwdef struct Parameters
    optDim::Integer
    imgFileDir::String
    lfsFileDir::String
    maxIteration::Real
    maxLyapunovGapForGenerator::Real
    maxLyapunovGapForVerifier::Real
    thresholdLyapunovGapForGenerator::Real
    thresholdLyapunovGapForVerifier::Real
    print::Bool
    padding::Bool
    omplConfig::Dict{Any, Any}=Dict()
end


struct HyperRectangle
    lb::Vector{<:Real}
    ub::Vector{<:Real}
end


struct Dynamics
    A::Matrix{<:Real}
    b::Vector{<:Real}
    numDim::Integer
end


@kwdef struct HybridSystem
    dynamics::Dict{Any, Dynamics}
    numMode::Integer
    numDim::Integer
end


@kwdef struct Env
    # For generality, we'll keep I & T set as vectors,
    # but generally they should be a single hyperrectagle.
    numStateDim::Integer
    numSpaceDim::Integer
    initSet::HyperRectangle
    termSet::HyperRectangle
    workspace::HyperRectangle
    obstacles::Vector{HyperRectangle}
    # hybridSystem::HybridSystem
end


mutable struct CounterExample
    x::Vector{<:Real}
    α::Real
    dynamics::Dynamics
    y::Vector{<:Real}
    isTerminal::Bool
    isUnsafe::Bool
end


struct LyapunovFunction
    a::Vector{<:Real}
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
include("plotFunc.jl")
include("prechecker.jl")
include("decomposition.jl")
include("simulator.jl")

end # module
