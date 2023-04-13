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
    obstacles::Vector{HyperRectangle} = HyperRectangle[]
end


mutable struct CounterExample
    x::Vector{<:Real}
    α::Real
    dynamics::Dynamics
    y::Vector{<:Real}
    isTerminal::Bool
    isUnsafe::Bool
    ith::Integer
end

CounterExamples = Vector{CounterExample}


# A trajectory sampled from a real system
@enum SampleStatus begin
    TRAJ_FOUND = 0
    TRAJ_INFEASIBLE = 1
    TRAJ_UNSAFE = 2
    TRAJ_MAX_ITER_REACHED = 3
end
struct SampleTrajectry
    X::Vector{Vector{<:Real}}
    U::Vector{Vector{<:Real}}
    status::SampleStatus
end

SampleTrajectries = Vector{SampleTrajectry}

# A trajectory simulated using the "learned" (Voronoi) controller
@enum SimStatus begin
    SIM_TERMINATED = 0
    SIM_INFEASIBLE = 1
    SIM_UNSAFE = 2
    SIM_MAX_ITER_REACHED = 3
end

struct SimTrajectory
    X::Vector{Vector{<:Real}}
    V::Vector{<:Real}
    status::SimStatus
end

SimTrajectories = Vector{SimTrajectory}


struct LyapunovFunction
    a::Vector{<:Real}
    b::Real
end


LyapunovFunctions = Vector{LyapunovFunction}


struct JuMPLyapunovFunction
    a::Vector{VariableRef}
    b::VariableRef
end


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
