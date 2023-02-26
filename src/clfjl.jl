module clfjl

using LinearAlgebra
using JuMP


struct Parameters
    execPath::String
    pathFilePath::String
    startPoint::Vector{Real}
    maxXNorm::Real
    maxIteration::Real
    maxLyapunovGapForGenerator::Real
    maxLyapunovGapForVerifier::Real
    thresholdLyapunovGapForGenerator::Real
    thresholdLyapunovGapForVerifier::Real
    do_print::Bool
end

struct Obstacle
    lb::Vector{Float64}
    ub::Vector{Float64}
end

struct Workspace
    lb::Vector{Float64}
    ub::Vector{Float64}
    obstacles::Vector{Obstacle}
end


struct Dynamics
    A::Matrix{Float64}
    b::Vector{Float64}
    numDim::Int64
end

struct HybridSystem
    dynamics::Dict{Int64, Dynamics}
    numMode::Int64
    numDim::Int64
end

struct CounterExample
    x::Vector{Float64}
    α::Float64
    dynamics::Dynamics
    y::Vector{Float64}
end

struct Rectangle{VT}
    lb::VT
    ub::VT
end

Base.in(rect::Rectangle, x) =
    all(t -> t[1] ≤ t[2] ≤ t[3], zip(rect.lb, x, rect.ub))

struct Flow{AT<:AbstractMatrix,BT<:AbstractVector}
    A::AT
    b::BT
end

struct Piece{VFT<:Vector{<:Flow},RT<:Rectangle}
    flows::VFT
    rect::RT
end

struct Image{AT<:Real,YT<:AbstractVector,FT<:Flow}
    α::AT
    y::YT
    flow::FT
end

struct Witness{XT<:AbstractVector,VVIT<:Vector{<:Vector{<:Image}}}
    x::XT
    img_cls::VVIT
end



include("sample.jl")
include("generator.jl")
include("verifier.jl")
include("learner.jl")

end # module
