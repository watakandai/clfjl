using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGISPolyhedralControl.jl")
else
    using CEGISPolyhedralControl
end
CPC = CEGISPolyhedralControl

solver() = Model(optimizer_with_attributes(
    HiGHS.Optimizer, "output_flag"=>false
))

flows = CPC.Flow{Matrix{Int},Vector{Int}}[]
rect = CPC.Rectangle([1, 1], [2, 2])
lfs = [[1, 1], [1, -1], [-10, 10], [-10, -10]]

Θ = 5
γmax = 100

@testset "piece: empty flows" begin
    x, γ = CPC.verify_piece(flows, rect, lfs, 0, 2, Θ, Θ, γmax, solver)
    @test γ ≈ γmax
end

flows = [
    CPC.Flow([0 -1; 0 0], [2, 0]),
    CPC.Flow([0 1; 0 0], [-1, 0])
]
rect = CPC.Rectangle([1, 1], [2, 2])
lfs = [[1, 1]]

Θ = 400
γmax = 100

@testset "piece: single lfs" begin
    x, γ = CPC.verify_piece(flows, rect, lfs, 2, 2, Θ, Θ, γmax, solver)
    @test x ≈ [0.4, 0.6]
    @test γ ≈ 0.2
end

flows = [
    CPC.Flow([0 -1; 0 0], [1, 0]),
    CPC.Flow([0 1; 0 0], [-1, 0])
]
rect = CPC.Rectangle([1, 1], [2, 2])
lfs = [[1, 1], [1, -1], [-10, 10], [-10, -10]]

Θ = 400
γmax = 100

@testset "piece: multi lfs" begin
    x, γ = CPC.verify_piece(flows, rect, lfs, 2, 2, Θ, Θ, γmax, solver)
    @test x ≈ [18/40, 22/40]
    @test γ ≈ 11/40
end

flows = [
    CPC.Flow([0 -1; -1 0], [1, 0]),
    CPC.Flow([0 1; 0 0], [-1, 0])
]
rect = CPC.Rectangle([1, 1], [2, 2])
lfs = [[1, 1], [1, -1], [-10, 10], [-10, -10]]

Θ = 400
γmax = 100

@testset "piece: multi lfs neg" begin
    x, γ = CPC.verify_piece(flows, rect, lfs, 2, 2, Θ, Θ, γmax, solver)
    @test x ≈ [1/2, 1/2]
    @test γ ≈ -1/2
end

FT = CPC.Flow{Matrix{Int},Vector{Int}}
pieces = CPC.Piece{Vector{FT},CPC.Rectangle{Vector{Int}}}[]
lfs = [[1, 1], [1, -1], [-10, 10], [-10, -10]]

Θ = 400
γmax = 100

@testset "all: empty pieces" begin
    x, γ = CPC.verify(pieces, lfs, 2, 2, Θ, Θ, γmax, solver)
    @test all(isnan, x)
    @test γ ≈ -Inf
end

pieces = [
    CPC.Piece([
        CPC.Flow([0 -1; -1 0], [1, 0]),
        CPC.Flow([0 1; 0 0], [-1, 0])
    ], CPC.Rectangle([1, 1], [2, 2])),
    CPC.Piece([
        CPC.Flow([0 0.5; 0.5 0], [0.5, 0]),
        CPC.Flow([0 -0.5; 0 0], [-0.5, 0])
    ], CPC.Rectangle([1, -2], [2, -1]))
]
lfs = [[1, 1], [1, -1], [-10, 10], [-10, -10]]

Θ = 400
γmax = 100

@testset "all: neg" begin
    x, γ, k = CPC.verify(pieces, lfs, 2, 2, Θ, Θ, γmax, solver)
    @test x ≈ [1/2, -1/2]
    @test γ ≈ -1/4
    @test k == 2
end

pieces = [
    CPC.Piece([
        CPC.Flow([0 -1; 0 0], [1, 0]),
        CPC.Flow([0 1; 0 0], [-1, 0])
    ], CPC.Rectangle([1, 1], [2, 2])),
    CPC.Piece([
        CPC.Flow([0 2; 0 0], [2, 0]),
        CPC.Flow([0 -2; 0 0], [-2, 0])
    ], CPC.Rectangle([1, -2], [2, -1]))
]
lfs = [[1, 1], [1, -1], [-10, 10], [-10, -10]]

Θ = 400
γmax = 100

@testset "all: pos" begin
    x, γ, k = CPC.verify(pieces, lfs, 2, 2, Θ, Θ, γmax, solver)
    @test x ≈ [18/40, -22/40]
    @test γ ≈ 22/40
    @test k == 2
end

nothing