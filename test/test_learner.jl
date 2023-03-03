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

α = 0.5
pieces = [CPC.Piece([
        CPC.Flow([0 -α; 1/α 0], [0, 0]),
        CPC.Flow([0 -1/α; α 0], [0, 0])
    ], CPC.Rectangle([-1, -1], [1, 1]))]

lfs_init = [[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]

τ = π/20
xmax = 15
γmax = 30
iter_max = 10

status, lfs = CPC.learn_controller(
    pieces, lfs_init, τ, 2, 2, xmax, γmax, iter_max, solver
)

@testset "iter max" begin
    @test status == CPC.MAX_ITER_REACHED
end

τ = π/20
xmax = 15
γmax = 30
iter_max = 100

status, lfs = CPC.learn_controller(
    pieces, lfs_init, τ, 2, 2, xmax, γmax, iter_max, solver
)

@testset "found" begin
    @test status == CPC.CONTROLLER_FOUND
end

α = 1.1
pieces = [CPC.Piece([
        CPC.Flow([α -1; 1 α], [0, 0]),
    ], CPC.Rectangle([-1, -1], [1, 1]))]

lfs_init = [[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]

τ = π/20
xmax = 15
γmax = 30
iter_max = 100

status, lfs = CPC.learn_controller(
    pieces, lfs_init, τ, 1, 2, xmax, γmax, iter_max, solver
)

@testset "infeasible" begin
    @test status == CPC.CONTROLLER_INFEASIBLE
end

nothing