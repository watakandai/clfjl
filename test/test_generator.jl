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

IT = CPC.Image{Float64,Vector{Int}}
WT = CPC.Witness{Vector{Int},Vector{Vector{IT}}}
wit_cls = Vector{WT}[]
lfs_init = Vector{Int}[]
Θ = 5
rmax = 100

@testset "compute lfs empty" begin
    lfs, r = CPC.compute_lfs(wit_cls, lfs_init, 2, 1, Θ, rmax, solver)
    @test isempty(lfs)
    @test r ≈ rmax
end

wit_cls = [[
    CPC.Witness([1], [
        [CPC.Image(9, [0.5])],
        [CPC.Image(5, [0.0]), CPC.Image(12, [1.0])]
    ])
]]
lfs_init = Vector{Int}[]
Θ = 5
rmax = 100

@testset "compute pf loop" begin
    lfs, r = CPC.compute_lfs(wit_cls, lfs_init, 2, 1, Θ, rmax, solver)
    @test length(lfs) == 1
    @test lfs[1] ≈ [1]
    @test r ≈ 0.5/9
end

wit_cls = [[
    CPC.Witness([1], [
        [CPC.Image(9, [0.5])],
        [CPC.Image(5, [0.0]), CPC.Image(8, [0.5])]
    ])
]]
lfs_init = Vector{Int}[]
Θ = 5
rmax = 100

@testset "compute pf loop" begin
    lfs, r = CPC.compute_lfs(wit_cls, lfs_init, 2, 1, Θ, rmax, solver)
    @test length(lfs) == 1
    @test lfs[1] ≈ [1]
    @test r ≈ 0.5/8
end

wit_cls = [[
    CPC.Witness([1], [
        [CPC.Image(9, [-0.5])],
        [CPC.Image(12, [-1.0])]
    ])
]]
lfs_init = Vector{Int}[]
Θ = 5
rmax = 100

@testset "compute pf no loop" begin
    lfs, r = CPC.compute_lfs(wit_cls, lfs_init, 2, 1, Θ, rmax, solver)
    @test length(lfs) == 1
    @test lfs[1] ≈ [1]
    @test r ≈ 2/12
end

wit_cls = [[
    CPC.Witness([1], [
        [CPC.Image(9, [-0.5])],
        [CPC.Image(12, [-1.0])]
    ])
]]
lfs_init = [[-0.25], [0.25]]
Θ = 5
rmax = 100

@testset "compute pf init active" begin
    lfs, r = CPC.compute_lfs(wit_cls, lfs_init, 2, 1, Θ, rmax, solver)
    @test length(lfs) == 1
    @test lfs[1] ≈ [1]
    @test r ≈ (1 - 0.25*0.5)/9
end

wit_cls = [
    [CPC.Witness([1], [
            [CPC.Image(9, [-3.0])],
            [CPC.Image(9, [-1.0])]
    ])], 
    [CPC.Witness([-1], [
            [CPC.Image(9, [-2.0])],
            [CPC.Image(9, [-3.0])]
    ])]
]
lfs_init = [[-0.1], [0.1]]
Θ = 5
rmax = 100

@testset "compute pf loop" begin
    lfs, r = CPC.compute_lfs(wit_cls, lfs_init, 2, 1, Θ, rmax, solver)
    @test length(lfs) == 2
    @test lfs[1] ≈ [1]
    @test lfs[2] ≈ [-0.1]
    @test r ≈ -0.1/9
end

wit_cls = [
    [CPC.Witness([1], [
            [CPC.Image(9, [-3.0])],
            [CPC.Image(9, [-1.0])]
    ])], 
    [CPC.Witness([-1], [
            [CPC.Image(0, [0.0])],
            [CPC.Image(0, [0.0])]
    ])]
]
lfs_init = [[-0.1], [0.1]]
Θ = 5
rmax = 100

@testset "compute pf cycle" begin
    lfs, r = CPC.compute_lfs(wit_cls, lfs_init, 2, 1, Θ, rmax, solver)
    @test length(lfs) == 2
    @test lfs[1] ≈ [1]
    @test norm(lfs[2]) < 1e-6
    @test r ≈ (1 - 0.1)/9
end

wit_cls = [
    [CPC.Witness([1], [
            [CPC.Image(9, [-3.0])],
            [CPC.Image(9, [-1.0])]
    ])], 
    [
        CPC.Witness([-1], [
            [CPC.Image(0, [0.0])],
            [CPC.Image(0, [0.0])]
        ]),
        CPC.Witness([-1], [
            [CPC.Image(9, [0.5])],
            [CPC.Image(9, [-3.0])]
        ])
    ]
]
lfs_init = [[-0.1], [0.1]]
Θ = 5
rmax = 100

@testset "compute pf cycle" begin
    lfs, r = CPC.compute_lfs(wit_cls, lfs_init, 2, 1, Θ, rmax, solver)
    @test length(lfs) == 2
    @test lfs[1] ≈ [1]
    @test lfs[2] ≈ [-0.75]
    @test r ≈ 0.25/9
end

nothing