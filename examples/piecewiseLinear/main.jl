using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
using Suppressor

# using clfjl
include("../../src/clfjl.jl")
const GUROBI_ENV = Gurobi.Env()


function main(;initLBs::Vector{<:Real},
               initUBs::Vector{<:Real},
               termLBs::Vector{<:Real},
               termUBs::Vector{<:Real},
               boundLBs::Vector{<:Real},
               boundUBs::Vector{<:Real},
               N::Integer,
               example::String,
               lines=Vector{Tuple{Vector{Float64}, Vector{Float64}}}())

    # Constraints
    initSet = clfjl.HyperRectangle(initLBs, initUBs)
    termSet = clfjl.HyperRectangle(termLBs, termUBs)
    workspace = clfjl.HyperRectangle(boundLBs, boundUBs)

    params = clfjl.Parameters(
        optDim=N,
        imgFileDir=joinpath(@__DIR__, example),
        lfsFileDir=@__DIR__,
        maxIteration=200,
        maxLyapunovGapForGenerator=10,
        maxLyapunovGapForVerifier=10,
        thresholdLyapunovGapForGenerator=1e-12,
        thresholdLyapunovGapForVerifier=0,
        print=true,
        padding=true
    )

    env::clfjl.Env = clfjl.Env(numStateDim=N,
                               numSpaceDim=N,
                               initSet=initSet,
                               termSet=termSet,
                               workspace=workspace,
                               obstacles=[])

    "Setup Gurobi"
    solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
                                            "OutputFlag"=>false))

    if  example == "BuckConverter"
        sampleFunc = clfjl.sampleBuckConverter
    elseif example == "SwitchStable"
        sampleFunc = clfjl.sampleSwitchStable
    elseif example == "MinFunc"
        sampleFunc = clfjl.sampleMinFunc
    end

    clfjl.synthesizeCLF(lines, params, env, solver, sampleFunc, clfjl.plot2DCLF)
end


# ------------------------------ Main ------------------------------ #
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    main(initLBs=[-0.5, -0.5],
         initUBs=[ 0.5,  0.5],
         termLBs=[-0.1, -0.1],
         termUBs=[ 0.1,  0.1],
         boundLBs=[-2.0, -4.0],
         boundUBs=[ 2.0,  4.0],
         N=2,
         example="MinFunc") # options=[BuckConverter, SwitchStable, MinFunc]
end
# ------------------------------------------------------------------ #
