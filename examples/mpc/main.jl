using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
using Suppressor

# using clfjl
include("../../src/clfjl.jl")
const GUROBI_ENV = Gurobi.Env()


function main(;x0::Vector{<:Real},
               initLB::Union{Vector{<:Real}, <:Real},
               initUB::Union{Vector{<:Real}, <:Real},
               termLB::Union{Vector{<:Real}, <:Real},
               termUB::Union{Vector{<:Real}, <:Real},
               boundLB::Union{Vector{<:Real}, <:Real},
               boundUB::Union{Vector{<:Real}, <:Real},
               inputLB::Union{Vector{<:Real}, <:Real},
               inputUB::Union{Vector{<:Real}, <:Real})

    N = length(x0)

    # Constraints
    initLBs = isa(initLB, Vector{<:Real}) ? initLB : initLB.*ones(N)
    initUBs = isa(initUB, Vector{<:Real}) ? initUB : initUB.*ones(N)
    termLBs = isa(termLB, Vector{<:Real}) ? termLB : termLB.*ones(N)
    termUBs = isa(termUB, Vector{<:Real}) ? termUB : termUB.*ones(N)
    boundLBs = isa(boundLB, Vector{<:Real}) ? boundLB : boundLB.*ones(N)
    boundUBs = isa(boundUB, Vector{<:Real}) ? boundUB : boundUB.*ones(N)
    inputLBs = isa(inputLB, Vector{<:Real}) ? inputLB : inputLB.*ones(N)
    inputUBs = isa(inputUB, Vector{<:Real}) ? inputUB : inputUB.*ones(N)
    initSet = clfjl.HyperRectangle(initLBs, initUBs)
    termSet = clfjl.HyperRectangle(termLBs, termUBs)
    workspace = clfjl.HyperRectangle(boundLBs, boundUBs)
    inputSet = clfjl.HyperRectangle(inputLBs, inputUBs)

    params = clfjl.Parameters(
        optDim=N,
        imgFileDir=joinpath(@__DIR__, "output$(N)D"),
        lfsFileDir=@__DIR__,
        maxIteration=1000,
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

    "Synthesize Control Lyapunov functions for the given env"
    function sampleSimpleCar(counterExamples::Vector{clfjl.CounterExample},
                             x0::Vector{<:Real},
                             env::clfjl.Env)
            # Ad = [1 0 1 0;
            #       0 1 1/2 1/2;
            #       0 0 1 0;
            #       0 0 0 1;]
            # Bd = [0 0;
            #       0 0;
            #       1 0;
            #       0 1]
            velocity = 0.1
            Ad = [1 0;
                  velocity 1]
            Bd = [1, 0][:, :]       # [:,:] converts vec to matrix
            # X = [x, y, θ], U = [v=1, ω]
            # Ad = [1 0 0;            # x' = x + v
            #       0 1 velocity;     # y' = y + vθ
            #       0 0 1;]           # θ' = θ + ω
            # Bd = [velocity 0;
            #       0 0;
            #       0 1;]
        return clfjl.sampleSimpleCar(counterExamples, x0, env, Ad, Bd, inputSet)
    end

    clfjl.synthesizeCLF(x0, params, env, solver, sampleSimpleCar, clfjl.plot2DCLF)
    # clfjl.synthesizeCLF(x0, params, env, solver, sampleSimpleCar, clfjl.plot3DCLF)
    # clfjl.synthesizeCLF(x0, params, env, solver, sampleSimpleCar)
end


# ------------------------------ Main ------------------------------ #
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    ## 4D: X=[x, y, v, θ] , U=[a, ω]
    # main(x0=[0., 0., 0., 0.],
    #      initLB=[0., -0.3, 0, 0],
    #      initUB=[0.,  0.3, 0, 0],
    #      termLB=[1.0, -0.1, -0.1, -0.5],
    #      termUB=[1.1,  0.1,  0.1,  0.5],
    #      boundLB=[-1.5, -0.5, -1.0, -pi],
    #      boundUB=[ 1.5,  0.5,  1.0, pi],
    #      inputLB=[-0.1, -0.1],
    #      inputUB=[ 0.1,  0.1])
    ## 2D: X=[θ, y] , U=[ω]
    main(x0=[pi/12, 0.5],
         initLB=[-pi/12, 0.5],
         initUB=[ pi/12, 0.6],
         termLB=[-pi/4, -0.15],
         termUB=[ pi/4,  0.15],
         boundLB=[-pi/2, -1.0],
         boundUB=[ pi/2,  1.0],
         inputLB=[-1],
         inputUB=[ 1])
    ## 3D: X=[x, y, θ], U=[v=1, ω]
    # main(x0=[0., 0.55, pi/12],
    #      initLB=[-0.1, 0.5, -pi/12],
    #      initUB=[ 0.1, 0.6,  pi/12],
    #      termLB=[2.0, -0.2, -pi/6],
    #      termUB=[4.0,  0.2,  pi/6],
    #      boundLB=[-1.0, -0.3, -pi/2],
    #      boundUB=[ 5.0,  1.5,  pi/2],
    #      inputLB=[1.0, -pi/12],
    #      inputUB=[1.0,  pi/12])
end
# ------------------------------------------------------------------ #
