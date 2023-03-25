using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
using Suppressor

using clfjl
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
        maxIteration=100,
        maxLyapunovGapForGenerator=10,
        maxLyapunovGapForVerifier=10,
        thresholdLyapunovGapForGenerator=1e-5,
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
            # Ad = [1 0;
            #       velocity 1]
            # Bd = [1, 0][:, :]       # [:,:] converts vec to matrix
            # X = [x, y, θ], U = [v=1, ω]
            Ad = [1 0 0;            # x' = x + v
                  0 1 velocity;     # y' = y + vθ
                  0 0 1;]           # θ' = θ + ω
            Bd = [velocity 0;
                  0 0;
                  0 1;]
        return clfjl.sampleSimpleCar(counterExamples, x0, env, Ad, Bd, inputSet)
    end

    # clfjl.synthesizeCLF(params, env, solver, sampleSimpleCar, clfjl.plot2DCLF)
    # clfjl.synthesizeCLF(x0, params, env, solver, sampleSimpleCar, clfjl.plot3DCLF)
    clfjl.synthesizeCLF(x0, params, env, solver, sampleSimpleCar)
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
    # main(x0=[pi/4, 0.2],
    #      initLB=[-pi/4, 0.1],
    #      initUB=[ pi/4, 0.3],
    #      termLB=[-pi/12, -0.05],
    #      termUB=[ pi/12,  0.05],
    #      boundLB=[-pi/2, -0.1],
    #      boundUB=[ pi/2,  1.0],
    #      inputLB=[-0.1],
    #      inputUB=[ 0.1])
    ## 3D: X=[x, y, θ], U=[v=1, ω]
    main(x0=[0., 0.2, pi/4],
         initLB=[-0.05, 0.1, -pi/4],
         initUB=[ 0.05, 0.3,  pi/4],
         termLB=[2.0, -0.05, -pi/6],
         termUB=[4.0,  0.05,  pi/6],
         boundLB=[-1.0, -0.3, -pi/2],
         boundUB=[ 5.0,  1.0,  pi/2],
         inputLB=[1.0, -pi/12],
         inputUB=[1.0,  pi/12])
end
# ------------------------------------------------------------------ #
