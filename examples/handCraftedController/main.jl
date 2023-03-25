using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
using Suppressor

using clfjl
const GUROBI_ENV = Gurobi.Env()


function main()
    N = 2
    x0 = [1., 1.]
    initLB=-1.1
    initUB=1.1
    termLB=-0.01
    termUB=0.01
    boundLB=-2
    boundUB=2
    initSet = clfjl.HyperRectangle(initLB.*ones(N), initUB.*ones(N))
    termSet = clfjl.HyperRectangle(termLB.*ones(N), termUB.*ones(N))
    workspace = clfjl.HyperRectangle(boundLB.*ones(N), boundUB.*ones(N))

    params = clfjl.Parameters(
        optDim=N,
        imgFileDir=joinpath(@__DIR__, "output"),
        maxIteration=30,
        maxLyapunovGapForGenerator=10,
        maxLyapunovGapForVerifier=10,
        thresholdLyapunovGapForGenerator=1e-5,
        thresholdLyapunovGapForVerifier=0,
        print=true,
        padding=true
    )

    A = [1 1;
         0 1]
    K = [-1/2 0;
         0 -1/2]
    A′ = A + K
    b = zeros(N)

    env::clfjl.Env = clfjl.Env(numStateDim=N,
                               numSpaceDim=N,
                               initSet=initSet,
                               termSet=termSet,
                               workspace=workspace,
                               obstacles=[])

    "Setup Gurobi"
    solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
                                            "OutputFlag"=>false))

    function sampleStabilityExample(counterExamples::Vector{clfjl.CounterExample},
                                    x0::Vector{<:Real},
                                    env::clfjl.Env)
        return clfjl.sampleStabilityExample(counterExamples::Vector{clfjl.CounterExample},
                                            x0::Vector{<:Real},
                                            env::clfjl.Env,
                                            A′::Matrix{<:Real},
                                            b::Vector{<:Real})
    end

    "Synthesize Control Lyapunov functions for the given env"
    clfjl.synthesizeCLF(x0, params, env, solver, sampleStabilityExample, clfjl.plot2DCLF)
end


# ------------------------------ Main ------------------------------ #
# X = [x, y, v, theta] in OMPL but we only use [x, y, v] for synthesizing CLFs
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    main()
end
# ------------------------------------------------------------------ #
