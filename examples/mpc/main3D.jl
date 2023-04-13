using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
using JLD2
using Suppressor
using SparseArrays
speye(N) = spdiagm(ones(N))

include("../../src/clfjl.jl")
const GUROBI_ENV = Gurobi.Env()


function main(;initLBs::Vector{<:Real},
               initUBs::Vector{<:Real},
               termLBs::Vector{<:Real},
               termUBs::Vector{<:Real},
               boundLBs::Vector{<:Real},
               boundUBs::Vector{<:Real},
               inputLBs::Vector{<:Real},
               inputUBs::Vector{<:Real},
               N::Integer,
               lines=Vector{Tuple{Vector{Float64}, Vector{Float64}}}())

    # Constraints
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


    "Sample a trajectory for the Dubin's Car model using the MPC contoller"
    function sampleSimpleCar3D(counterExamples::Vector{clfjl.CounterExample},
        x0::Vector{<:Real},
        env::clfjl.Env;
        xT::Vector{<:Real}=Float64[])

        # Dynamics: X = [θ, y, x], U = [v=1, ω]
        velocity = 0.1
        Ad = [1 0 0;            # θ' = θ + ω
              velocity 1 0;     # y' = y + vθ
              0 0 1;]           # x' = x + v
        Bd = [0 1;
              0 0;
              velocity 0;]
        # Weights
        (nx, nu) = size(Bd)
        Q = spdiagm([0.1, 1, 0.0])           # Weights for Xs from 0:N-1
        QN = 1 * Q                          # Weights for the terminal state X at N (Xn or xT)
        R = 1 * spdiagm([0.0, 1])
        RD = 1 * spdiagm([0.0, 1])
        numHorizon = 10

        function simulateSimpleCar(x0_, xT_, env_, numStep_)
            return clfjl.simulateMPC(x0_, xT_, env_, numStep_,
                            Ad, Bd, Q, R, QN, RD, numHorizon, inputSet,
                            useSet=false)
        end

        return clfjl.sampleCounterExample(counterExamples, x0, env;
                    xT=xT, simulateFunc=simulateSimpleCar)
    end

    clfjl.synthesizeCLF(lines, params, env, solver, sampleSimpleCar3D, clfjl.plot3DCLF)
end


# ------------------------------ Main ------------------------------ #
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    ## 3D: X=[θ, y, x], U=[v=1, ω]
    main(initLBs=[-pi/3, -0.3, 0.0],
         initUBs=[ pi/3,  0.3, 0.0],
         termLBs=[-pi/12, -0.3, 0.0],
         termUBs=[ pi/12,  0.3, 5.0],
         boundLBs=[-pi/2, -1.0, -1.0],
         boundUBs=[ pi/2,  1.0,  6.0],
         inputLBs=[1.0, -1.0],
         inputUBs=[1.0,  1.0],
         N=3;
         lines=[([pi/3, 0.3, 0.0], [0., 0., 4.])])
end
# ------------------------------------------------------------------ #
