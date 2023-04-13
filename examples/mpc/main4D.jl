using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
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
        maxIteration=500,
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
                               workspace=workspace)

    "Setup Gurobi"
    solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
                                            "OutputFlag"=>false))

    "Sample a trajectory for the Dubin's Car model using the MPC contoller"
    function sampleSimpleCar4D(counterExamples::Vector{clfjl.CounterExample},
        x0::Vector{<:Real},
        env::clfjl.Env;
        xT::Vector{<:Real}=Float64[])

        # Dynamics: X = [y, θ], U=[ω]
        velocity = 0.1
        Ad = [1 0 0 0;
              velocity 1 0 0;
              0 0 1 0;
              0 0 velocity 1]
        Bd = [1 0;
              0 0;
              0 1;
              0 0]
        # Weights
        (nx, nu) = size(Bd)
        Q = spdiagm([0.1, 1, 0.1, 1])              # Weights for Xs from 0:N-1
        QN = Q                        # Weights for the terminal state X at N (Xn or xT)
        R = 1 * speye(nu)
        RD = 1 * speye(nu)
        numHorizon = 10

        function simulateSimpleCar4D(x0_, xT_, env_, numStep_)
            return clfjl.simulateMPC(x0_, xT_, env_, numStep_,
                            Ad, Bd, Q, R, QN, RD, numHorizon, inputSet,
                            useSet=false)
        end

        return clfjl.sampleCounterExample(counterExamples, x0, env;
                    xT=xT, simulateFunc=simulateSimpleCar4D, useStabilityAlpha=true, maxIteration=3)
    end

    # Either choose
    # 1. LV x LC: useProbVerifier=false, checkLyapunovCondition doesn't matter
    # 2. PSV1 x PSC: useProbVerifier=true, checkLyapunovCondition=false
    # 3. PSV2 x PSC: useProbVerifier=true, checkLyapunovCondition=true
    useProbVerifier = false
    checkLyapunovCondition = true

    vfunc(args...) = clfjl.probVerifyCandidateCLF(
        args...; checkLyapunovCondition=checkLyapunovCondition)

    if useProbVerifier
        @time clfjl.synthesizeCLF(lines, params, env, solver, sampleSimpleCar4D;
                                  verifyCandidateCLFFunc=vfunc)
    else
        @time clfjl.synthesizeCLF(lines, params, env, solver, sampleSimpleCar4D)
    end

end


# ------------------------------ Main ------------------------------ #
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    ## 2D: X=[θ, y] , U=[ω]
    main(initLBs=[-pi/3, -0.3, -pi/3, -0.3],
         initUBs=[ pi/3,  0.3,  pi/3,  0.3],
         termLBs=[-pi/12, -0.3, -pi/12, -0.3],
         termUBs=[ pi/12,  0.3,  pi/12,  0.3],
         boundLBs=[-pi/2, -1.0, -pi/2, -1.0],
         boundUBs=[ pi/2,  1.0,  pi/2,  1.0],
         inputLBs=[-1, -1],
         inputUBs=[ 1,  1],
         N=4;
         lines=[([pi/3,  0.3, pi/3,  0.3], [0., 0., 0., 0.])])
end
# ------------------------------------------------------------------ #
