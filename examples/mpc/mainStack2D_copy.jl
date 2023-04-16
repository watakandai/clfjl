.using LinearAlgebra
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

    # Dynamics: X = [y, θ], U=[ω]
    velocity = 0.1
    Ad = [1 0;
        velocity 1]
    Bd = [1, 0][:, :]       # [:,:] converts vec to matrix
    # Weights
    Q = spdiagm([0.1, 1])              # Weights for Xs from 0:N-1

    # for iStack in 5:nStack
    iStack = 6

    lines_ = map(l -> (repeat(l[1], iStack), repeat(l[2], iStack)), lines)

    # Constraints
    initLBs_ = repeat(initLBs, iStack)
    initUBs_ = repeat(initUBs, iStack)
    termLBs_ = repeat(termLBs, iStack)
    termUBs_ = repeat(termUBs, iStack)
    boundLBs_ = repeat(boundLBs, iStack)
    boundUBs_ = repeat(boundUBs, iStack)
    inputLBs_ = repeat(inputLBs, iStack)
    inputUBs_ = repeat(inputUBs, iStack)
    initSet = clfjl.HyperRectangle(initLBs_, initUBs_)
    termSet = clfjl.HyperRectangle(termLBs_, termUBs_)
    workspace = clfjl.HyperRectangle(boundLBs_, boundUBs_)
    inputSet = clfjl.HyperRectangle(inputLBs_, inputUBs_)

    params = clfjl.Parameters(
        optDim=N*iStack,
        imgFileDir=joinpath(@__DIR__, "Stack", "output$(N*iStack)D"),
        lfsFileDir=@__DIR__,
        maxIteration=1000,
        maxLyapunovGapForGenerator=10,
        maxLyapunovGapForVerifier=10,
        thresholdLyapunovGapForGenerator=1e-12,
        thresholdLyapunovGapForVerifier=0,
        print=true,
        padding=true
    )

    env = clfjl.Env(numStateDim=N*iStack,
                    numSpaceDim=N*iStack,
                    initSet=initSet,
                    termSet=termSet,
                    workspace=workspace)

    "Setup Gurobi"
    solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
                                            "OutputFlag"=>false))


    "Sample a trajectory for the Dubin's Car model using the MPC contoller"
    function sampleSimpleCar(counterExamples::Vector{clfjl.CounterExample},
                                x0::Vector{<:Real},
                                env::clfjl.Env;
                                xT::Vector{<:Real}=Float64[])

        Ad_ = kron(I(iStack), Ad)
        Bd_ = kron(I(iStack), Bd)
        Q_ = kron(I(iStack), Q)
        QN = Q_                        # Weights for the terminal state X at N (Xn or xT)
        (nx, nu) = size(Bd_)
        R = 1 * speye(nu)
        RD = 1 * speye(nu)

        numHorizon = 10
        function simulateSimpleCar(x0_, xT_, env_, numStep_)
            return clfjl.simulateMPC(x0_, xT_, env_, numStep_,
                            Ad_, Bd_, Q_, R, QN, RD, numHorizon, inputSet,
                            useSet=false)
        end

        return clfjl.sampleCounterExample(counterExamples, x0, env;
                    xT=xT, simulateFunc=simulateSimpleCar, useStabilityAlpha=true)
    end

    # Either choose
    # 1. LV x LC: useProbVerifier=false, checkLyapunovCondition doesn't matter
    # 2. PSV1 x PSC: useProbVerifier=true, checkLyapunovCondition=false
    # 3. PSV2 x PSC: useProbVerifier=true, checkLyapunovCondition=true
    useProbVerifier = true
    checkLyapunovCondition = false

    vfunc(args...) = clfjl.probVerifyCandidateCLF(
        args...; checkLyapunovCondition=checkLyapunovCondition, numSample=1000)

    if useProbVerifier
        @time clfjl.synthesizeCLF(lines_, params, env, solver, sampleSimpleCar;
                                verifyCandidateCLFFunc=vfunc)
    else
        @time clfjl.synthesizeCLF(lines_, params, env, solver, sampleSimpleCar)
    end

end


# ------------------------------ Main ------------------------------ #
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    ## 2D: X=[θ, y] , U=[ω]
    main(initLBs=[-pi/3, -0.3],
         initUBs=[ pi/3,  0.3],
         termLBs=[-pi/12, -0.3],
         termUBs=[ pi/12,  0.3],
         boundLBs=[-pi/2, -1.0],
         boundUBs=[ pi/2,  1.0],
         inputLBs=[-1],
         inputUBs=[ 1],
         N=2;
         lines=[([pi/3,  0.3], [0., 0.])])
end
# ------------------------------------------------------------------ #
