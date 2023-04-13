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
                               workspace=workspace)

    "Setup Gurobi"
    solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
                                            "OutputFlag"=>false))

    "Sample a trajectory for the Dubin's Car model using the MPC contoller"
    function sampleCartPole(counterExamples::Vector{clfjl.CounterExample},
                              x0::Vector{<:Real},
                              env::clfjl.Env;
                              xT::Vector{<:Real}=Float64[])
        dt = 0.1
        # Dynamics: X = [x, ẋ, θ, θ̇], U=[force]
        A = [0 1 0 0;
            0 0 0.716 0;
            0 0 0 1;
            0 0 15.76 0]
        B = [0, 0.9755, 0, 1.46][:,:]       # [:,:] converts vec to matrix
        Ad = A*dt + I
        Bd = B*dt
        # Weights
        (nx, nu) = size(Bd)
        Q = spdiagm([10, 1, 10, 1])         # Weights for Xs from 0:N-1
        QN = Q                              # Weights for the terminal state X at N (Xn or xT)
        R = speye(nu)
        RD = speye(nu)
        numHorizon = 5

        function simulateCartPole(x0_, xT_, env_, numStep_)
            return clfjl.simulateMPC(x0_, xT_, env_, numStep_,
                            Ad, Bd, Q, R, QN, RD, numHorizon, inputSet,
                            useSet=true)
        end

        X, U, status = clfjl.sampleCounterExample(counterExamples, x0, env;
                    xT=xT, simulateFunc=simulateCartPole, useStabilityAlpha=true)

        # println("="^100)
        # println(X, U, status)
        # println("="^100)
        ith = counterExamples[end].ith
        (nx, nu) = size(B)
        pX = plot(reduce(hcat, X)', layout=(nx, 1))
        pU = plot(reduce(hcat, U)', layout=(nu, 1))
        l = @layout [a b]
        plot(pX, pU, layout=l)
        clfjl.savefigure(params.imgFileDir, "trajectory$(ith).png")
    end

    samplefunc(args...; kwargs...) = clfjl.sampleCartPole(args...; kwargs..., imgFileDir=params.imgFileDir)
    clfjl.synthesizeCLF(lines, params, env, solver, samplefunc)
    # clfjl.synthesizeCLF(lines, params, env, solver, sampleCartPole)
end


# ------------------------------ Main ------------------------------ #
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    ## 4D: X = [x, ẋ, θ, θ̇], U=[force]
    main(initLBs=[-0.1, -1.0, -π/6, -pi/12],
         initUBs=[ 0.1,  1.0,  π/6,  pi/12],
         termLBs=[-1.0, -0.1, -π/12, -π/6],
         termUBs=[ 1.0,  0.1,  π/12,  π/6],
         boundLBs=[-2, -5.0, -π/2, -2π ],
         boundUBs=[ 2,  5.0,  π/2,  2π],
         inputLBs=[-25],
         inputUBs=[ 25],
         N=4;
         lines=[([0.1, 1.0, π/6, π/12], [0.0, 0.0, 0.0, 0.0])])
end
# ------------------------------------------------------------------ #
