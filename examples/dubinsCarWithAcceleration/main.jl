module ExampleReachAvoid

using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
using Suppressor

# using clfjl
include("../../src/clfjl.jl")

const GUROBI_ENV = Gurobi.Env()

# X = [x, y, v], U = [α, θ]
function main()
    # execPath = "/Users/kandai/Documents/projects/research/clf/build/DubinsCarWithAcceleration"
    execPath = "/home/kandai/Documents/projects/research/ControlLyapunovFunctionPlanners//build/DubinsCarWithAcceleration"
    configPath = joinpath(@__DIR__, "config.yaml")

    config::Dict{Any, Any} = YAML.load(open(configPath))

    N = 3
    x0 = config["start"][1:N]

    # Under the hood, X = [x, y, v, θ], U = [α, ω]
    # In the optimization, we assume X = [x, y, v], U = [α, θ]
    lb = [-0.1, 0.9, 0.0]
    ub = [ 0.1, 1.1, 0.1]
    initSet = clfjl.HyperRectangle(lb, ub)
    @assert all(lb .<= x0) && all(x0 .<= ub)    # x0 ∈ I

    # lb = config["goal"][1:N] .- config["goalThreshold"]     # TODO: fix velocity threshold
    # ub = config["goal"][1:N] .+ config["goalThreshold"]
    lb = config["goalLowerBound"][1:N]
    ub = config["goalUpperBound"][1:N]
    termSet = clfjl.HyperRectangle(lb, ub)
    @assert !(all(lb .<= x0) && all(x0 .<= ub)) # x0 ∉ T

    lb = config["lowerBound"][1:N]
    ub = config["upperBound"][1:N]
    workspace = clfjl.HyperRectangle(lb, ub)
    @assert all(lb .<= x0) && all(x0 .<= ub)    # x0 ∈ I

    # Must translate to [α, ω] -> [α, θ]
    lb = [config["controlLowerBound"][1], config["lowerBound"][N+1]]
    ub = [config["controlUpperBound"][1], config["upperBound"][N+1]]
    inputSet = clfjl.HyperRectangle(lb, ub)

    params = clfjl.Parameters(
        optDim=N,
        imgFileDir=joinpath(@__DIR__, "output"),
        lfsFileDir=@__DIR__,
        maxIteration=1000,
        maxLyapunovGapForGenerator=10,
        maxLyapunovGapForVerifier=10,
        thresholdLyapunovGapForGenerator=1e-12,
        thresholdLyapunovGapForVerifier=0,
        print=true,
        padding=true,
        omplConfig=config,
    )

    pathFilePath = joinpath(pwd(), "path.txt")
    dt = config["propagationStepSize"]

    function getDynamicsf(α, θ, dt_)::clfjl.Dynamics
        A = [1 0 cos(θ)*dt_;
             0 1 sin(θ)*dt_;
             0 0 1]
        b = [0, 0, α*dt_]
        # f(x) = A * x + b
        return clfjl.Dynamics(A, b, N)
    end
    filterStateFunc(x, u) = x[1:N]
    filterInputFunc(x, u) = [u[1], x[N+1]]  # u=[α, ω], x=[x, y, v, θ] => u[1]=α, x[N+1]=θ

    function setOmplConfigFunc(omplConfig_, x0_, xT_)
        omplConfig__ = deepcopy(omplConfig_)
        θ_ = atan(xT_[2] - x0_[2], xT_[1] - x0_[1])
        omplConfig__["start"][1:N] = x0_
        omplConfig__["start"][N+1] = θ_
        omplConfig__["goalLowerBound"][1:N] .+= 0.05
        omplConfig__["goalUpperBound"][1:N] .-= 0.05
        omplConfig__["goalLowerBound"][N+1] = θ_ - 0.1
        omplConfig__["goalUpperBound"][N+1] = θ_ + 0.1
        return omplConfig__
    end

    env = clfjl.Env(numStateDim=config["numStateDim"],
                    numSpaceDim=config["numSpaceDim"],
                    initSet=initSet,
                    termSet=termSet,
                    workspace=workspace,
                    obstacles=[])

    "Setup Gurobi"
    solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
                                            "OutputFlag"=>false))

    numTrial = 10
    function sampleOMPLDubinsCar(iter,
                                 counterExamples::Vector{clfjl.CounterExample},
                                 x0_::Vector{<:Real},
                                 env::clfjl.Env)
        return clfjl.sampleOMPLDubin(iter,
                                     counterExamples, x0_, env,
                                     N, execPath, pathFilePath,
                                     config, inputSet,
                                     getDynamicsf,
                                     filterStateFunc,
                                     filterInputFunc,
                                     setOmplConfigFunc, numTrial)
    end

    "Synthesize Control Lyapunov functions for the given env"
    clfjl.synthesizeCLF(x0, params, env, solver, sampleOMPLDubinsCar, clfjl.plot3DCLF)
end


# ------------------------------ Main ------------------------------ #
# X = [x, y, v, theta] in OMPL but we only use [x, y, v] for synthesizing CLFs
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    main()
end
# main()
# ------------------------------------------------------------------ #


end # module
