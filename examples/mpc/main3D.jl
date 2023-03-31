using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
using JLD2
using Suppressor

# using clfjl
include("../../src/clfjl.jl")
const GUROBI_ENV = Gurobi.Env()


function main(;lines::Vector{Tuple{Vector{Float64}, Vector{Float64}}},
               initLB::Union{Vector{<:Real}, <:Real},
               initUB::Union{Vector{<:Real}, <:Real},
               termLB::Union{Vector{<:Real}, <:Real},
               termUB::Union{Vector{<:Real}, <:Real},
               boundLB::Union{Vector{<:Real}, <:Real},
               boundUB::Union{Vector{<:Real}, <:Real},
               inputLB::Union{Vector{<:Real}, <:Real},
               inputUB::Union{Vector{<:Real}, <:Real},
               N::Integer)

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
                             x0_::Vector{<:Real},
                             env_::clfjl.Env;
                             xT::Vector{<:Real}=[])
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
        # X = [θ, y, x], U = [v=1, ω]
        # Ad = [1 0 0;            # θ' = θ + ω
        #       0 1 velocity;     # y' = y + vθ
        #       0 0 1;]           # x' = x + v
        # Bd = [0 1;
        #       0 0;
        #       velocity 0;]
        termSet = clfjl.HyperRectangle(env_.termSet.lb[1:2], env_.termSet.ub[1:2])
        workspace = clfjl.HyperRectangle(env_.workspace.lb[1:2], env_.workspace.ub[1:2])
        inputSet_ = clfjl.HyperRectangle([inputSet.lb[2]], [inputSet.ub[2]])
        env__::clfjl.Env = clfjl.Env(numStateDim=N,
                                    numSpaceDim=N,
                                    initSet=initSet,
                                    termSet=termSet,
                                    workspace=workspace,
                                    obstacles=[])
        if length(xT) != 0
            xT_ = xT[1:2]
        else
            xT_ = xT
        end

        println("x0=$(x0_), xT=$(xT)")
        clfjl.sampleSimpleCar(counterExamples, x0_[1:2], env__, Ad, Bd, inputSet_; xT=xT_)
        recentIth = counterExamples[end].ith
        filtCes = filter(c->c.ith == recentIth, counterExamples)
        x0 = x0_[3]
        # println(filtCes[1].x, ", ", x0_[3])
        for i in 1:length(filtCes)
            filtCes[i].x = vcat(filtCes[i].x, [x0])
            filtCes[i].y = vcat(filtCes[i].y, [x0+velocity])
            A = []
            b = []
            A = [filtCes[i].dynamics.A zeros(2);
                 zeros(2)' 1]               # add x+v to the dynamics
            b = [filtCes[i].dynamics.b; velocity]          # add x+v to the dynamics
            filtCes[i].dynamics = clfjl.Dynamics(A, b, 3)
            # println("x=$(filtCes[i].x), y=$(filtCes[i].y)")
            x0 += velocity
        end
    end

    # clfjl.synthesizeCLF(lines, params, env, solver, sampleSimpleCar, clfjl.plot2DCLF)
    clfjl.synthesizeCLF(lines, params, env, solver, sampleSimpleCar, clfjl.plot3DCLF)
    # clfjl.synthesizeCLF(lines, params, env, solver, sampleSimpleCar)
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
    # main(lines=[([pi/3,  0.3], [0., 0.])],
    #      initLB=[-pi/3, -0.3],
    #      initUB=[ pi/3,  0.3],
    #      termLB=[-pi/12, -0.3],
    #      termUB=[ pi/12,  0.3],
    #      boundLB=[-pi/2, -1.0],
    #      boundUB=[ pi/2,  1.0],
    #      inputLB=[-1],
    #      inputUB=[ 1],
    #      N=2)
    # main(lines=[([-pi/3, -0.3], [0., 0.]),
    #             ([ pi/3, 0.3], [0., 0.])],
    #      initLB=[-pi/3, -0.3],
    #      initUB=[ pi/3,  0.3],
    #      termLB=[-pi/12, -0.3],
    #      termUB=[ pi/12,  0.3],
    #      boundLB=[-pi/2, -3.0],
    #      boundUB=[ pi/2,  3.0],
    #      inputLB=[-1],
    #      inputUB=[ 1],
    #      N=2)
    ## 3D: X=[θ, y, x], U=[v=1, ω]
    main(lines=[([ pi/3,  0.3, 0.0], [0., 0., 0.]),
                ([-pi/3, -0.3, 0.0], [0., 0., 0.])],
         initLB=[-pi/3, -0.3, 0.0],
         initUB=[ pi/3,  0.3, 0.0],
         termLB=[-pi/12, -0.3, 0.0],
         termUB=[ pi/12,  0.3, 20.0],
         boundLB=[-pi/2, -1.0, -20.0],
         boundUB=[ pi/2,  1.0,  20.0],
         inputLB=[1.0, -1],
         inputUB=[1.0,  1],
         N=3)
end
# ------------------------------------------------------------------ #
