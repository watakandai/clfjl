using DelimitedFiles
using SparseArrays, OSQP

@enum SampleStatus begin
    TRAJ_FOUND = 0
    TRAJ_INFEASIBLE = 1
    TRAJ_UNSAFE = 2
    TRAJ_MAX_ITER_REACHED = 3
end

StateVector = Vector{<:Real}
StateTraj = Vector{StateVector}
InputTraj = Vector{StateVector}

# Utility function
speye(N) = spdiagm(ones(N))


function callOracle(execPath::String, config::Dict{Any, Any})::String
    optionStrs::Vector{String} = Vector{String}[]
    if "obstacles" in keys(config) && !isnothing(config["obstacles"])
        optionStrs = [toObstacleString(o) for o in config["obstacles"]]
        prepend!(optionStrs, ["--obstacles"])
    end

    for (k, v) in config
        if k ∈ ["obstacles", "dynamics"]
            continue
        end

        if isa(v, Vector{<:Number})
            append!(optionStrs, ["--$k"])
            append!(optionStrs, map(e -> string(e), v))
        else
            append!(optionStrs, ["--$k", string(v)])
        end
    end
    optionCmd::Cmd = Cmd(optionStrs)
    # println(`$execPath $optionCmd`)
    return read(`$execPath $optionCmd`, String)
end


function toObstacleString(o::Dict{Any, Any})::String
    if o["type"] == "Circle"
        x = o["x"]
        y = o["y"]
        r = o["r"]
        return "Circle,$x,$y,$r"
    elseif o["type"] == "Square"
        x = o["x"]
        y = o["y"]
        l = o["l"]
        return "Square,$x,$y,$l"
    elseif o["type"] == "Convex"
        retStrings = ["Convex"]
        numConstraint = length(o["b"])
        for iConst in 1:numConstraint
            a = o["A"][iConst, :]
            b = o["b"][iConst]
            vec = vcat(a, b)
            vecString = join(map(v->string(v), vec), ",")
            push!(retStrings, vecString)
        end
        return join(retStrings, ",")
    end
end


"Sample a trajectory for the Dubin's Car model using the OMPL sampling-based planners"
function sampleOMPLDubin(counterExamples::Vector{CounterExample},
                         x0::Vector{<:Real},
                         env::Env,
                         N::Integer,
                         execPath::String,
                         pathFilePath::String,
                         omplConfig::Dict{Any, Any},
                         inputSet::HyperRectangle,
                         getDynamicsf::Function)

    @assert length(x0) == N
    status, X, U = simulateOMPLDubin(x0,
                                     env,
                                     N,
                                     execPath,
                                     pathFilePath,
                                     omplConfig)

    # Add all data points in the data as unsafe counterexamples
    isUnsafe = status != TRAJ_FOUND
    u0 = rand() * (inputSet.ub - inputSet.lb) + inputSet.lb

    if length(X) == 1
        dynamics = getDynamicsf(u0...)
        ce = CounterExample(X[1], -1, dynamics, X[1], false, isUnsafe)
        push!(counterExamples, ce)
        return
    end

    if length(counterExamples) == 0
        for i in 1:length(X)-1
            u = Float64.(U[i+1])
            dynamics = getDynamicsf(u...)
            ce = CounterExample(X[i], -1, dynamics, X[i+1], false, isUnsafe)
            push!(counterExamples, ce)
        end
    else
        u = Float64.(U[2])
        dynamics = getDynamicsf(u...)
        ce = CounterExample(X[1], -1, dynamics, X[2], false, isUnsafe)
        push!(counterExamples, ce)
    end
end


function simulateOMPLDubin(x0::Vector{<:Real},
                           env::Env,
                           N::Integer,
                           execPath::String,
                           pathFilePath::String,
                           omplConfig::Dict{Any, Any},
                           numTrial::Integer=5,
                           )::Tuple{SampleStatus, StateTraj, InputTraj}

    inTerminalSet(x) = all(env.termSet.lb .<= x) && all(x .<= env.termSet.ub)

    outOfBound(x) = any(x .<= env.workspace.lb) && any(env.workspace.ub .<= x)
    # Ideally, we must convert all obstacles to convex obstacles
    inObstacles(x) = any(map(o->all(o.lb .≤ x) && all(x .≤ o.ub), env.obstacles))
    obstacles = "obstacles" in keys(omplConfig) ? omplConfig["obstacles"] : []
    convexObs = filter(d->d["type"]=="Convex", obstacles)
    inUnreachRegion(x) = any(map(o -> all(o["A"]*x+o["b"] .<= 0), convexObs))

    X = [x0]
    U = []
    if outOfBound(x0) || inObstacles(x0) || inUnreachRegion(x0)
        status = TRAJ_UNSAFE
        return status, X, U
    end

    # if x0 is out of bound, return TRAJ_UNSAFE
    Xs = []
    Us = []
    costs = []
    status = TRAJ_INFEASIBLE
    config = deepcopy(omplConfig)
    config["start"] = x0

    xT = (env.termSet.lb + env.termSet.ub) / 2
    xT = xT[1:2]

    for iTraj in 1:numTrial

        X = [x0]
        U = []
        cost = Inf
        outputStr = callOracle(execPath, config)

        if contains(outputStr, "Found a solution") &&
           !contains(outputStr, "Solution is approximate")
           # Check if the last state is
            data = readdlm(pathFilePath)
            numData = size(data, 1)
            X = [data[i, 1:N] for i in 1:numData]
            U = [data[i, N+1:end-1] for i in 1:numData]
            monoDist = all([norm(X[i][1:2]-xT,2) >= norm(X[i+1][1:2]-xT,2) for i in 1:numData-1])
            # monoDist = norm(X[1][1:2]-xT, 2) >= norm(X[2][1:2]-xT, 2)
            if inTerminalSet(X[end]) && monoDist
                if contains(outputStr, "Found solution with cost ")
                    r = r"Found solution with cost (\d+\.\d+)"
                    csts = [parse(Float64, m[1]) for m in eachmatch(r, outputStr)]
                    cost = minimum(csts)
                end
                status = TRAJ_FOUND
            end
            # status == TRAJ_INFEASIBLE
        end

        push!(costs, cost)
        push!(Xs, X)
        push!(Us, U)
    end
    ind = argmin(costs)
    return status, Xs[ind], Us[ind]
end


function simulateMPC(Ad, Bd, Q, R, QN, x0, xr, xmin, xmax, umin, umax;
                     numHorizon=10, numStep=100, stopCondition=nothing)
    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    """
    Minimize
        (xN-xr)^T QN (xN-xr) + ∑{k=0:N-1} (xk-xr)^T Q (xk-xr) + uk^T R uk
    s.t.
        x{k+1} = A xk + B uk
        xmin ≤ xk ≤ xmax
        umin ≤ uk ≤ umax
        x0 = \bar{x} (Initial state)
    Refer to https://osqp.org/docs/examples/mpc.html
    """
    x0 = Float64.(x0)
    N = numHorizon
    (nx, nu) = size(Bd)

    # - quadratic objective
    P = blockdiag(kron(speye(N), Q), QN, kron(speye(N), R))
    # - linear objective
    q = [repeat(-Q * xr, N); -QN * xr; zeros(N*nu)]
    # - linear dynamics
    Ax = kron(speye(N + 1), -speye(nx)) + kron(spdiagm(-1 => ones(N)), Ad)
    Bu = kron([spzeros(1, N); speye(N)], Bd)
    Aeq = [Ax Bu]
    leq = [-x0; zeros(N * nx)]
    ueq = leq
    # - input and state constraints
    Aineq = speye((N + 1) * nx + N * nu)
    lineq = [repeat(xmin, N + 1); repeat(umin, N)]
    uineq = [repeat(xmax, N + 1); repeat(umax, N)]
    # - OSQP constraints
    A, l, u = [Aeq; Aineq], [leq; lineq], [ueq; uineq]

    # Create an OSQP model
    m = OSQP.Model()

    # Setup workspace
    # OSQP.setup!(m; P=P, q=q, A=A, l=l, u=u, warm_start=true, verbose=false)
    OSQP.setup!(m; P=P, q=q, A=A, l=l, u=u, warm_start=true, verbose=true)

    # Simulate in closed loop
    X::Vector{Vector{<:Real}} = [x0]
    U = []
    @time for _ in 1 : numStep
        # Solve
        res = OSQP.solve!(m)

        # Check solver status
        if res.info.status != :Solved
            error("OSQP did not solve the problem!")
        end

        # Apply first control input to the plant
        ctrl = res.x[(N+1)*nx+1:(N+1)*nx+nu]
        push!(U, ctrl)
        xNext = Ad * x0 + Bd * ctrl
        push!(X, xNext)

        if !isnothing(stopCondition) && stopCondition(xNext)
            break
        end

        x0 = xNext
        @assert length(xNext) == length(xr)

        # Update initial state
        l[1:nx], u[1:nx] = -x0, -x0
        OSQP.update!(m; l=l, u=u)
    end
    return X, U
end


function simulateSimpleCar(Ad, Bd, x0::Vector{<:Real},
                           termSet::HyperRectangle,
                           bound::HyperRectangle,
                           inputSet::HyperRectangle;
                           maxIteration::Integer=10,
                           numStep::Integer=100,
                           numHorizon::Integer=20,
                           )::Tuple{SampleStatus, StateTraj, InputTraj}

    xT =  (termSet.lb + termSet.ub) / 2
    (nx, nu) = size(Bd)
    @assert size(Ad) == (nx, nx)
    @assert length(x0) == nx
    @assert length(termSet.lb) == nx
    @assert length(termSet.ub) == nx
    @assert length(bound.lb) == nx
    @assert length(bound.ub) == nx
    @assert length(inputSet.lb) == nu
    @assert length(inputSet.ub) == nu

    # Objective function
    Q = spdiagm(ones(nx))           # Weights for Xs from 0:N-1
    QN = Q                          # Weights for the terminal state X at N (Xn or xT)
    R = 0.1 * speye(nu)

    inTerminalSet(x) = all(termSet.lb .<= x) && all(x .<= termSet.ub)
    outOfBound(x) = any(x .<= bound.lb) && any(bound.ub .<= x)
    stopCondition(x) = inTerminalSet(x) || outOfBound(x)

    status = TRAJ_INFEASIBLE
    X::Vector{Vector{<:Real}} = [x0]
    U::Vector{Vector{<:Real}} = []

    iter = 1
    while status == TRAJ_INFEASIBLE
        try
            X, U = simulateMPC(Ad, Bd, Q, R, QN, x0, xT,
                               bound.lb, bound.ub,
                               inputSet.lb, inputSet.ub;
                               numStep=numStep,
                               stopCondition=stopCondition,
                               numHorizon=numHorizon)
        catch
            status = TRAJ_UNSAFE
            break
        end
        if inTerminalSet(X[end])
            status = TRAJ_FOUND
            break
        elseif outOfBound(X[end])
            status = TRAJ_UNSAFE
            break
        end
        iter += 1
        numStep *= 2        # It's likely that it couldn't reach the terminal set. Increase #steps
        if iter >= maxIteration
            status = TRAJ_MAX_ITER_REACHED
            break
        end
    end
    return status, X, U
end


"Sample a trajectory for the Dubin's Car model using the MPC contoller"
function sampleSimpleCar(counterExamples::Vector{CounterExample},
                         x0::Vector{<:Real},
                         env::Env,
                         Ad::Matrix{<:Real},
                         Bd::Matrix{<:Real},
                         inputSet::HyperRectangle,
                         addAllIfUnsafe::Bool=true)::Nothing
    numDim = length(x0)
    status, X, U = simulateSimpleCar(sparse(Ad),
                                     sparse(Bd),
                                     x0,
                                     env.termSet,
                                     env.workspace,
                                     inputSet)
    # Choices if Unsafe
    # 1. Add all data points in the data as unsafe counterexamples
    # 2. Only add last two to unsafe counterexamples
    isUnsafe = status != TRAJ_FOUND

    if isUnsafe && length(X) == 1
        dynamics = Dynamics(Ad, Bd[:,1], numDim)
        ce = CounterExample(X[1], -1, dynamics, X[1], false, true)
        push!(counterExamples, ce)
    end

    if isUnsafe && !addAllIfUnsafe
        for i in 1:length(X)-1
            b = Bd * U[i]
            dynamics = Dynamics(Ad, b, numDim)
            ce = CounterExample(X[i], -1, dynamics, X[i+1], false, i == length(X)-1)
            push!(counterExamples, ce)
        end
        return
    end

    for i in 1:length(X)-1
        b = Bd * U[i]
        dynamics = Dynamics(Ad, b, numDim)
        ce = CounterExample(X[i], -1, dynamics, X[i+1], false, isUnsafe)
        push!(counterExamples, ce)
    end
end


function simulateOptControl(A, B, b, x0, xT, boundary)::Tuple{SampleStatus, StateTraj, InputTraj}
end


function simulateCloseLoop(A, b, x0;
                           numStep=100, stopCondition=nothing)::Tuple{StateTraj, InputTraj}
    x::Vector{<:Real} = x0
    X::Vector{Vector{<:Real}} = [x0]
    U::Vector{Vector{<:Real}} = []
    for _ in 1:numStep
        xNext = A*x + b
        push!(X, xNext)
        if !isnothing(stopCondition) && stopCondition(xNext)
            break
        end
        x = xNext
    end
    return X, U
end


"Stability Example with Known Control. u=-1/2*I"
function simulateStabilityExample(x0, termSet, bound;
                                  maxIteration::Integer=10,
                                  numStep::Integer=100,
                                  )::Tuple{SampleStatus, StateTraj, InputTraj}
    N = 2
    A = [1 1;
         0 1]
    b = zeros(N)
    K = [-1/2 0;
         0 -1/2]
    A′ = A + K
    inTerminalSet(x) = all(termSet.lb .<= x) && all(x .<= termSet.ub)
    outOfBound(x) = any(x .<= bound.lb) && any(bound.ub .<= x)
    stopCondition(x) = inTerminalSet(x) || outOfBound(x)

    status = TRAJ_INFEASIBLE
    X = nothing
    U = nothing

    iter = 1
    while status == TRAJ_INFEASIBLE
        X, U = simulateCloseLoop(A′, b, x0; numStep=numStep, stopCondition=stopCondition)
        if inTerminalSet(X[end])
            status = TRAJ_FOUND
            break
        elseif outOfBound(X[end])
            status = TRAJ_UNSAFE
            break
        end
        iter += 1
        numStep *= 2        # It's likely that it couldn't reach the terminal set. Increase #steps
        if iter >= maxIteration
            status = TRAJ_MAX_ITER_REACHED
            break
        end
    end
    return status, X, U
end


"Sample a trajectory for the 2D toy example using the hand-crafted controller"
function sampleStabilityExample(counterExamples::Vector{CounterExample},
                                x0::Vector{<:Real},
                                env::Env,
                                A::Matrix{<:Real},
                                b::Vector{<:Real})
    numDim = length(x0)
    dynamics = clfjl.Dynamics(A, b, numDim)

    status, X, U = simulateStabilityExample(x0, env.termSet, env.workspace)
    if status == TRAJ_MAX_ITER_REACHED
        return
    end
    # Choices if Unsafe
    # 1. Add all data points in the data as unsafe counterexamples
    # 2. Only add last two to unsafe counterexamples
    isUnsafe = status != TRAJ_FOUND

    for i in 1:length(X)-1
        ce = CounterExample(X[i], -1, dynamics, X[i+1], false, isUnsafe)
        push!(counterExamples, ce)
    end
end


function StabilityExample(;x=[1., 1.],
                          termLB::Union{Vector{<:Real}, <:Real}=-0.01,
                          termUB::Union{Vector{<:Real}, <:Real}=0.01,
                          boundLB::Union{Vector{<:Real}, <:Real}=-1.1,
                          boundUB::Union{Vector{<:Real}, <:Real}=1.1)::Nothing
    N = 2
    @assert length(x) == N

    termLBs = isa(termLB, Vector{<:Real}) ? termLB : termLB.*ones(N)
    termUBs = isa(termUB, Vector{<:Real}) ? termUB : termUB.*ones(N)
    boundLBs = isa(boundLB, Vector{<:Real}) ? boundLB : boundLB.*ones(N)
    boundUBs = isa(boundUB, Vector{<:Real}) ? boundUB : boundUB.*ones(N)
    termSet = clfjl.HyperRectangle(termLBs, termUBs)
    bound = clfjl.HyperRectangle(boundLBs, boundUBs)

    scatter([0], [0], markershape=:circle)
    numTraj = 100
    for i in 1:numTraj
        x = rand(N) * 2 .- 1
        status, X, U = simulateStabilityExample(x, termSet, bound)
        Xs = [x[1] for x in X]
        Ys = [x[2] for x in X]
        plot!(Xs, Ys, aspect_ratio=:equal)
        # scatter!(Xs, Ys, aspect_ratio=:equal, markershape=:circle)
    end
    savefig(joinpath(@__DIR__, "StabilityExample.png"))
end
# StabilityExample(x=[-1., 1.], boundLB=-1.1, boundUB=1.1)
