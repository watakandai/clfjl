using DelimitedFiles
using SparseArrays, OSQP
using Plots

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


function defaultSetOmplConfigFunc(omplConfig::Dict{Any, Any}, x0::Vector{<:Real}, args...)::Dict{Any, Any}
    omplConfig_ = deepcopy(omplConfig)
    omplConfig_["start"] = x0
    return omplConfig_
end

"Sample a trajectory for the Dubin's Car model using the OMPL sampling-based planners"
function sampleOMPLDubin(iter,
                        counterExamples::Vector{CounterExample},
                         x0::Vector{<:Real},
                         env::Env,
                         N::Integer,
                         execPath::String,
                         pathFilePath::String,
                         omplConfig::Dict{Any, Any},
                         inputSet::HyperRectangle,
                         getDynamicsf::Function,
                         filterStateFunc::Function=(x,u)->x,
                         filterInputFunc::Function=(x,u)->u,
                         setOmplConfigFunc::Function=defaultSetOmplConfigFunc,
                         numTrial::Integer=5)

    @assert length(x0) == N
    status, X, U, Dt = simulateOMPLDubin(x0,
                                         env,
                                         N,
                                         execPath,
                                         pathFilePath,
                                         omplConfig,
                                         setOmplConfigFunc,
                                         numTrial)

    # Add all data points in the data as unsafe counterexamples
    isUnsafe = status != TRAJ_FOUND
    println("isUnsafe: ", isUnsafe, ", status: ", status, ", length(X): ", length(X))

    numDim = 4
    time = 1:length(X)
    labels = ["x", "y", "v", "θ"]
    plot(legend=true)
    for i in 1:numDim
        plot!(time, getindex.(X, i), label=labels[i])
    end
    savefigure("/home/kandai/Documents/projects/research/clfjl/examples/dubinsCarWithAcceleration/output", "$(iter)Trajectory.png")

    if length(X) == 1
        dt = 0.1
        u0 = rand() * (inputSet.ub - inputSet.lb) + inputSet.lb
        dynamics = getDynamicsf(u0..., dt)
        α = 1.0
        x = filterStateFunc(X[1], u0)
        # ce = CounterExample(x, α, dynamics, x, false, isUnsafe, X)
        ce = CounterExample(x, α, dynamics, x, false, isUnsafe)
        push!(counterExamples, ce)
        return
    end
    # if length(counterExamples) == 0 # We are doing for loop rn, let's do ce=1 next.
        # for i in 1:length(X)-1
        #     dt = Dt[i+1]
        #     u = Float64.(U[i+1])
        #     x = filterStateFunc(X[i], u)
        #     x′ = filterStateFunc(X[i+1], u)
        #     uarg = filterInputFunc(X[i], u)
        #     dynamics = getDynamicsf(uarg..., dt)
        #     α = norm(x′-x, 2)
        #     ce = CounterExample(x, α, dynamics, x′, false, isUnsafe)
        #     push!(counterExamples, ce)
        # end
    # else
        dt = Dt[2]
        u = Float64.(U[2])
        x = filterStateFunc(X[1], u)
        x′ = filterStateFunc(X[2], u)
        uarg = filterInputFunc(X[1], u)
        dynamics = getDynamicsf(uarg..., dt)
        α = norm(x′-x, 2)
        # ce = CounterExample(x, α, dynamics, x′, false, isUnsafe, X)
        ce = CounterExample(x, α, dynamics, x′, false, isUnsafe)
        push!(counterExamples, ce)
    # end
end


function simulateOMPLDubin(x0::Vector{<:Real},
                           env::Env,
                           N::Integer,
                           execPath::String,
                           pathFilePath::String,
                           omplConfig::Dict{Any, Any},
                           setOmplConfigFunc::Function=defaultSetOmplConfigFunc,
                           numTrial::Integer=5,
                           )

    inTerminalSet(x) = all(env.termSet.lb .<= x) && all(x .<= env.termSet.ub)
    outOfBound(x) = any(x .< env.workspace.lb) || any(env.workspace.ub .< x)
    # Ideally, we must convert all obstacles to convex obstacles
    inObstacles(x) = any(map(o->all(o.lb .≤ x) && all(x .≤ o.ub), env.obstacles))
    obstacles = "obstacles" in keys(omplConfig) ? omplConfig["obstacles"] : []
    convexObs = filter(d->d["type"]=="Convex", obstacles)
    inUnreachRegion(x) = any(map(o -> all(o["A"]*x+o["b"] .<= 0), convexObs))

    simN = omplConfig["numStateDim"]

    xT = (env.termSet.lb + env.termSet.ub) / 2
    config = setOmplConfigFunc(omplConfig, x0, xT)
    xT = xT[1:2]

    if outOfBound(x0) || inObstacles(x0) || inUnreachRegion(x0)
        status = TRAJ_UNSAFE
        return status, [config["start"]], [], []
    end

    Xs = []
    Us = []
    Dts = []
    costs = []
    status = TRAJ_INFEASIBLE

    for iTraj in 1:numTrial

        X = [config["start"]]
        U = []
        dt = []
        cost = Inf
        outputStr = callOracle(execPath, config)
        println(outputStr)

        if contains(outputStr, "Found a solution") &&
           !contains(outputStr, "Solution is approximate")
           # Check if the last state is
            data = readdlm(pathFilePath)
            numData = size(data, 1)
            X = [data[i, 1:simN] for i in 1:numData]
            U = [data[i, simN+1:end-1] for i in 1:numData]
            dt = [data[i, end] for i in 1:numData]
            # monoDist = all([norm(X[i][1:2]-xT,2) >= norm(X[i+1][1:2]-xT,2) for i in 1:numData-1])
            monoDist = true
            println("Terminal condition: $(inTerminalSet(X[end][1:N])), at $(X[end]), monoDist=$(monoDist)")
            if inTerminalSet(X[end][1:N]) && monoDist
                if contains(outputStr, "Found solution with cost ")
                    r = r"Found solution with cost (\d+\.\d+)"
                    csts = [parse(Float64, m[1]) for m in eachmatch(r, outputStr)]
                    cost = minimum(csts)
                end
                status = TRAJ_FOUND
                return status, X, U, dt
            end
            # status == TRAJ_INFEASIBLE
        end

        # println("Traj:$iTraj, statu=$status, cost=$cost")
        push!(costs, cost)
        push!(Xs, X)
        push!(Us, U)
        push!(Dts, dt)
    end
    ind = argmin(costs)
    return status, Xs[ind], Us[ind], Dts[ind]
end


function simulateMPCSet(Ad, Bd, Q, R, QN, x0, xTmin_, xTmax_, xmin, xmax, umin, umax;
                     numHorizon=10, numStep=100, stopCondition=nothing)
    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    """
    Minimize
        ∑{k=0:N-1} (xk-xT)^T Q (xk-xT) + (xN-xT)^T QN (xN-xT) + ∑{k=0:N-1} uk^T R uk
    s.t.
        x{k+1} = A xk + B uk
        xmin ≤ xk ≤ xmax
        umin ≤ uk ≤ umax
        x0 = \bar{x} (Initial state)

    We let xT be in the decision variable, so that we can change the terminal condition,
    and replace the constraint xk-xT with xl for all x in the objective function.

    Let X = x = (x(0),x(1),...,x(N), u(0),...,u(N-1), xl(0),xl(1),...,xl(N), xT(0), xT(1),...,xT(N))
    size = (N + 1)*nx  + N*nu + (N + 1)*nx + (N + 1)*nx

    Minimize
        ∑{l=0:N-1} xl^T Q xl + xM^T QN xM + ∑{k=0:N-1} uk^T R uk
    s.t.
        x{k+1} = A xk + B uk
        xmin ≤ xk ≤ xmax
        umin ≤ uk ≤ umax
        x0 = \bar{x} (Initial state)
        xl = xk - xT        => xk -xl - xT = 0 ∀l = 0,1,...,N
        xTmin ≤ xT ≤ xTmax


    Refer to https://osqp.org/docs/examples/mpc.html
    """
    x0 = Float64.(x0)
    N = numHorizon
    (nx, nu) = size(Bd)

    # diff = xTmax_ .- xTmin_
    # dt = 0.25 * diff
    # xTmin = xTmin_ + dt
    # xTmax = xTmax_ - dt
    xTmin = xTmin_
    xTmax = xTmax_
    println("xTmin = $xTmin, xTmax = $xTmax")

    # - quadratic objective
    P = blockdiag(kron(speye(N+1), spzeros(nx,nx)), kron(speye(N), R), kron(speye(N), Q), QN, kron(speye(N+1), spzeros(nx,nx)))
    # - linear objective
    # q = [repeat(-Q * xr, N); -QN * xr; zeros(N*nu)]
    q = [zeros((N+1)*nx); zeros(N*nu); zeros(2*(N+1)*nx)]

    # - linear dynamics
    Ax = kron(speye(N + 1), -speye(nx)) + kron(spdiagm(-1 => ones(N)), Ad)
    Bu = kron([spzeros(1, N); speye(N)], Bd) # last speye(1) is for xT
    Cd = spzeros((N+1)*nx, 2*(N+1)*nx) # last spzeros is for xT
    Aeq = [Ax Bu Cd]

    # - linear terminal constraint x(k) - xl(k) - xT(k) = 0 ∀k=(0,1,...,N)
    At = kron(speye(N+1), speye(nx))        # for +x
    Bt = spzeros(nx*(N+1), N*nu)            # for u=0
    Ct1 = kron(speye(N+1), -speye(nx))      # for -xl
    Ct2 = kron(speye(N+1), -speye(nx))      # for -xT
    Aeq2 = [At Bt Ct1 Ct2]

    leq = [-x0; zeros(N * nx)]
    ueq = leq

    leq2 = zeros((N+1)*nx)
    ueq2 = leq2
    # - input and state constraints
    Aineq = [blockdiag(kron(speye(N+1), speye(nx)), kron(speye(N), speye(nu))) spzeros((N+1)*nx + N*nu, 2*(N+1)*nx);
             spzeros((N+1)*nx, (N+1)*nx + N*nu + (N+1)*nx) kron(speye(N+1), speye(nx))]
    # Aineq = speye((N + 1) * nx + N * nu + nx) # last +nx is for xT
    lineq = [repeat(xmin, N + 1); repeat(umin, N); repeat(xTmin, N + 1)] # last xTmin is for xT
    uineq = [repeat(xmax, N + 1); repeat(umax, N); repeat(xTmax, N + 1)] # last xTmax is for xT
    # - OSQP constraints
    A, l, u = [Aeq; Aeq2; Aineq], [leq; leq2; lineq], [ueq; ueq2; uineq]

    # Create an OSQP model
    m = OSQP.Model()

    # println("+"^100)
    # Setup workspace
    OSQP.setup!(m; P=P, q=q, A=A, l=l, u=u, warm_start=true, verbose=false)
    # OSQP.setup!(m; P=P, q=q, A=A, l=l, u=u, warm_start=true, verbose=true)

    # Simulate in closed loop
    X::Vector{Vector{<:Real}} = [x0]
    U = []
    @time for iStep in 1 : numStep
        # Solve
        res = OSQP.solve!(m)

        # Check solver status
        if res.info.status != :Solved
            println("Could not Solve!")
            println(X, U)
            error("OSQP did not solve the problem!")
        end

        # println("Solved!")
        # println(res.x)

        # Apply first control input to the plant
        ctrl = res.x[(N+1)*nx+1:(N+1)*nx+nu]

        push!(U, ctrl)
        xNext = Ad * x0 + Bd * ctrl
        # println(ctrl, ", ", xNext, ", ", stopCondition(xNext), ", ", iStep)

        push!(X, xNext)

        if !isnothing(stopCondition) && stopCondition(xNext)
            break
        end

        x0 = xNext
        # @assert length(xNext) == length(xr)

        # Update initial state
        l[1:nx], u[1:nx] = -x0, -x0
        OSQP.update!(m; l=l, u=u)
    end
    return X, U
end


function simulateMPC(Ad, Bd, Q, R, QN, RD, x0, xr, xmin, xmax, umin, umax;
                     numHorizon=10, numStep=100, stopCondition=nothing)
    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    """
    Minimize
        ∑{k=0:N-1} (xk-xr)^T Q (xk-xr) + (xN-xr)^T QN (xN-xr) + ∑{k=0:N-1} uk^T R uk
    s.t.
        x{k+1} = A xk + B uk
        xmin ≤ xk ≤ xmax
        umin ≤ uk ≤ umax
        x0 = \bar{x} (Initial state)
    Refer to https://osqp.org/docs/examples/mpc.html
    """

    println("Solving MPC at x0=$x0, xr=$xr")
    x0 = Float64.(x0)
    N = numHorizon
    (nx, nu) = size(Bd)
    EYE2 = 2*speye(N)
    EYE2[1, 1] = 1.0
    EYE2[end, end] = 1.0
    OFFD = spdiagm(-1 => ones(N - 1), 1 => ones(N - 1))
    # display(kron(speye(N), R) + kron(EYE2, RD) - kron(OFFD, RD))

    # - quadratic objective
    P = blockdiag(
        kron(speye(N), Q),
        QN,
        kron(speye(N), R) + kron(EYE2, RD) - kron(OFFD, RD)
        )
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
    OSQP.setup!(m; P=P, q=q, A=A, l=l, u=u, warm_start=true, verbose=false)
    # OSQP.setup!(m; P=P, q=q, A=A, l=l, u=u, warm_start=true, verbose=true)

    # Simulate in closed loop
    X::Vector{Vector{<:Real}} = [x0]
    U = []
    @time for _ in 1 : numStep
        # Solve
        res = OSQP.solve!(m)

        # Check solver status
        if res.info.status != :Solved
            println("OSQP did not solve the problem!")
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
                           numHorizon::Integer=10,
                           xT::Vector{<:Real}=[]
                           )::Tuple{SampleStatus, StateTraj, InputTraj}

    (nx, nu) = size(Bd)
    @assert size(Ad) == (nx, nx)
    @assert length(x0) == nx
    @assert length(termSet.lb) == nx
    @assert length(termSet.ub) == nx
    @assert length(bound.lb) == nx
    @assert length(bound.ub) == nx
    # @assert length(inputSet.lb) == nu
    # @assert length(inputSet.ub) == nu

    # # Objective function
    if length(x0) == 2
        # Q = spdiagm([0.1, 1])              # Weights for Xs from 0:N-1
        # QN = Q                        # Weights for the terminal state X at N (Xn or xT)
        # R = 100 * speye(nu)
        Q = spdiagm([0.1, 1])              # Weights for Xs from 0:N-1
        QN = Q                        # Weights for the terminal state X at N (Xn or xT)
        R = 1 * speye(nu)
        RD = 1 * speye(nu)
    else
        Q = spdiagm([10, 1, 1])           # Weights for Xs from 0:N-1
        QN = 1 * Q                          # Weights for the terminal state X at N (Xn or xT)
        R = 1 * speye(nu)
        RD = 1 * speye(nu)
    end
    # println(Q, QN, R)
    if length(xT) == 0
        xT =  (termSet.lb + termSet.ub) / 2
    end

    inTerminalSet(x) = all(termSet.lb .<= x) && all(x .<= termSet.ub)
    outOfBound(x) = any(x .< bound.lb) || any(bound.ub .< x)
    stopCondition(x) = inTerminalSet(x) || outOfBound(x)

    status = TRAJ_INFEASIBLE
    X::Vector{Vector{<:Real}} = [x0]
    U::Vector{Vector{<:Real}} = []

    iter = 1
    while status == TRAJ_INFEASIBLE
        try
            X, U = simulateMPC(Ad, Bd, Q, R, QN, RD, x0, xT, # termSet.lb, termSet.ub
                               bound.lb, bound.ub,
                               inputSet.lb, inputSet.ub;
                               numStep=numStep,
                               stopCondition=stopCondition,
                               numHorizon=numHorizon)
            # X, U = simulateMPCSet(Ad, Bd, Q, R, QN, x0, termSet.lb, termSet.ub,
            #                     bound.lb, bound.ub,
            #                     inputSet.lb, inputSet.ub;
            #                     numStep=numStep,
            #                     stopCondition=stopCondition,
            #                     numHorizon=numHorizon)
        catch e
            println(e)
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
                         inputSet::HyperRectangle;
                         xT::Vector{<:Real}=[])
    numDim = length(x0)
    status, X, U = simulateSimpleCar(sparse(Ad),
                                     sparse(Bd),
                                     x0,
                                     env.termSet,
                                     env.workspace,
                                     inputSet;
                                     xT=xT)
    # Choices if Unsafe
    # 1. Add all data points in the data as unsafe counterexamples
    # 2. Only add last two to unsafe counterexamples
    isUnsafe = status != TRAJ_FOUND

    if length(counterExamples) == 0
        ith = 1
    else
        ith = counterExamples[end].ith + 1
    end

    if length(X) == 1
        dynamics = Dynamics(Ad, Bd[:,1], numDim)
        α = 1
        ce = CounterExample(X[1], α, dynamics, X[1], false, isUnsafe, ith)
        push!(counterExamples, ce)
        return
    end

    # if length(counterExamples) == 0
        # for i in 1:length(X)-1
        #     b = Bd * U[i]
        #     dynamics = Dynamics(Ad, b, numDim)
        #     α = norm(X[i+1]-X[i], 2)
        #     ce = CounterExample(X[i], α, dynamics, X[i+1], false, isUnsafe, ith)
        #     push!(counterExamples, ce)
        # end

    # end
    # else
        b = Bd * U[1]
        dynamics = Dynamics(Ad, b, numDim)
        # α = norm(X[2]-X[1], 2)
        α = norm(X[1], 2)
        ce = CounterExample(X[1], α, dynamics, X[2], false, isUnsafe, ith)
        push!(counterExamples, ce)
    # end
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
    outOfBound(x) = any(x .< bound.lb) || any(bound.ub .< x)
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



function runCloseLoop(A, b, x0;
    numStep=100, stopCondition=nothing)::Tuple{StateTraj, InputTraj}
end


function simulateBuckConverter(x0, termSet, bound;
                               maxIteration::Integer=10,
                               numStep::Integer=100,
                            #    k1::Float64=2.42591,
                               k1::Float64=1.5,
                               k2::Float64=1.,
                               xT::Vector{Float64}=[0., 0.])
    """
    DC-DC Buck Converter Model from
    https://files.cercomp.ufg.br/weby/up/762/o/Slides-Seminario.pdf?1658240138
    """
    inTerminalSet(x) = all(termSet.lb .<= x) && all(x .<= termSet.ub)
    outOfBound(x) = any(x .< bound.lb) || any(bound.ub .< x)
    stopCondition(x) = inTerminalSet(x) || outOfBound(x)

    dt = 0.1
    A = [0 1;
         -1 -1]
    b = [0, 1][:,:]
    Ad = A*dt + I
    Bd = b*dt

    h(x) = k1*(x[1] − xT[1]) + k2*(x[2] - xT[2])
    controlFunc(x) = h(x) > 0 ? [0] : [1]

    status = TRAJ_INFEASIBLE
    X::Vector{Vector{<:Real}} = [x0]
    As::Vector{Matrix{<:Real}} = []
    bs::Vector{Vector{<:Real}} = []

    iter = 1
    while status == TRAJ_INFEASIBLE
        # Initialize all
        x = x0
        X = [x0]
        As = []
        bs = []
        # Run the simulatio n loop
        for _ in 1:numStep
            xNext = Ad*x + Bd*u
            push!(X, xNext)
            push!(As, Ad)
            push!(bs, Bd*u)
            if inTerminalSet(X[end])
                status = TRAJ_FOUND
                break
            elseif outOfBound(X[end])
                status = TRAJ_UNSAFE
                break
            end
            x = xNext
        end

        iter += 1
        numStep *= 2        # It's likely that it couldn't reach the terminal set. Increase #steps
        if iter >= maxIteration
            status = TRAJ_MAX_ITER_REACHED
            break
        end
    end
    return status, X, As, bs
end


function simulateSwitchStable(x0, termSet, bound;
                              maxIteration::Integer=10,
                              numStep::Integer=100,
                              xT=nothing)
    """
    DC-DC Buck Converter Model from
    https://lucris.lub.lu.se/ws/portalfiles/portal/4673551/8571461.pdf
    """
    inTerminalSet(x) = all(termSet.lb .<= x) && all(x .<= termSet.ub)
    outOfBound(x) = any(x .< bound.lb) || any(bound.ub .< x)
    stopCondition(x) = inTerminalSet(x) || outOfBound(x)

    controlSwitch(x) = x[1]*x[2] ≥ 0 ? true : false
    dt = 0.1
    A1 = [-0.1  1.0;
          -10.0 -0.1]
    A2 = [-0.1  10.0;
           -1.0 -0.1]
    Ad1 = A1*dt + I
    Ad2 = A2*dt + I

    status = TRAJ_INFEASIBLE
    X::Vector{Vector{<:Real}} = [x0]
    As::Vector{Matrix{<:Real}} = []
    bs::Vector{Vector{<:Real}} = []

    iter = 1
    while status == TRAJ_INFEASIBLE
        # Initialize all
        x = x0
        X = [x0]
        As = []
        bs = []
        # Run the simulatio n loop
        for _ in 1:numStep
            if controlSwitch(x)
                xNext = Ad1*x
                push!(As, Ad1)
            else
                xNext = Ad2*x
                push!(As, Ad2)
            end
            push!(X, xNext)
            push!(bs, zeros(2))
            if inTerminalSet(X[end])
                status = TRAJ_FOUND
                break
            elseif outOfBound(X[end])
                status = TRAJ_UNSAFE
                break
            end
            x = xNext
        end

        iter += 1
        numStep *= 2        # It's likely that it couldn't reach the terminal set. Increase #steps
        if iter >= maxIteration
            status = TRAJ_MAX_ITER_REACHED
            break
        end
    end
    return status, X, As, bs
end


function simulateMinFunc(x0, termSet, bound;
                         maxIteration::Integer=10,
                         numStep::Integer=100,
                         xT=nothing)
    """
    DC-DC Buck Converter Model from
    https://lucris.lub.lu.se/ws/portalfiles/portal/4673551/8571461.pdf
    """
    inTerminalSet(x) = all(termSet.lb .<= x) && all(x .<= termSet.ub)
    outOfBound(x) = any(x .< bound.lb) || any(bound.ub .< x)
    stopCondition(x) = inTerminalSet(x) || outOfBound(x)

    dt = 0.1
    A1 = [-5 -4; -1 -2]
    B = [-3, -21][:,:]
    k = [1., 0.]
    k1 = [3., 2.]
    k2 = k - k1
    A = A1 - B*transpose(k1)
    A2 = A + B*transpose(k2)
    Ad1 = A1 *dt + I
    Ad2 = A2 *dt + I
    controlSwitch(x) = dot(k, x) <= 0 ? true : false

    status = TRAJ_INFEASIBLE
    X::Vector{Vector{<:Real}} = [x0]
    As::Vector{Matrix{<:Real}} = []
    bs::Vector{Vector{<:Real}} = []

    iter = 1
    while status == TRAJ_INFEASIBLE
        # Initialize all
        x = x0
        X = [x0]
        As = []
        bs = []
        # Run the simulatio n loop
        for _ in 1:numStep
            if controlSwitch(x)
                xNext = Ad1*x
                push!(As, Ad1)
            else
                xNext = Ad2*x
                push!(As, Ad2)
            end
            push!(X, xNext)
            push!(bs, zeros(2))
            if inTerminalSet(X[end])
                status = TRAJ_FOUND
                break
            elseif outOfBound(X[end])
                status = TRAJ_UNSAFE
                break
            end
            x = xNext
        end

        iter += 1
        numStep *= 2        # It's likely that it couldn't reach the terminal set. Increase #steps
        if iter >= maxIteration
            status = TRAJ_MAX_ITER_REACHED
            break
        end
    end
    return status, X, As, bs
end

"Sample a trajectory for the 2D piecewise linear system (Filippov systems)"
function samplePiecewiseLinearSystem(counterExamples::Vector{CounterExample},
                                     x0::Vector{<:Real},
                                     env::Env,
                                     simulateFunc::Function;
                                     useTrajectory::Bool=false,
                                     xT=Vector{Float64}l())
    numDim = length(x0)
    xT = length(xT) == 0 ? zeros(numDim) : xT
    status, X, As, bs = simulateFunc(x0, env.termSet, env.workspace; xT=xT)
    isUnsafe = status != TRAJ_FOUND

    if length(counterExamples) == 0
        ith = 1
    else
        ith = counterExamples[end].ith + 1
    end

    if length(X) == 1
        dynamics = Dynamics(rand(numDim, numDim), rand(numDim), numDim)
        α = 1
        ce = CounterExample(X[1], α, dynamics, X[1], false, isUnsafe, ith)
        push!(counterExamples, ce)
        return
    else
        if useTrajectory
            for i in 1:length(X)-1
                A = As[i]
                b = bs[i]
                dynamics = Dynamics(A, b, numDim)
                ce = CounterExample(X[i], -1, dynamics, X[i+1], false, isUnsafe, ith)
                push!(counterExamples, ce)
            end
        else
            A = As[1]
            b = bs[1]
            dynamics = Dynamics(A, b, numDim)
            α = norm(X[2]-X[1], 2)
            ce = CounterExample(X[1], α, dynamics, X[2], false, isUnsafe, ith)
            push!(counterExamples, ce)
        end
    end

end
