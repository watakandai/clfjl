using DelimitedFiles
using SparseArrays, OSQP
using Plots

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
    # println("isUnsafe: ", isUnsafe, ", status: ", status, ", length(X): ", length(X))

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


function callMPCSet(Ad, Bd, Q, R, QN, x0, xTmin_, xTmax_, xmin, xmax, umin, umax;
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
    # println("xTmin = $xTmin, xTmax = $xTmax")

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
    U::Vector{Vector{<:Real}} = []
    As::Vector{Matrix{<:Real}} = []
    bs::Vector{Vector{<:Real}} = []
    @time for iStep in 1 : numStep
        # Solve
        res = OSQP.solve!(m)

        # Check solver status
        if res.info.status != :Solved
            println("Could not Solve!")
            # println(X)
            # println(U)
            # error("OSQP did not solve the problem!")
            return X, U, As, bs
        end

        # println("Solved!")
        # println(res.x)

        # Apply first control input to the plant
        ctrl = res.x[(N+1)*nx+1:(N+1)*nx+nu]

        push!(U, ctrl)
        xNext = Ad * x0 + Bd * ctrl
        push!(As, Ad)
        push!(bs, Bd * ctrl)
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
    return X, U, As, bs
end


function callMPC(Ad, Bd, Q, R, QN, RD, x0, xr, xmin, xmax, umin, umax;
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
    U::Vector{Vector{<:Real}} = []
    As::Vector{Matrix{<:Real}} = []
    bs::Vector{Vector{<:Real}} = []
    @time for _ in 1 : numStep
        # Solve
        res = OSQP.solve!(m)

        # Check solver status
        if res.info.status != :Solved
            println("Could not Solve!")
            # println(X)
            # println(U)
            # error("OSQP did not solve the problem!")
            return X, U, As, bs
        end

        # Apply first control input to the plant
        ctrl = res.x[(N+1)*nx+1:(N+1)*nx+nu]
        push!(U, ctrl)
        xNext = Ad * x0 + Bd * ctrl
        push!(As, Ad)
        push!(bs, Bd * ctrl)
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
    return X, U, As, bs
end


function simulateMPC(x0::Vector{<:Real},
                     xT::Vector{<:Real},
                     env::Env,
                     numStep::Integer,
                     Ad, Bd, Q, R, QN, RD, numHorizon, inputSet;
                     useSet::Bool=false)

    inTerminalSet(x) = all(env.termSet.lb .<= x) && all(x .<= env.termSet.ub)
    outOfBound(x) = any(x .< env.workspace.lb) || any(env.workspace.ub .< x)
    stopCondition(x) = inTerminalSet(x) || outOfBound(x)

    status = TRAJ_INFEASIBLE
    X::Vector{Vector{<:Real}} = [x0]
    U::Vector{Vector{<:Real}} = []
    As::Vector{Matrix{<:Real}} = []
    bs::Vector{Vector{<:Real}} = []

    try
        if useSet
            X, U, As, bs = callMPCSet(Ad, Bd, Q, R, QN, x0,
                                      env.termSet.lb, env.termSet.ub,
                                      env.workspace.lb, env.workspace.ub,
                                      inputSet.lb, inputSet.ub;
                                      numStep=numStep,
                                      stopCondition=stopCondition,
                                      numHorizon=numHorizon)
        else
            X, U, As, bs = callMPC(Ad, Bd, Q, R, QN, RD, x0, xT,
                                   env.workspace.lb, env.workspace.ub,
                                   inputSet.lb, inputSet.ub;
                                   numStep=numStep,
                                   stopCondition=stopCondition,
                                   numHorizon=numHorizon)
        end
    catch e
        println(e)
        status = TRAJ_UNSAFE
    end

    if inTerminalSet(X[end])
        status = TRAJ_FOUND
    elseif outOfBound(X[end])
        status = TRAJ_UNSAFE
    end
    return X, U, As, bs, status
end


"Default function for simulating using the step function"
function simulateWithStepFunc(x0::Vector{<:Real},
                              xT::Vector{<:Real},
                              env::Env,
                              numStep::Integer,
                              stepFunc)

    inTerminalSet(x) = all(env.termSet.lb .<= x) && all(x .<= env.termSet.ub)
    outOfBound(x) = any(x .< env.workspace.lb) || any(env.workspace.ub .< x)

    status = TRAJ_INFEASIBLE
    X::Vector{Vector{<:Real}} = [x0]
    U::Vector{Vector{<:Real}} = []
    As::Vector{Matrix{<:Real}} = []
    bs::Vector{Vector{<:Real}} = []
    x = x0

    for _ in 1:numStep
        X, U, As, bs = stepFunc(x, xT, X, U, As, bs) # stepFunc must contain the dynamics
        xNext = X[end]
        if inTerminalSet(xNext)
            status = TRAJ_FOUND
            break
        elseif outOfBound(xNext)
            status = TRAJ_UNSAFE
            break
        end
        x = xNext
    end
    return X, U, As, bs, status
end


"Default function for sampling trajectory. Either pass simulateFunc or stepFunc"
function sampleTrajectory(x0::Vector{<:Real},
                          xT::Vector{<:Real},
                          env::Env,
                          numStep::Integer,
                          maxIteration::Integer;
                          stepFunc=nothing,
                          simulateFunc=nothing)

    status = TRAJ_INFEASIBLE
    iter = 1

    if isnothing(simulateFunc) && isnothing(stepFunc)
        throw(ArgumentError("Must provide either 'simulateFunc' or 'stepFunc'."))
    end

    if isnothing(simulateFunc)
        simulateFunc = (args...) -> simulateWithStepFunc(args..., stepFunc)
    end
    X::Vector{Vector{<:Real}} = [x0]
    U::Vector{Vector{<:Real}} = []
    As::Vector{Matrix{<:Real}} = []
    bs::Vector{Vector{<:Real}} = []
    while status == TRAJ_INFEASIBLE
        # Run the simulatio n loop
        X, U, As, bs, status = simulateFunc(x0, xT, env, numStep)
        iter += 1
        numStep *= 2        # It's likely that it couldn't reach the terminal set. Increase #steps
        if iter >= maxIteration
            status = TRAJ_MAX_ITER_REACHED
            break
        end
    end
    return X, U, As, bs, status
end


"Sample a counter example by sampling trajectory"
function sampleCounterExample(counterExamples::Vector{CounterExample},
                              x0::Vector{<:Real},
                              env::Env;
                              numStep::Integer=100,
                              maxIteration::Integer=10,
                              useTrajectory::Bool=false,
                              useStabilityAlpha::Bool=false,
                              xT::Vector{<:Real}=Float64[],
                              stepFunc=nothing,
                              simulateFunc=nothing)

    numDim = length(x0)
    xT = length(xT) == 0 ? (env.termSet.lb + env.termSet.ub) / 2 : xT

    X, U, As, bs, status = sampleTrajectory(x0, xT, env, numStep, maxIteration;
                                            stepFunc=stepFunc,
                                            simulateFunc=simulateFunc)
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
        return X, U, status
    else
        if useTrajectory
            for i in 1:length(X)-1
                A = As[i]
                b = bs[i]
                dynamics = Dynamics(A, b, numDim)
                if useStabilityAlpha
                    α = norm(X[i], 2) # This is for stability examples
                else
                    α = norm(X[i+1]-X[i], 2)
                end
                ce = CounterExample(X[i], α, dynamics, X[i+1], false, isUnsafe, ith)
                push!(counterExamples, ce)
            end
        else
            A = As[1]
            b = bs[1]
            dynamics = Dynamics(A, b, numDim)
            if useStabilityAlpha
                α = norm(X[1], 2) # This is for stability examples
            else
                α = norm(X[2]-X[1], 2)
            end
            ce = CounterExample(X[1], α, dynamics, X[2], false, isUnsafe, ith)
            push!(counterExamples, ce)
        end
    end
    return X, U, status
end


"Min Func Example"
function sampleMinFunc(counterExamples::Vector{CounterExample},
                       x0::Vector{<:Real},
                       env::Env;
                       xT::Vector{<:Real}=Float64[])
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

    function stepFunc(x, xT, X, U, As, bs)
        if controlSwitch(x)
            xNext = Ad1*x
            push!(As, Ad1)
        else
            xNext = Ad2*x
            push!(As, Ad2)
        end
        push!(X, xNext)
        push!(bs, zeros(2))
        return X, U, As, bs
    end

    return sampleCounterExample(counterExamples, x0, env;
                                xT=xT, stepFunc=stepFunc)
end


"Switch Stable System Example"
function sampleSwitchStable(counterExamples::Vector{CounterExample},
                            x0::Vector{<:Real},
                            env::Env;
                            xT::Vector{<:Real}=Float64[])

    controlSwitch(x) = x[1]*x[2] ≥ 0 ? true : false
    dt = 0.1
    A1 = [-0.1  1.0;
            -10.0 -0.1]
    A2 = [-0.1  10.0;
            -1.0 -0.1]
    Ad1 = A1*dt + I
    Ad2 = A2*dt + I

    function stepFunc(x, xT, X, U, As, bs)
        if controlSwitch(x)
            xNext = Ad1*x
            push!(As, Ad1)
        else
            xNext = Ad2*x
            push!(As, Ad2)
        end
        push!(X, xNext)
        push!(bs, zeros(2))
        return X, U, As, bs
    end

    return sampleCounterExample(counterExamples, x0, env;
                                xT=xT, stepFunc=stepFunc)
end


"Buck Converter System Example"
function sampleBuckConverter(counterExamples::Vector{CounterExample},
                             x0::Vector{<:Real},
                             env::Env;
                             xT::Vector{<:Real}=Float64[])
    """
    DC-DC Buck Converter Model from
    https://files.cercomp.ufg.br/weby/up/762/o/Slides-Seminario.pdf?1658240138
    """
    dt = 0.1
    A = [0 1;
         -1 -1]
    b = [0, 1][:,:]
    Ad = A*dt + I
    Bd = b*dt
    h(x) = k1*(x[1] − xT[1]) + k2*(x[2] - xT[2])
    controlFunc(x) = h(x) > 0 ? [0] : [1]

    function stepFunc(x, xT, X, U, As, bs)
        u = controlFunc(x)
        xNext = Ad*x + Bd*u
        push!(X, xNext)
        push!(U, u)
        push!(As, Ad)
        push!(bs, Bd*u)
        return X, U, As, bs
    end

    return sampleCounterExample(counterExamples, x0, env;
                                xT=xT, stepFunc=stepFunc)
end


"Stability Example with Known Control. u=-1/2*I"
function sampleStabilityExample(counterExamples::Vector{CounterExample},
                                x0::Vector{<:Real},
                                env::Env;
                                xT::Vector{<:Real}=Float64[])

    N = 2
    A = [1 1;
         0 1]
    b = zeros(N)
    K = [-1/2 0;
         0 -1/2]
    A′ = A + K

    function stepFunc(x, xT, X, U, As, bs)
        xNext = A′*x + b
        push!(X, xNext)
        push!(As, A′)
        push!(bs, b)
    return X, U, As, bs
    end

    return sampleCounterExample(counterExamples, x0, env;
                xT=xT, stepFunc=stepFunc)
end


"Example"
function sampleCartPole(counterExamples::Vector{CounterExample},
                        x0::Vector{<:Real},
                        env::Env;
                        xT::Vector{<:Real}=Float64[],
                        imgFileDir::String)
    """
    https://danielpiedrahita.wordpress.com/portfolio/cart-pole-control/
    """
    dt = 0.1
    # Dynamics: X = [x, ẋ, θ, θ̇], U=[force]
    A = [0 1 0 0;
         0 0 0.716 0;
         0 0 0 1;
         0 0 15.76 0]
    B = [0, 0.9755, 0, 1.46][:,:]       # [:,:] converts vec to matrix
    K = [-3.1626, -4.2691, 38.9192, 9.9633]

    function stepFunc(x, xT, X, U, As, bs)
        u = -transpose(K)*x
        dx = A*x + B*[u]
        xNext = x + dx*dt
        push!(U, [u])
        push!(X, xNext)
        push!(As, A*dt+I)
        push!(bs, B*[u]*dt)
        return X, U, As, bs
    end

    X, U, status = sampleCounterExample(counterExamples, x0, env;
                xT=xT, stepFunc=stepFunc)

    # println("="^100)
    # println(X)
    # println(U)
    # println(status)
    # println("="^100)
    ith = counterExamples[end].ith
    (nx, nu) = size(B)
    pX = plot(reduce(hcat, X)', layout=(nx, 1))
    pU = plot(reduce(hcat, U)', layout=(nu, 1))
    l = @layout [a b]
    plot(pX, pU, layout=l)
    savefigure(imgFileDir, "trajectory$(ith).png")
end


"Example"
function sample3(counterExamples::Vector{CounterExample},
    x0::Vector{<:Real},
    env::Env;
    xT::Vector{<:Real}=Float64[])

    function stepFunc(x, xT, X, U, As, bs)
    return X, U, As, bs
    end

    return sampleCounterExample(counterExamples, x0, env;
                xT=xT, stepFunc=stepFunc)
end


