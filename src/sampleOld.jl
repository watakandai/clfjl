using DelimitedFiles


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


function sampleTrajectory4D(counterExamples::Vector{CounterExample},
                          samplePoint::Vector{<:Real},
                          params::Parameters,
                          env::Env)

    @assert length(samplePoint) == params.optDim

    # if the witness is in unsafe region, return.
    for o in env.obstacles
        if all(o.lb .≤ samplePoint) && all(samplePoint .≤ o.ub)
            # throw(DomainError(samplePoint, "Trying to sample in an obstacle region"))
            isUnsafe = true
            isTerminal = false
            dynamics = env.hybridSystem.dynamics[1]
            push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
            return
        end
    end

    if !isnothing(params.config["obstacles"])
        for o in params.config["obstacles"]
            if o["type"] == "Convex"
                if all(o["A"] * samplePoint[1:env.numSpaceDim] + o["b"] .<= 0)
                    isUnsafe = true
                    isTerminal = false
                    dynamics = env.hybridSystem.dynamics[1]
                    push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
                end
            end
        end
    end

    tempConfig = deepcopy(params.config)
    outputStr = callOracle(params.execPath, tempConfig)

    if contains(outputStr, "Found a solution")
        if contains(outputStr, "Solution is approximate. Distance to actual goal is")
        end
        println(outputStr)
        filepath = params.pathFilePath
        data = readdlm(filepath)
        dataToCounterExamples4D(counterExamples, data, params, env, params.config)
    else
        dynamics = env.hybridSystem.dynamics[1]
        isTerminal = false
        isUnsafe = true
        push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
    end
end


function sampleOMPLDubin(counterExamples::Vector{CounterExample},
                         x0::Vector{<:Real},
                         params::Parameters,
                         env::Env,
                         N::Integer,
                         execPath::String,
                         pathFilePath::String,
                         omplConfig::Dict{Any, Any},
                         inputSet::HyperRectangle,
                         getDynamicsf::Function)

    @assert length(x0) == N
    println("=============== Initial Set ===============")
    println(x0)
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

    for i in 1:length(X)-1
        u = Float64.(U[i])
        dynamics = getDynamicsf(u...)
        ce = CounterExample(X[i], -1, dynamics, X[i+1], false, isUnsafe)
        push!(counterExamples, ce)
    end
    # u = Float64.(U[1])
    # dynamics = getDynamicsf(u...)
    # ce = CounterExample(X[1], -1, dynamics, X[2], false, isUnsafe)
    # push!(counterExamples, ce)
end


function simulateOMPLDubin(x0::Vector{<:Real},
                           env::Env,
                           N::Integer,
                           execPath::String,
                           pathFilePath::String,
                           omplConfig::Dict{Any, Any},
                           numTrial::Integer=5,
                           )::Tuple{SimStatusCode, StateTraj, InputTraj}

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
            # monoDist = all([norm(X[i][1:2],2) >= norm(X[i+1][1:2],2) for i in 1:numData-1])
            monoDist = norm(X[1][1:2], 2) >= norm(X[2][1:2], 2)
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


function sampleTrajectory2D(counterExamples::Vector{CounterExample},
                            samplePoint::Vector{<:Real},
                            params::Parameters,
                            env::Env)

    @assert length(samplePoint) == params.optDim

    # if the witness is in unsafe region, return.
    for o in env.obstacles
        if all(o.lb .≤ samplePoint) && all(samplePoint .≤ o.ub)
            # throw(DomainError(samplePoint, "Trying to sample in an obstacle region"))
            isUnsafe = true
            isTerminal = false
            dynamics = env.hybridSystem.dynamics[1]
            push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
            return
        end
    end

    for o in params.config["obstacles"]
        if o["type"] == "Convex"
            if all(o["A"] * samplePoint + o["b"] .<= 0)
                isUnsafe = true
                isTerminal = false
                dynamics = env.hybridSystem.dynamics[1]
                push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
            end
        end
    end

    tempConfig::Dict{Any, Any} = deepcopy(params.config)
    tempConfig["start"] = vcat(samplePoint, 0)
    outputStr::String = ""

    if abs(samplePoint[1]) <= samplePoint[2]
        orient = -1.57
    elseif -abs(samplePoint[1]) >= samplePoint[2]
        orient = 1.57
    elseif abs(samplePoint[2]) < samplePoint[1]
        orient = 3.14
    else
        orient = 0
    end
    tempConfig["start"][3] = orient
    outputStr = callOracle(params.execPath, tempConfig)

    if contains(outputStr, "Found a solution")
        # println(outputStr)
        filepath::String = params.pathFilePath
        data = readdlm(filepath)
        dataToCounterExamples(counterExamples, data, params, env, tempConfig)
    else
        dynamics = env.hybridSystem.dynamics[1]
        isTerminal = false
        isUnsafe = true
        push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
    end
end


function sampleTrajectoryManual(counterExamples::Vector{CounterExample},
                          samplePoint::Vector{<:Real},
                          params::Parameters,
                          env::Env)

    @assert length(samplePoint) == params.optDim

    # if the witness is in unsafe region, return.
    for o in env.obstacles
        if all(o.lb .≤ samplePoint) && all(samplePoint .≤ o.ub)
            # throw(DomainError(samplePoint, "Trying to sample in an obstacle region"))
            isUnsafe = true
            isTerminal = false
            dynamics = env.hybridSystem.dynamics[1]
            push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
            return
        end
    end

    for o in params.config["obstacles"]
        if o["type"] == "Convex"
            if all(o["A"] * samplePoint + o["b"] .<= 0)
                isUnsafe = true
                isTerminal = false
                dynamics = env.hybridSystem.dynamics[1]
                push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
            end
        end
    end

    if abs(samplePoint[1]) <= samplePoint[2]
        orient = -1.57
    elseif -abs(samplePoint[1]) >= samplePoint[2]
        orient = 1.57
    elseif abs(samplePoint[2]) < samplePoint[1]
        orient = 3.14
    else
        orient = 0
    end

    data = simulateTrajectory(vcat(samplePoint, orient), env, params)
    dataToCounterExamples(counterExamples, data, params, env, params.config)
end


function dataToCounterExamples(counterExamples, data, params, env, config)

    N = params.optDim

    nData::Integer = size(data, 1)
    X::Matrix{Real} = data[1:end-1, 1:N]
    X′::Matrix{Real} = data[2:end, 1:N]
    Φ::Vector{Real} = data[2:end, 3]

    orientToMode::Dict{Real, Integer} = Dict(0.00 => 1,
                                            -0.00 => 1,
                                             1.57 => 2,
                                             3.14 => 3,
                                             -3.14 => 3,
                                             -1.57 => 4)

    xT = config["goal"][1:N]

    for i in 1:nData-1

        x::Vector{Real} = X[i, :]
        x′::Vector{Real} = X′[i, :]

        if norm(x-xT, 2) < norm(x′-xT, 2)
            println("Norm: ", norm(x-xT, 2), ", ", norm(x′-xT, 2))
            throw(DomainError((x, x′, xT), "Distance is not monotically decreasing"))
        end

        orient::Real = round(Φ[i], digits=2)
        q::Integer = orientToMode[orient]
        dynamics = env.hybridSystem.dynamics[q]

        normOfWitness::Real = norm(x, 1)
        α::Real = normOfWitness * (1 + opnorm(dynamics.A, 1))

        isTerminal::Bool = i == nData-1
        isUnsafe::Bool = false
        push!(counterExamples, CounterExample(x, α, dynamics, x′, isTerminal, isUnsafe))
    end
end


function simulateTrajectory(samplePoint, env, params)
    N = params.optDim
    trajectory = []
    function inTerminalRegion(x)
        return all(env.termSet.lb .<= x[1:N]) && all(x[1:N] .<= env.termSet.ub)
    end

    function getMode(x)
        if abs(x[1]) <= x[2]
            mode = 4
            orient = -1.57
        elseif -abs(x[1]) >= x[2]
            mode = 2
            orient = 1.57
        elseif abs(x[2]) < x[1]
            mode = 3
            orient = 3.14
        else
            mode = 1
            orient = 0.0
        end
        return mode, orient
    end

    x = samplePoint
    push!(trajectory, samplePoint)

    while(!inTerminalRegion(x))
        m, o = getMode(x)
        x′ = env.hybridSystem.dynamics[m].A * x[1:N] + env.hybridSystem.dynamics[m].b
        push!(trajectory, vcat(x′, o))
        x = x′
    end

    if length(trajectory) == 1
        m, o = getMode(x)
        x′ = env.hybridSystem.dynamics[m].A * x[1:N] + env.hybridSystem.dynamics[m].b
        push!(trajectory, vcat(x′, o))
    end

    return reduce(hcat, trajectory)'
end


function dataToCounterExamples4D(counterExamples, data, params, env, config)

    N = 3

    nData::Integer = size(data, 1)
    X::Matrix{Real} = data[1:end-1, 1:N]
    X′::Matrix{Real} = data[2:end, 1:N]
    Q::Vector{Real} = data[2:end, end-1]

    xT = config["goal"][1:N]

    for i in 1:nData-1

        x::Vector{Real} = X[i, :]
        x′::Vector{Real} = X′[i, :]
        q = Int(Q[i])

        # if norm(x-xT, 2) < norm(x′-xT, 2)
        #     println("Norm: ", norm(x-xT, 2), ", ", norm(x′-xT, 2))
        #     throw(DomainError((x, x′, xT), "Distance is not monotically decreasing"))
        # end

        dynamics = env.hybridSystem.dynamics[q]
        normOfWitness::Real = norm([x[1], x[2], cos(x[3]), sin(x[3])], 1)
        α::Real = normOfWitness * (1 + opnorm(dynamics.A, 1))

        isTerminal::Bool = i == nData-1
        isUnsafe::Bool = false
        push!(counterExamples, CounterExample([x[1], x[2], cos(x[3]), sin(x[3])],
                                              α,
                                              dynamics,
                                              [x′[1], x′[2], cos(x′[3]), sin(x′[3])],
                                              isTerminal,
                                              isUnsafe))
    end
end


function sampleTrajectory3D(counterExamples::Vector{CounterExample},
                          samplePoint::Vector{<:Real},
                          params::Parameters,
                          env::Env)

    @assert length(samplePoint) == params.optDim

    # if the witness is in unsafe region, return.
    for o in env.obstacles
        if all(o.lb .≤ samplePoint) && all(samplePoint .≤ o.ub)
            # throw(DomainError(samplePoint, "Trying to sample in an obstacle region"))
            isUnsafe = true
            isTerminal = false
            dynamics = env.hybridSystem.dynamics[1]
            push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
            return
        end
    end

    if !isnothing(params.config["obstacles"])
        for o in params.config["obstacles"]
            if o["type"] == "Convex"
                if all(o["A"] * samplePoint[1:env.numSpaceDim] + o["b"] .<= 0)
                    isUnsafe = true
                    isTerminal = false
                    dynamics = env.hybridSystem.dynamics[1]
                    push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
                end
            end
        end
    end

    tempConfig = deepcopy(params.config)
    outputStr = callOracle(params.execPath, tempConfig)

    if contains(outputStr, "Found a solution")
        if contains(outputStr, "Solution is approximate. Distance to actual goal is")
        end
        println(outputStr)
        filepath = params.pathFilePath
        data = readdlm(filepath)
        dataToCounterExamples3D(counterExamples, data, params, env, params.config)
    else
        dynamics = env.hybridSystem.dynamics[1]
        isTerminal = false
        isUnsafe = true
        push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
    end
end


function dataToCounterExamples3D(counterExamples, data, params, env, config)

    N = params.optDim
    numOrientation = params.config["numOrientation"]

    nData::Integer = size(data, 1)
    X::Matrix{Real} = data[1:end-1, 1:N]
    X′::Matrix{Real} = data[2:end, 1:N]
    U::Vector{Real} = data[2:end, end-1]

    initTheta = data[1, 4]
    thetaToIdx = Dict(i * (-2*pi/numOrientation) => i for i in 1:numOrientation)
    currOrientIdx = 0
    for (theta, i) in thetaToIdx
        if isapprox(theta, initTheta)
            currOrientIdx = i
        end
    end


    for i in 1:nData-1

        x::Vector{Real} = X[i, :]
        x′::Vector{Real} = X′[i, :]
        u = Int(U[i])

        mode = (currOrientIdx, u)
        nextOrientIdx = env.hybridSystem.transitions[mode]
        currOrientIdx = nextOrientIdx

        dynamics = env.hybridSystem.dynamics[mode]
        normOfWitness::Real = norm(x, 1)
        α::Real = normOfWitness * (1 + opnorm(dynamics.A, 1))

        isTerminal::Bool = i == nData-1
        isUnsafe::Bool = false
        push!(counterExamples, CounterExample(x,
                                              α,
                                              dynamics,
                                              x′,
                                              isTerminal,
                                              isUnsafe))
    end
end
