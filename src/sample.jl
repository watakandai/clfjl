using DelimitedFiles


function callOracle(execPath::String, config::Dict{Any, Any})::String
    optionStrs::Vector{String} = Vector{String}[]
    if !isnothing(config["obstacles"])
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
    x = o["x"]
    y = o["y"]
    if o["type"] == "Circle"
        r = o["r"]
        return "Circle,$x,$y,$r"
    elseif o["type"] == "Square"
        l = o["l"]
        return "Square,$x,$y,$l"
    end
end


function sampleTrajectory(counterExamples::Vector{CounterExample},
                          samplePoint::Vector{Real},
                          params::Parameters,
                          env::Env)

    N::Integer = env.hybridSystem.numDim
    # if the witness is in unsafe region, return.
    for o in env.obstacles
        if all(o.lb .≤ samplePoint[1:2]) && all(samplePoint[1:2] .≤ o.ub)
            throw(DomainError(x, "Trying to sample in an obstacle region"))
            return
        end
    end

    tempConfig::Dict{Any, Any} = deepcopy(params.config)
    tempConfig["start"] = samplePoint
    outputStr::String = ""

    if abs(samplePoint[1]) < samplePoint[2]
        orient = -1.57
    elseif -abs(samplePoint[1]) > samplePoint[2]
        orient = 1.57
    elseif abs(samplePoint[2]) < samplePoint[1]
        orient = 3.14
    else
        orient = 0
    end
    tempConfig["start"][N] = orient
    outputStr = callOracle(params.execPath, tempConfig)
    # for orient in [0, 1.57, 3.14, -1.57]
    #     tempConfig["start"][N] = orient
    #     outputStr = callOracle(params.execPath, tempConfig)
    #     if contains(outputStr, "Found a solution")
    #         break
    #     end
    # end

    if contains(outputStr, "Found a solution")
        # println(outputStr)
        filepath::String = params.pathFilePath
        data = readdlm(filepath)
        nData::Integer = size(data, 1)

        X::Matrix{Real} = data[1:end-1, 1:N]
        X′::Matrix{Real} = data[2:end, 1:N]
        Φ::Vector{Real} = X′[:, N]

        orientToMode::Dict{Real, Integer} = Dict(0.00 => 1,
                                                 1.57 => 2,
                                                 3.14 => 3,
                                                 -3.14 => 3,
                                                 -1.57 => 4)

        for i in 1:nData-1

            x::Vector{Real} = X[i, :]
            x′::Vector{Real} = X′[i, :]
            # println(x)

            orient::Real = round(Φ[i], digits=2)
            q::Integer = orientToMode[orient]
            dynamics::Dynamics = env.hybridSystem.dynamics[q]

            normOfWitness::Real = norm(x, 1)
            α::Real = normOfWitness * (1 + opnorm(dynamics.A, 1))

            isTerminal::Bool = i == nData-1
            isUnsafe::Bool = false
            push!(counterExamples, CounterExample(x, α, dynamics, x′, isTerminal, isUnsafe))

            # if i == nData-1
            #     println(x′)
            # end
        end
    else
        dynamics = env.hybridSystem.dynamics[1]
        isTerminal = false
        isUnsafe = true
        push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
    end
end
