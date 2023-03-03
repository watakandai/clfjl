using DelimitedFiles


function callOracle(execPath, config)
    optionStrs = Vector{String}[]
    if !isnothing(config["obstacles"])
        optionStrs::Vector{String} = [toObstacleString(o) for o in config["obstacles"]]
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
    optionCmd = Cmd(optionStrs)
    # println(`$execPath $optionCmd`)
    return read(`$execPath $optionCmd`, String)
end


function toObstacleString(o)
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


function sampleTrajectory(counterExamples, samplePoint, config, params, hybridSystem, workspace)

    # if the witness is in unsafe region, return.
    for o in workspace.obstacles
        if o.lb ≤ samplePoint && samplePoint ≤ o.ub
            return
        end
    end

    tempConfig = deepcopy(config)
    tempConfig["start"] = samplePoint
    outputStr = ""
    for orient in [0, 1.57, 3.14, -1.57]
        tempConfig["start"][3] = orient
        outputStr = callOracle(params.execPath, tempConfig)
        if contains(outputStr, "Found a solution")
            break
        end
    end

    if contains(outputStr, "Found a solution")
        filepath = params.pathFilePath
        data = readdlm(filepath)
        nData = size(data, 1)

        X = data[1:end-1, 1:3]
        X′ = data[2:end, 1:3]
        Φ = X′[:, 3]

        orientToMode = Dict(0.00 => 1,
                            1.57 => 2,
                            3.14 => 3,
                            -3.14 => 3,
                            -1.57 => 4)

        for i in 1:nData-1

            x = X[i, :]
            x′ = X′[i, :]

            orient = Φ[i]
            orient = round(Φ[i], digits=2)
            q = orientToMode[orient]
            dynamics = hybridSystem.dynamics[q]

            normOfWitness = norm(x, 1)
            α = normOfWitness * (1 + opnorm(dynamics.A, 1))

            isTerminal = i == nData-1
            isUnsafe = false
            push!(counterExamples, CounterExample(x, α, dynamics, x′, isTerminal, isUnsafe))
        end
    else
        dynamics = hybridSystem.dynamics[1]
        isTerminal = false
        isUnsafe = true
        push!(counterExamples, CounterExample(samplePoint, 0, dynamics, samplePoint, isTerminal, isUnsafe))
        # throw(DomainError("Path Planner could not find any solution", outputStr))
    end
end
