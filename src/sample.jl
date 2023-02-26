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


function sample(wit_cls, witness, Dpieces, nDpieces, M, execPath, config)

    # TODO: if the witness is in unsafe region, return.
    if !isnothing(config["obstacles"])
        for o in config["obstacles"]
            if o["x"] - o["l"]/2 <= witness[1] &&\
            witness[1] <= o["x"] + o["l"]/2 &&\
            o["y"] - o["l"]/2 <= witness[2] &&\
            witness[2] <= o["y"] + o["l"]/2
            return
            end
        end
    end

    tempConfig = deepcopy(config)
    tempConfig["start"] = witness
    outputStr = callOracle(execPath, tempConfig)

    if contains(outputStr, "Found a solution")
        filepath = joinpath(pwd(), "path.txt")
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
            normOfWitness = norm(x, 1)

            img_cls = [IT_[] for q in 1:M]
            for (k, piece) in enumerate(Dpieces)
                flow = piece.flows[q]
                α = normOfWitness * (1 + nDpieces[k][q])
                push!(img_cls[q], Image(α, x′, flow))
            end
            push!(wit_cls, [Witness(x, img_cls)])
        end
    else
        DomainError("Path Planner could not find any solution")
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
    outputStr = callOracle(params.execPath, tempConfig)

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

            push!(counterExamples, CounterExample(x, α, dynamics, x′))
        end
    else
        DomainError("Path Planner could not find any solution")
    end
end
