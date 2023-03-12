import YAML
using DelimitedFiles
using Plots
using Deldir

rotgrid(grd, θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)] * grd


function circleShape(x, y, radius)
    θ = LinRange(0, 2*π, 500)
    x .+ radius*sin.(θ), y .+ radius*cos.(θ)
end


function rectangle(xmin, ymin, w, h, ϕ=0)
    vecs = [0 w w 0; 0 0 h h]
    rot_vecs = rotgrid(vecs, ϕ)
    Shape(xmin .+ rot_vecs[1, :], ymin .+ rot_vecs[2, :])
end


function rectangleCenter(x, y, w, h, ϕ=0)
    vecs = [-w/2 w/2 w/2 -w/2; -h/2 -h/2 h/2 h/2]
    rot_vecs = rotgrid(vecs, ϕ)
    Shape(x .+ rot_vecs[1, :], y .+ rot_vecs[2, :])
end


function drawGeometry(geometry; kwargs...)
    plot!(geometry, seriestype=[:shape]; kwargs...)
end


function plot_env(env)
    # Plot the Boundary
    xmin = Float64(env.workspace.lb[1])
    xmax = Float64(env.workspace.ub[1])
    ymin = Float64(env.workspace.lb[2])
    ymax = Float64(env.workspace.ub[2])

    X = [xmin, xmin, xmax, xmax, xmin]
    Y = [ymin, ymax, ymax, ymin, ymin]
    plot(X, Y, c=:black, lw=2, aspect_ratio=:equal, legend=false)

    xmin = Float64(env.initSet.lb[1])
    xmax = Float64(env.initSet.ub[1])
    ymin = Float64(env.initSet.lb[2])
    ymax = Float64(env.initSet.ub[2])
    X = [xmin, xmin, xmax, xmax, xmin]
    Y = [ymin, ymax, ymax, ymin, ymin]
    plot!(X, Y, c=:gray, lw=2, aspect_ratio=:equal, legend=false)

    xmin = Float64(env.termSet.lb[1])
    xmax = Float64(env.termSet.ub[1])
    ymin = Float64(env.termSet.lb[2])
    ymax = Float64(env.termSet.ub[2])
    X = [xmin, xmin, xmax, xmax, xmin]
    Y = [ymin, ymax, ymax, ymin, ymin]
    plot!(X, Y, c=:black, lw=2, aspect_ratio=:equal, legend=false)

    # Plot Obstacles
    for obstacle ∈ env.obstacles
        w = obstacle.ub[1] - obstacle.lb[1]
        l = obstacle.ub[2] - obstacle.lb[2]
        square = rectangle(obstacle.lb[1], obstacle.lb[2], w, l)
        plot!(square, lw=2, color=:black, fillalpha=1, legend=:none)
        # if obstacle["type"] == "Square"
        #     square = rectangleCenter(obstacle["x"], obstacle["y"], obstacle["l"], obstacle["l"])
        #     plot!(square, lw=2, color=:black, fillalpha=1)
        # elseif obstacle["type"] == "Circle"
        #     circle = circleShape(obstacle["x"], obstacle["y"], obstacle["r"])
        #     drawGeometry(circle; lw=2, color=:black, fillalpha=1)
        # end
    end
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


function plotCLF(iter, counterExamples, env, params, lfs, filename="")

    plot_env(env)

    x = range(1.1*env.workspace.lb[1], 1.1*env.workspace.ub[1], length=100)
    y = range(1.1*env.workspace.lb[2], 1.1*env.workspace.ub[2], length=100)
    Vtemp(x_, y_) = clfjl.V([x_, y_], lfs)
    z = @. Vtemp(x', y)
    contour!(x, y, Vtemp, levels=[0], color=:red, style=:dot, linewidth=2, legend=:none)
    contour!(x, y, z, levels=100, color=:turbo, colorbar=true)

    listOfPoints = map(c -> ifelse(c.isTerminal, [c.x], [c.x]), counterExamples)
    XY = reduce(vcat, listOfPoints; init=Vector{Vector{Float64}}())
    X = map(x -> x[1], XY)
    Y = map(x -> x[2], XY)
    scatter!(X, Y, markersize = 3)

    xmin = Float64(env.workspace.lb[1])
    xmax = Float64(env.workspace.ub[1])
    ymin = Float64(env.workspace.lb[2])
    ymax = Float64(env.workspace.ub[2])

    del, vor, _ = deldir(X, Y, [xmin, xmax, ymin, ymax], 1e-9)
    # Dx, Dy = edges(del)
    Vx, Vy = edges(vor)

    plot!(Vx, Vy, style=:dash, color=:black, label = "Voronoi")
    if isnothing(params.imgFileDir)
        print(pwd())
    else
        filepath = joinpath(params.imgFileDir, "$filename$iter.png")
        savefig(filepath)
    end
    nothing
end
