import YAML
using DelimitedFiles
# import VoronoiCells: Point2, Rectangle, voronoicells, corner_coordinates#, plot as plotVoronoi
# import VoronoiCells: Point2
# using GeometryBasics
using Plots
using Deldir
# using VoronoiCells
# using VoronoiDelaunay

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

function plot_env(config)
    # Plot the Boundary
    xmin, xmax, ymin, ymax =
        config["xBound"][1], config["xBound"][2], config["yBound"][1], config["yBound"][2]
    X = [xmin, xmin, xmax, xmax, xmin]
    Y = [ymin, ymax, ymax, ymin, ymin]
    plot(X, Y, c=:black, lw=2, aspect_ratio=:equal, legend=false)
    # Plot Obstacles
    if isnothing(config["obstacles"])
        return
    end

    for obstacle ∈ config["obstacles"]
        if obstacle["type"] == "Square"
            square = rectangleCenter(obstacle["x"], obstacle["y"], obstacle["l"], obstacle["l"])
            plot!(square, lw=2, color=:black, fillalpha=1)
        elseif obstacle["type"] == "Circle"
            circle = circleShape(obstacle["x"], obstacle["y"], obstacle["r"])
            drawGeometry(circle; lw=2, color=:black, fillalpha=1)
        end
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


function plotCLF(iter, config, counterExamples, hybridSystem, lfs, x, filename, filedir=nothing)

    # println("Plotting Env")
    plot_env(config)

    # println("Plot Contour")
    f(x, y) = maximum(map(lf -> dot(lf[1], [x,y]) + lf[2], lfs))
    x = range(1.1*config["xBound"][1], 1.1*config["xBound"][2], length=100)
    y = range(1.1*config["yBound"][1], 1.1*config["yBound"][2], length=100)
    z = @. f(x', y)
    contour!(x, y, z, levels=20, color=:turbo, colorbar=true)

    # println("Plotting CounterExamples")
    # listOfPoints = map(c -> ifelse(c.isTerminal, [c.x, c.y], [c.x]), counterExamples)
    listOfPoints = map(c -> ifelse(c.isTerminal, [c.x], [c.x]), counterExamples)
    XY = reduce(vcat, listOfPoints; init=Vector{Vector{Float64}}())
    X = map(x -> x[1], XY)
    Y = map(x -> x[2], XY)
    scatter!(X, Y, markersize = 3)

    # println("Converting CounterExamples X, Y to Point2")
    xmin = Float64(config["xBound"][1])
    xmax = Float64(config["xBound"][2])
    # xdist = xmax - xmin
    ymin = Float64(config["yBound"][1])
    ymax = Float64(config["yBound"][2])
    # ydist = ymax - ymin

    # X = map(x -> min(max(0, x / xdist + 0.5), 1), X)
    # Y = map(y -> min(max(0, y / ydist + 0.5), 1), Y)
    # println("Constructing Voronoi Cells")
    del, vor, summ = deldir(X, Y, [xmin, xmax, ymin, ymax], 1e-9)
    Dx, Dy = edges(del)
    Vx, Vy = edges(vor)

    # println("Plotting Points")
    # scatter!(points, markersize = 6, label = "generators")
    # println("Plotting Voronoi Edges")
    # ps = corner_coordinates(tess)
    # plot!(Dx, Dy, label = "Delaunay")
    plot!(Vx, Vy, style = :dash, label = "Voronoi")
    if isnothing(filedir)
        print(pwd())
    else
        filepath = joinpath(filedir, "$filename$iter.png")
        savefig(filepath)
    end
    nothing
end


function plotSamples(iter, config, counterExamples, hybridSystem, lfs, x, filename)

    # println("Plotting Env")
    plot_env(config)

    # println("Plotting X, Y")
    listOfPoints = map(c -> ifelse(c.isTerminal, [c.x, c.y], [c.x]), counterExamples)
    XY = reduce(vcat, listOfPoints; init=Vector{Vector{Float64}}())
    X = map(x -> x[1], XY)
    Y = map(x -> x[2], XY)
    scatter!(X, Y, markersize = 3)

    savefig("$filename$iter.png")
    nothing
end
