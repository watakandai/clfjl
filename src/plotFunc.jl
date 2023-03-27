import YAML
using DelimitedFiles
using Plots
using Deldir
using LinearAlgebra

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


function plotEnv(env)
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


function plot2DCLF(iter, counterExamples, env, params, lfs; regions=nothing, filename="")

    plotEnv(env)
    plotUnreachableRegion(env, regions=regions)
    x = range(1.1*env.workspace.lb[1], 1.1*env.workspace.ub[1], length=100)
    y = range(1.1*env.workspace.lb[2], 1.1*env.workspace.ub[2], length=100)
    Vtemp(x_, y_) = clfjl.V([x_, y_], lfs)
    z = @. Vtemp(x', y)
    contour!(x, y, Vtemp, levels=[0], color=:red, style=:dot, linewidth=2, legend=:none)
    contour!(x, y, z, levels=100, color=:turbo, colorbar=true)

    for ce in filter(c -> !c.isUnsafe, counterExamples)
        x = ce.x
        y = ce.y
        plot!([x[1], y[1]], [x[2], y[2]], arrow=(:open, 1), markershapes=[:circle, :star5], markersize=2, color=:blue)
    end
    # ces = reduce(vcat, listOfPoints; init=Vector{Vector{Float64}}())
    # X = map(c -> c.x[1], ces)
    # Y = map(c -> c.x[2], ces)
    # scatter!(X, Y, markersize = 3)
    scatter!([counterExamples[end].x[1]], [counterExamples[end].x[2]], markersize = 3, color=:black)

    unsafeCes = filter(c -> c.isUnsafe, counterExamples)
    # XY_ = reduce(vcat, unSafelistOfPoints; init=Vector{Vector{Float64}}())
    X_ = map(c -> c.x[1], unsafeCes)
    Y_ = map(c -> c.x[2], unsafeCes)
    scatter!(X_, Y_, markersize=2, color=:red)

    xmin = Float64(env.workspace.lb[1])
    xmax = Float64(env.workspace.ub[1])
    ymin = Float64(env.workspace.lb[2])
    ymax = Float64(env.workspace.ub[2])

    # del, vor, _ = deldir(X, Y, [xmin, xmax, ymin, ymax], 1e-9)
    # # Dx, Dy = edges(del)
    # Vx, Vy = edges(vor)

    # plot!(Vx, Vy, style=:dash, color=:black, label = "Voronoi")
    if isnothing(params.imgFileDir)
        print(pwd())
    else
        if !isdir(params.imgFileDir)
            mkdir(params.imgFileDir)
        end
        filepath = joinpath(params.imgFileDir, "$filename$iter.png")
        savefig(filepath)
    end
end


function plot3DCLF(iter, counterExamples, env, params, lfs; regions=nothing, filename="")

    Nd = params.optDim
    Ns = env.numSpaceDim

    x = range(1.1*env.workspace.lb[1], 1.1*env.workspace.ub[1], length=100)
    y = range(1.1*env.workspace.lb[2], 1.1*env.workspace.ub[2], length=100)
    z = range(1.1*env.workspace.lb[3], 1.1*env.workspace.ub[3], length=5)
    sumVals = sum(maximum.(map(v->abs.(collect(v)), zip(env.workspace.lb[1:Nd], env.workspace.ub[1:Nd]))))
    clims = (-sumVals, sumVals)

    for z_ in z
        plotEnv(env)
        plotUnreachableRegion(env, regions=regions)

        Vtemp(x_, y_) = clfjl.V([x_, y_, z_], lfs)
        z = @. Vtemp(x', y)

        contour!(x, y, Vtemp, levels=[0], color=:red, style=:dot, linewidth=2, legend=:none, clims=clims)
        contour!(x, y, z, levels=100, color=:turbo, colorbar=true, clims=clims)

        listOfPoints = map(c -> ifelse(c.x==c.y, [c.x], [c.x, c.y]) , counterExamples)
        XY = reduce(vcat, listOfPoints; init=Vector{Vector{Float64}}())
        X = map(x -> x[1], XY)
        Y = map(x -> x[2], XY)
        scatter!(X, Y, markersize = 3)

        xmin = Float64(env.workspace.lb[1])
        xmax = Float64(env.workspace.ub[1])
        ymin = Float64(env.workspace.lb[2])
        ymax = Float64(env.workspace.ub[2])

        listOfPoints = map(c -> [c.x], counterExamples)
        XY = reduce(vcat, listOfPoints; init=Vector{Vector{Float64}}())
        X = map(x -> x[1], XY)
        Y = map(x -> x[2], XY)
        # del, vor, _ = deldir(X, Y, [xmin, xmax, ymin, ymax], 1e-10)
        # # Dx, Dy = edges(del)
        # Vx, Vy = edges(vor)

        # plot!(Vx, Vy, style=:dash, color=:black, label = "Voronoi")

        if isnothing(params.imgFileDir)
            print(pwd())
        else
            if !isdir(params.imgFileDir)
                mkdir(params.imgFileDir)
            end
            filepath = joinpath(params.imgFileDir, "$(filename)$(iter)@z=$z_.png")
            savefig(filepath)
        end
    end
end


function plot4DCLF(iter, counterExamples, params, lfs; regions=nothing, filename="")

    Nd = params.optDim
    Ns = env.numSpaceDim

    x = range(1.1*env.workspace.lb[1], 1.1*env.workspace.ub[1], length=100)
    y = range(1.1*env.workspace.lb[2], 1.1*env.workspace.ub[2], length=100)
    # θs = range(0, 2*π, length=12)
    θs = [0]

    for θ in θs
        plotEnv(env)
        plotUnreachableRegion(env, regions=regions)

        Vtemp(x_, y_) = clfjl.V([x_, y_, cos(θ), sin(θ)], lfs)
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

        # del, vor, _ = deldir(X, Y, [xmin, xmax, ymin, ymax], 1e-4)
        # # Dx, Dy = edges(del)
        # Vx, Vy = edges(vor)

        # plot!(Vx, Vy, style=:dash, color=:black, label = "Voronoi")

        if isnothing(params.imgFileDir)
            print(pwd())
        else
            if !isdir(params.imgFileDir)
                mkdir(params.imgFileDir)
            end
            deg = 180 / pi * θ
            filepath = joinpath(params.imgFileDir, "$(filename)$(iter)@θ=$deg.png")
            savefig(filepath)
        end
    end
end


function plotUnreachableRegion(env; regions=nothing)
    if isnothing(regions)
        return
    end

    for lfs in regions
        for lf in lfs
            xmin = env.workspace.lb[1]
            xmax = env.workspace.ub[1]
            ymin = env.workspace.lb[2]
            ymax = env.workspace.ub[2]
            if isapprox(lf.a[1], 0.0)
                y = - lf.b / lf.a[2]
                plot!([xmin, xmax], [y, y], color=:black, lw=2)
            elseif isapprox(lf.a[2], 0.0)
                x = - lf.b / lf.a[1]
                plot!([x, x], [ymin, ymax], color=:black, lw=2)
            else
                tilt = - lf.a[1] / lf.a[2]
                if abs(tilt) > 1
                    fy(y) = (- lf.b - lf.a[2] * y) / lf.a[1]
                    y = range(env.workspace.lb[2], env.workspace.ub[2], length=100)
                    x = fy.(y)
                else
                    fx(x) = (- lf.b - lf.a[1] * x) / lf.a[2]
                    x = range(env.workspace.lb[1], env.workspace.ub[1], length=100)
                    y = fx.(x)
                end
                plot!(x, y, color=:black, lw=2)
            end
        end
    end
end


function plotCellDecomposition(counterExamples, rectangles, params, env)

    for rectangle in rectangles
        xmin = Float64(rectangle.lb[1])
        xmax = Float64(rectangle.ub[1])
        ymin = Float64(rectangle.lb[2])
        ymax = Float64(rectangle.ub[2])
        X = [xmin, xmin, xmax, xmax, xmin]
        Y = [ymin, ymax, ymax, ymin, ymin]
        plot!(X, Y, c=:black, lw=2, aspect_ratio=:equal, legend=false)
    end

    "Plot Sample Points"
    listOfPoints = map(c -> ifelse(c.isTerminal, [c.x], [c.x]), counterExamples)
    XY = reduce(vcat, listOfPoints; init=Vector{Vector{Float64}}())
    X = map(x -> x[1], XY)
    Y = map(x -> x[2], XY)
    scatter!(X, Y, markersize = 3)

    xmin = Float64(env.workspace.lb[1])
    xmax = Float64(env.workspace.ub[1])
    ymin = Float64(env.workspace.lb[2])
    ymax = Float64(env.workspace.ub[2])

    "Plot Voronoi Regions"
    del, vor, _ = deldir(X, Y, [xmin, xmax, ymin, ymax], 1e-7)
    Vx, Vy = edges(vor)
    plot!(Vx, Vy, style=:dash, color=:black, label = "Voronoi")
end


function plotProjectionToConvexSet(x, lfs::LyapunovFunctions)
    """
    Convex Set is represented as Ax <= b
    We want to plot a projection of point x0 onto the convex set.

    We compute a projection and the distance to each hyperplane ai^Tx + bi == 0
    """
    A = map(lf -> collect(lf.a), lfs)
    b = map(lf -> -lf.b, lfs)
    A = reduce(hcat, A)'

    minDist = Inf
    minXp = A \ b

    println("Project a point $x. current minXp=$minXp")

    for lf in lfs
        xp = x - (dot(lf.a, x) + lf.b) / norm(lf.a, 2)^2 * lf.a
        println("Line: $lf, Projection: $xp, ")

        if V(xp, lfs) <= 0
            distance = abs(dot(lf.a, x) + lf.b) / norm(lf.a, 2)
            println("Distance: $distance")
            if distance < minDist
                minDist = min(minDist, distance)
                minXp = xp
            end
        end
    end

    data = map(p -> collect(p), zip(x, minXp))
    println("Data", data)
    scatter!([x[1]], [x[2]], markersize=10, shape=:star)
    plot!(data..., color=:blue, style=:dot, markersize=3)
end
