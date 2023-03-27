@enum SimStatus begin
    SIM_TERMINATED = 0
    SIM_INFEASIBLE = 1
    SIM_UNSAFE = 2
    SIM_MAX_ITER_REACHED = 3
end
using JLD2
using LinearAlgebra

function simulateWithCLFs(x0, lfs, counterExamples, env; numStep=100, withVoronoiControl::Bool=true)::Tuple{SimStatus, StateTraj}

    inTerminal(x) = all(env.termSet.lb .<= x) && all(x .<+ env.termSet.ub)
    outOfBound(x) = any(x .<= env.workspace.lb) && any(env.workspace.ub .<= x)
    inObstacles(x) = any(map(o->all(o.lb .≤ x) && all(x .≤ o.ub), env.obstacles))

    x = x0
    X = [x0]
    ceX = map(c -> c.x, counterExamples)
    dynamicsList = map(c -> c.dynamics, counterExamples)

    if outOfBound(x0) || inObstacles(x0)
        status = SIM_UNSAFE
        return status, X
    end

    for iStep in 1:numStep
        if withVoronoiControl
            i = argmin(norm.(map(cex-> cex - x, ceX), 2))
            counterExample = counterExamples[i]
            x′ = counterExample.dynamics.A * x + counterExample.dynamics.b
        else
            nextX = map(d -> d.A * x + d.b, dynamicsList)
            Vs = [V(xn, lfs) for xn in nextX]
            i = argmin(Vs)
            x′ = nextX[i]
        end

        push!(X, x′)
        x = x′
        if inTerminal(x)
            status = SIM_TERMINATED
            return status, X
        elseif outOfBound(x) || inObstacles(x)
            status = SIM_UNSAFE
            return status, X
        end
    end
    status = SIM_MAX_ITER_REACHED
    return status, X
end


function simulateWithCLFs(lfs, counterExamples, env;
                          numSample::Integer=10, numStep::Integer=100)::Vector{Tuple{SimStatus, StateTraj}}
    # Try both. Want to see if the latter one works.
    n = length(env.initSet.ub)
    function rndX()
        return env.initSet.lb + rand(n) .* (env.initSet.ub - env.initSet.lb)
    end
    return [simulateWithCLFs(rndX(), lfs, counterExamples, env, numStep=numStep) for i in 1:numSample]
end


function plotTrajectories(trajectories, lfs, env; imgFileDir::String=pwd(), filename="SimulatedControlTrajectories")
    plotEnv(env)

    x = range(1.1*env.workspace.lb[1], 1.1*env.workspace.ub[1], length=100)
    y = range(1.1*env.workspace.lb[2], 1.1*env.workspace.ub[2], length=100)
    Vtemp(x_, y_) = V([x_, y_], lfs)
    z = @. Vtemp(x', y)
    contour!(x, y, Vtemp, levels=[0], color=:red, style=:dot, linewidth=2, legend=:none)
    contour!(x, y, z, levels=100, color=:turbo, colorbar=true)

    for (status, trajectory) in trajectories
        x0 = trajectory[1][1]
        y0 = trajectory[1][2]
        X = [x[1] for x in trajectory]
        Y = [x[2] for x in trajectory]
        plot!(X, Y, lw=2, label=String(Symbol(status)))
        scatter!(X, Y, color=:blue)
        scatter!([x0], [y0], color=:red, markersize=2, shape=:circle)
    end

    if !isdir(imgFileDir)
        mkdir(imgFileDir)
    end
    filepath = joinpath(imgFileDir, "$filename.png")
    savefig(filepath)
end


