using Combinatorics
using JLD2
using LinearAlgebra
using JuMP
using Gurobi


function simulateWithCLFs(x0,
                          lfs::LyapunovFunctions,
                          counterExamples::CounterExamples,
                          env::Env;
                          minStep=tempMinStep,
                          numStep::Integer=100,
                          withVoronoiControl::Bool=true,
                          checkLyapunovCondition::Bool=false)::SimTrajectory

    x = x0
    X = [x0]
    Vs = [V(x, lfs)]
    if length(counterExamples) == 0
        return SimTrajectory(X, Vs, SIM_INFEASIBLE)
    end

    inTerminal(x) = all(env.termSet.lb .<= x) && all(x .<+ env.termSet.ub)
    outOfBound(x) = any(x .< env.workspace.lb) || any(env.workspace.ub .< x)
    inObstacles(x) = any(map(o->all(o.lb .≤ x) && all(x .≤ o.ub), env.obstacles))

    safeCEs = filter(c -> !c.isUnsafe, counterExamples)

    if outOfBound(x0) || inObstacles(x0)
        status = SIM_UNSAFE
        return SimTrajectory(X, Vs, status)
    end

    for iStep in 1:numStep
        if withVoronoiControl
            i = argmin(norm.(map(c -> c.x - x, safeCEs), 2))
            c = safeCEs[i]
            x′ = c.dynamics.A * x + c.dynamics.b
            v′ = V(x′, lfs)
        else
            x′, v′ = minStep(x, lfs, safeCEs)
        end

        if checkLyapunovCondition && v′ >= Vs[end]
            status = SIM_INFEASIBLE
            return SimTrajectory(X, Vs, status)
        end

        push!(X, x′)
        push!(Vs, v′)
        x = x′

        if inTerminal(x)
            status = SIM_TERMINATED
            return SimTrajectory(X, Vs, status)
        elseif outOfBound(x) || inObstacles(x)
            status = SIM_UNSAFE
            return SimTrajectory(X, Vs, status)
        end

    end
    status = SIM_MAX_ITER_REACHED
    return SimTrajectory(X, Vs, status)
end


function tempMinStep(x::Vector{<:Real}, lfs::LyapunovFunctions, counterExamples::CounterExamples)
    dynamicsList = map(c -> c.dynamics, counterExamples)
    nextX = map(d -> d.A * x + d.b, dynamicsList)
    Vs = [V(xn, lfs) for xn in nextX]
    i = argmin(Vs)
    return nextX[i], Vs[i]
end


function defaultMinStep(x::Vector{<:Real}, lfs::LyapunovFunctions, stepFunc, Nu::Integer)
    """
    We can optimize the input u s.t. min V(x')
    Find Input U that min V(x')
    Dyanmics:   x' = A*x + B*u                          (Affine Dynamics)
             OR x'= Ai*x + Bi*ui for i∈I, if x∈Xi       (Piecewise Affine)
    """

    # Let's say Nu=2 for now.
    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(Gurobi.Env()),
                                        "OutputFlag"=>false))

    vmin = @variable(model)
    u = @variable(model, [1:Nu])
    x′ = stepFunc(x, u)
    for lf in lfs
        @constraint(model, vmin <= takeImage(lf, x′))
    end

    @objective(model, Max, vmin)
    optimize(model)

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    return value(x′), value(vmin)
end


function simulateWithCLFs(lfs::LyapunovFunctions,
                          counterExamples::CounterExamples,
                          env::Env;
                          numSample::Integer=10,
                          numStep::Integer=100,
                          withVoronoiControl::Bool=true,
                          checkLyapunovCondition::Bool=false)::SimTrajectories
    # Try both. Want to see if the latter one works.
    n = length(env.initSet.ub)
    function rndX()
        return env.initSet.lb + rand(n) .* (env.initSet.ub - env.initSet.lb)
    end
    return map(i -> simulateWithCLFs(rndX(),
                                     lfs,
                                     counterExamples,
                                     env;
                                     numStep=numStep,
                                     withVoronoiControl=withVoronoiControl), 1:numSample)
end


function plotTrajectories(trajectories::SimTrajectories,
                          lfs::LyapunovFunctions,
                          env::Env;
                          imgFileDir::String=pwd(),
                          filename::String="SimulatedControlTrajectories",
                          numTraj::Integer=-1)
    if length(lfs[1].a) == 2
        plotTrajectories2D(trajectories, lfs, env;
                           imgFileDir=imgFileDir,
                           filename=filename,
                           numTraj=numTraj)
    elseif length(lfs[1].a) == 3
        plotTrajectories3D(trajectories, lfs, env;
                           imgFileDir=imgFileDir,
                           filename=filename,
                           numTraj=numTraj)
    else
        # do nothing
    end
end


function plotTrajectories2D(trajectories::SimTrajectories,
                            lfs::LyapunovFunctions,
                            env::Env;
                            imgFileDir::String=pwd(),
                            filename::String="SimulatedControlTrajectories",
                            numTraj::Integer=-1)

    allSafe = all([traj.status == SIM_TERMINATED for traj in trajectories])
    filename = allSafe ? string(filename,"AllSafe") : filename
    plotOnlyFailures = !allSafe

    plotEnv(env)

    dxy = 0.1 * (env.workspace.ub - env.workspace.lb)
    Xs = range(env.workspace.lb[1]-dxy[1], env.workspace.ub[1]+dxy[1], length=100)
    Ys = range(env.workspace.lb[2]-dxy[2], env.workspace.ub[2]+dxy[2], length=100)
    Vtemp(x_, y_) = V([x_, y_], lfs)
    Zs = @. Vtemp(Xs', Ys)
    contour!(Xs, Ys, Vtemp, levels=[0], color=:red, style=:dot, linewidth=2, legend=:none)
    contour!(Xs, Ys, Zs, levels=100, color=:turbo, colorbar=true)

    iter = 1
    for simTrajectory in trajectories
        if plotOnlyFailures && simTrajectory.status == SIM_TERMINATED
            continue
        end
        X = [x[1] for x in simTrajectory.X]
        Y = [x[2] for x in simTrajectory.X]
        plot!(X, Y, lw=2, label=String(Symbol(simTrajectory.status)), arrow=(:open, 0.5))
        iter += 1
        if numTraj > 0 && iter > numTraj
            break
        end
    end
    savefigure(imgFileDir, "$filename.png")
end


function plotTrajectories3D(trajectories::SimTrajectories,
                            lfs::LyapunovFunctions,
                            env::Env;
                            imgFileDir::String=pwd(),
                            filename::String="SimulatedControlTrajectories",
                            numTraj::Integer=-1)

    allSafe = all([traj.status == SIM_TERMINATED for traj in trajectories])
    filename = allSafe ? string(filename,"AllSafe") : filename
    plotOnlyFailures = !allSafe

    axis = ["x", "y", "z"]

    for c in collect(combinations(1:3,2))
        i, j = c
        plotEnv2DDim(env, i, j)
        iter = 1
        for simTrajectory in trajectories
            if plotOnlyFailures && simTrajectory.status == SIM_TERMINATED
                continue
            end
            X = [x[i] for x in simTrajectory.X]
            Y = [x[j] for x in simTrajectory.X]
            plot!(X, Y, lw=2, label=String(Symbol(simTrajectory.status)))

            iter += 1
            if numTraj > 0 && iter > numTraj
                break
            end
        end
        savefigure(imgFileDir, "$filename$(axis[i])$(axis[j]).png")
    end
end


function plotEnv2DDim(env::Env, i::Integer, j::Integer)
    xmin = Float64(env.workspace.lb[i])
    xmax = Float64(env.workspace.ub[i])
    ymin = Float64(env.workspace.lb[j])
    ymax = Float64(env.workspace.ub[j])

    X = [xmin, xmin, xmax, xmax, xmin]
    Y = [ymin, ymax, ymax, ymin, ymin]
    plot(X, Y, c=:black, lw=2, aspect_ratio=:equal, legend=false,
        guidefontsize=18,
        tickfontsize=16,
        right_margin=12Plots.mm,
        left_margin=12Plots.mm)

    xmin = Float64(env.initSet.lb[i])
    xmax = Float64(env.initSet.ub[i])
    ymin = Float64(env.initSet.lb[j])
    ymax = Float64(env.initSet.ub[j])
    X = [xmin, xmin, xmax, xmax, xmin]
    Y = [ymin, ymax, ymax, ymin, ymin]
    plot!(X, Y, c=:blue, lw=2, aspect_ratio=:equal, legend=false)

    xmin = Float64(env.termSet.lb[i])
    xmax = Float64(env.termSet.ub[i])
    ymin = Float64(env.termSet.lb[j])
    ymax = Float64(env.termSet.ub[j])
    X = [xmin, xmin, xmax, xmax, xmin]
    Y = [ymin, ymax, ymax, ymin, ymin]
    plot!(X, Y, c=:green, lw=2, aspect_ratio=:equal, legend=false)
end
