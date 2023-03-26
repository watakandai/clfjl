using JLD2
using clfjl
using Plots


function plotTrajectories(trajectories, env; imgFileDir::String=pwd())
    clfjl.plotEnv(env)
    for (status, trajectory) in trajectories
        x0 = trajectory[1][1]
        y0 = trajectory[1][2]
        X = [x[1] for x in trajectory]
        Y = [x[2] for x in trajectory]
        plot!(X, Y, lw=2, label=String(Symbol(status)))
        scatter!([x0], [y0], markersize=2, shape=:circle)
    end
    if !isdir(imgFileDir)
        mkdir(imgFileDir)
    end
    filepath = joinpath(imgFileDir, "SimulatedControlTrajectories.png")
    savefig(filepath)
end


function main(;imgFileDir::String=pwd())
    @load joinpath(@__DIR__, "learnedCLFs") lfs counterExamples env
    trajectories = clfjl.simulateWithCLFs(lfs, counterExamples, env; numStep=1000)

    inTerminal(x) = all(env.termSet.lb .<= x) && all(x .<+ env.termSet.ub)
    outOfBound(x) = any(x .<= env.workspace.lb) && any(env.workspace.ub .<= x)
    inObstacles(x) = any(map(o->all(o.lb .≤ x) && all(x .≤ o.ub), env.obstacles))

    println(env.termSet)
    plotTrajectories(trajectories, env; imgFileDir=imgFileDir)
end



main()
