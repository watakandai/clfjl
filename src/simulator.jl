@enum SimStatus begin
    SIM_TERMINATED = 0
    SIM_INFEASIBLE = 1
    SIM_UNSAFE = 2
    SIM_MAX_ITER_REACHED = 3
end
using LinearAlgebra

function simulateWithCLFs(x0, lfs, counterExamples, env; numStep=100)::Tuple{SimStatus, StateTraj}

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

    for _ in 1:numStep
        i = argmin(norm.(map(cex-> cex - x, ceX), 2))
        counterExample = counterExamples[i]
        x′ = counterExample.dynamics.A * x + counterExample.dynamics.b

        # nextX = map(d -> d.A * x + d.b, dynamicsList)
        # i = argmin([V(xn, lfs) for xn in nextX])
        # x′ = nextX[i]

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
