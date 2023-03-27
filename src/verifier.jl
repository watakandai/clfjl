function verifyCandidateCLF(
    counterExamples::Vector{CounterExample},
    lfs::LyapunovFunctions,
    env::Env,
    solver,
    N::Integer,
    thresholdLyapunovGapForVerifier::Real;
    unreachableRegions=nothing)::Tuple{Vector, Float64}

    # isInitSetNegative(lfs, env, solver, N)

    maxGap = -1.0
    bestCounterExamplePoint = zeros(N)
    bestCounterExample = nothing

    for counterExample in counterExamples

        if counterExample.isUnsafe
            continue
        end

        inTerminal = isInTerminalRegion(counterExample, counterExamples, env, solver, N)
        # inTerminal = true

        if inTerminal
            dim = Vector(1:N)
            opBoundInd = Vector(1:2)
            isLB = [true, false]
            bounds = [env.termSet.lb, env.termSet.ub]
            for (idim, iopb) in Iterators.product(dim, opBoundInd)
                (maxGap,
                 bestCounterExamplePoint,
                 bestCounterExample) = _verifyCandidateCLF(
                    counterExample,
                    counterExamples,
                    lfs,
                    env,
                    solver,
                    N,
                    maxGap,
                    bestCounterExamplePoint,
                    bestCounterExample,
                    dim=idim,
                    lb=isLB[iopb],
                    bound=bounds[iopb],
                    unreachableRegions=unreachableRegions)
            end
        else
            (maxGap,
             bestCounterExamplePoint,
             bestCounterExample) = _verifyCandidateCLF(
                counterExample,
                counterExamples,
                lfs,
                env,
                solver,
                N,
                maxGap,
                bestCounterExamplePoint,
                bestCounterExample,
                unreachableRegions=unreachableRegions)
        end
    end

    if maxGap >= 0
        println("|--------------------------------------|")
        x = bestCounterExample.x
        println("Found a counterExample $bestCounterExamplePoint that satisfies V(y)>=V(x) at $x")
        x = bestCounterExamplePoint
        y = bestCounterExample.dynamics.A * x + bestCounterExample.dynamics.b
        println(V(y, lfs), " >= ", V(x, lfs), " + $maxGap")
        println("By Taking Dynamics: ", bestCounterExample.dynamics)
        println("|--------------------------------------|")
    end

    return bestCounterExamplePoint, maxGap
end


function _verifyCandidateCLF(
    counterExample::CounterExample,
    counterExamples::Vector{CounterExample},
    lfs::LyapunovFunctions,
    env::Env,
    solver,
    N::Integer,
    maxGap,
    bestCounterExamplePoint,
    bestCounterExample;
    dim=nothing, lb=nothing, bound=nothing, unreachableRegions=nothing)

    for lf in lfs

        model = solver()
        x = @variable(model, [1:N])
        gap = @variable(model)

        # x ∉ U
        if !isnothing(unreachableRegions)
            for lfs in unreachableRegions
                for lf in lfs
                    # NOTE: WE DEFINED UNSAFE REGION TO BE a^Tx+b<=0 ONLY in the precheker.jl
                    # It was only meant to OMPL to be used but I'm using it now
                    @constraint(model, takeImage(lf, x) >= 0)
                end
            end
        end

        # @constraint(model, x .== [-0.45533232057538586, 0.12976138563310022])

        # Constraint 1: x ∈ S
        @constraint(model, x .≥ env.workspace.lb)
        @constraint(model, x .≤ env.workspace.ub)
        if !isnothing(dim) && !isnothing(bound)
            if lb
                @constraint(model, x[dim] ≤ bound[dim])
            else
                @constraint(model, x[dim] ≥ bound[dim])
            end
        end
        # Constraint 2: x ∈ R_k
        # Voronoi Constraints s.t. x is contained in the same area as the witness
        for otherCounterExample in filter(c->c!=counterExample && !c.isUnsafe, counterExamples)
            diff = (otherCounterExample.x-counterExample.x)
            vecx = x - counterExample.x
            @constraint(model, dot(diff, vecx) ≤ norm(diff, 2)^2 / 2)
        end

        # Constraint 3: V(x′) >= V(x)
        # It's suppose to be V(x') < V(x) but we want to find a counterexample,
        # such that V(x') >= V(x) + maxGap
        # Recall V(x) = max_j aj^T * x + bj,
        x′ = counterExample.dynamics.A * x + counterExample.dynamics.b
        valy = takeImage(lf, x′)
        for lf in lfs
            valx = takeImage(lf, x)
            # There's only 1 valy. For example, we get
            # -10 >= -10 + gap,     => gap=0
            # -10 >= -30 + gap ,    => gap=20
            # In this case V(y)=-10 and V(x)=max aj^T * x + bj = -10
            # Thus, the maximum gap that satisfies all the constraints is 0.
            # Larger the gap is, the better the counterexample is.
            @constraint(model, valy ≥ valx + gap)
            # Only search in the safe region.
            @constraint(model, valx ≤ 0)
        end

        @objective(model, Max, gap)
        optimize!(model)

        # if gap is <0, then Lyapunov function does not hold anymore. Found counterexample
        if primal_status(model) == FEASIBLE_POINT && value.(gap) > maxGap
            maxGap = value.(gap)
            bestCounterExamplePoint = value.(x)
            bestCounterExample = counterExample
        end
    end
    return maxGap, bestCounterExamplePoint, bestCounterExample
end


function isInTerminalRegion(counterExample, counterExamples, env, solver, N)

    model = solver()
    x = @variable(model, [1:N])

    # Constraint 1: x ∈ S
    @constraint(model, x .≥ env.workspace.lb)
    @constraint(model, x .≤ env.workspace.ub)
    @constraint(model, x .≥ env.termSet.lb)
    @constraint(model, x .≤ env.termSet.ub)

    # Constraint 2: x ∈ R_k
    # Voronoi Constraints s.t. x is contained in the same area as the witness
    for otherCounterExample in filter(c->c!=counterExample && !c.isUnsafe, counterExamples)
        diff = (otherCounterExample.x-counterExample.x)
        vecx = x - counterExample.x
        @constraint(model, dot(diff, vecx) ≤ norm(diff, 2)^2 / 2)
    end

    @objective(model, Min, 0)
    optimize!(model)
    if primal_status(model) == FEASIBLE_POINT
        return true
    end
    return false
end


function isInitSetNegative(lfs::LyapunovFunctions,
                           env::Env,
                           solver,
                           N::Integer)::Bool
    currMaxV = -1
    currMaxX = nothing

    for lf in lfs
        model = solver()
        x = @variable(model, [1:N])
        # maxV = @variable(model, upper_bound=1)
        obj = @variable(model)

        # Constraint 1: x ∈ I
        @constraint(model, x .≥ env.initSet.lb)
        @constraint(model, x .≤ env.initSet.ub)

        # Constraint 3: V(x) <= γ, if γ>0, then V(x)>0
        valx = takeImage(lf, x)
        # valx: -100 <= γ, γ=-100
        # valx: 100 <= γ, γ=100
        # The maximum V that satisfies all constraints is 100.
        # @constraint(model, valx <= maxV)
        @constraint(model, valx >= obj)

        @objective(model, Max, obj)
        # @objective(model, Max, maxV)
        optimize!(model)

        if primal_status(model) == FEASIBLE_POINT
            x = value.(x)
            obj = objective_value(model)
            if obj > currMaxV
                currMaxV = obj
                currMaxX = x
            end
        end
    end

    if currMaxV > 0
        throw(DomainError(currMaxV, "∃x ∈ I s.t. Vx)>0. x=$currMaxX"))
        return false
    end
    return true
end
