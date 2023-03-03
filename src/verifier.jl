function verifyCandidateCLF(
        counterExamples::Vector{CounterExample},
        hybridSystem::HybridSystem,
        lfs::Vector{Tuple{Vector{Float64}, Float64}},
        params::Parameters,
        workspace::Workspace,
        solver
    )

    N = 2

    safeCounterExamples = filter(c->!c.isUnsafe, counterExamples)
    unsafeCounterExamples = filter(c->c.isUnsafe, counterExamples)

    maxGap = -1.0
    bestCounterExamplePoint = [.0, .0, .0]
    bestCounterExample = nothing

    for counterExample in safeCounterExamples

        model = solver()
        x = @variable(model, [1:N])
        maxV = @variable(model)

        # Constraint 1: x ∈ S
        @constraint(model, x .≥ workspace.lb[1:N])
        @constraint(model, x .≤ workspace.ub[1:N])

        # Constraint 2: x ∈ R_k
        # Voronoi Constraints s.t. x is contained in the same area as the witness
        for otherCounterExample in filter(x->x!=counterExample, counterExamples)
            diff = (otherCounterExample.x-counterExample.x)[1:N]
            vecx = x - counterExample.x[1:N]
            @constraint(model, dot(diff, vecx) ≤ norm(diff, 2)^2 / 2)
        end

        # Constraint 3: V(x) <= γ, if γ>0, then V(x)>0
        for lf in lfs
            a, b = lf
            valx = dot(a, x) + b
            # valx: -100 <= γ, γ=-100
            # valx: 100 <= γ, γ=100
            # The maximum V that satisfies all constraints is 100.
            @constraint(model, valx <= maxV)
        end

        @objective(model, Min, maxV)
        optimize!(model)

        if value.(maxV) > 0
            maxV = value.(maxV)
            print("Found an counterExample in the safe Voronoi region ")
            println("with the violation of $maxV")
            if maxV > maxGap
                maxGap = maxV
                bestCounterExamplePoint[1:N] = value.(x)
                bestCounterExample = counterExample
            end
        end
    end

    for counterExample in unsafeCounterExamples

        model = solver()
        x = @variable(model, [1:N])
        γ = @variable(model)

        # Constraint 1: x ∈ S
        @constraint(model, x .≥ workspace.lb[1:N])
        @constraint(model, x .≤ workspace.ub[1:N])

        # Constraint 2: x ∈ R_k
        # Voronoi Constraints s.t. x is contained in the same area as the witness
        for otherCounterExample in filter(x->x!=counterExample, counterExamples)
            diff = (otherCounterExample.x-counterExample.x)[1:N]
            vecx = x - counterExample.x[1:N]
            @constraint(model, dot(diff, vecx) ≤ norm(diff, 2)^2 / 2)
        end

        # Constraint 3: V_j(x)≥γ ∀j, If γ≤0 => V(x)≤0
        for lf in lfs
            a, b = lf
            valx = dot(a, x) + b
            @constraint(model, valx ≥ γ)
        end

        @objective(model, Max, γ)
        optimize!(model)

        if value.(γ) <= 0
            γ = -value.(γ)
            print("Found an counterExample in the unsafe Voronoi region ")
            println("with the violation of $γ")

            if γ > maxGap
                maxGap = γ
                bestCounterExamplePoint[1:N] = value.(x)
                bestCounterExample = counterExample
            end
        end
    end

    # TODO: Must check if the current Polyhedron@x intersects with obstacles
    for counterExample in counterExamples
        if counterExample.isUnsafe
            continue
        end
        for lf in lfs

            model = solver()
            x = @variable(model, [1:N])
            # maxVx = @variable(model, upper_bound=0)
            # gap = @variable(model, upper_bound=params.maxLyapunovGapForVerifier)
            gap = @variable(model)

            # Constraint 1: x ∈ S
            @constraint(model, x .≥ workspace.lb[1:N])
            @constraint(model, x .≤ workspace.ub[1:N])

            # Constraint 2: x ∈ R_k
            # Voronoi Constraints s.t. x is contained in the same area as the witness
            for otherCounterExample in filter(x->x!=counterExample, counterExamples)
                diff = (otherCounterExample.x-counterExample.x)[1:N]
                vecx = x - counterExample.x[1:N]
                @constraint(model, dot(diff, vecx) ≤ norm(diff, 2)^2 / 2)
            end

            # Constraint 3: V(x′) >= V(x)
            # It's suppose to be V(x') < V(x) but we want to find a counterexample,
            # such that V(x') >= V(x) + maxGap
            # Recall V(x) = max_j aj^T * x + bj,
            x′ = counterExample.dynamics.A[1:N,1:N]*x + counterExample.dynamics.b[1:N]
            a, b = lf
            valy = dot(a[1:N], x′) + b
            for lf in lfs
                a, b = lf
                valx = dot(a[1:N], x) + b
                # There's only 1 valy. For example, we get
                # -10 >= -10 + gap,     => gap=0
                # -10 >= -30 + gap ,    => gap=20
                # In this case V(y)=-10 and V(x)=max aj^T * x + bj = -10
                # Thus, the maximum gap that satisfies all the constraints is 0.
                # Larger the gap is, the better the counterexample is.
                @constraint(model, valy ≥ valx + gap)
            end

            @objective(model, Max, gap)
            optimize!(model)

            # if gap is <0, then Lyapunov function does not hold anymore. Found counterexample
            # if primal_status(model) == FEASIBLE_POINT
            if primal_status(model) == FEASIBLE_POINT && value.(gap) > maxGap
                maxGap = value.(gap)
                bestCounterExamplePoint[1:N] = value.(x)
                bestCounterExample = counterExample
            end
        end
    end

    if maxGap >= 0
        println("|--------------------------------------|")
        println("Found a counterExample $bestCounterExamplePoint that satisfies V(y)>=V(x)")
        vertices = findVoronoiRegion(solver, workspace, bestCounterExample, counterExamples)
        println("Searched Region: ", vertices)
        println(bestCounterExample.x, " -> ", bestCounterExample.y)
        println("By Taking Dynamics: ", bestCounterExample.dynamics)
        println("|--------------------------------------|")
    end

    return bestCounterExamplePoint, maxGap
end


function findVoronoiRegion(solver, workspace, counterExample, counterExamples)

    N = 2
    function solve(idim)
        Xs = []
        model = solver()
        x = @variable(model, [1:N])
        # Constraint 1: x ∈ S
        @constraint(model, x .≥ workspace.lb[1:N])
        @constraint(model, x .≤ workspace.ub[1:N])
        # Constraint 2: x ∈ R_k
        # Voronoi Constraints s.t. x is contained in the same area as the witness
        for otherCounterExample in filter(x->x!=counterExample, counterExamples)
            diff = (otherCounterExample.x-counterExample.x)[1:N]
            vecx = x - counterExample.x[1:N]
            @constraint(model, dot(diff, vecx) ≤ norm(diff, 2)^2 / 2)
        end
        @objective(model, Min, x[idim])
        optimize!(model)
        push!(Xs, value.(x))

        model = solver()
        x = @variable(model, [1:N])
        # Constraint 1: x ∈ S
        @constraint(model, x .≥ workspace.lb[1:N])
        @constraint(model, x .≤ workspace.ub[1:N])
        # Constraint 2: x ∈ R_k
        # Voronoi Constraints s.t. x is contained in the same area as the witness
        for otherCounterExample in filter(x->x!=counterExample, counterExamples)
            diff = (otherCounterExample.x - counterExample.x)[1:N]
            vecx = x - counterExample.x[1:N]
            @constraint(model, dot(diff, vecx) ≤ norm(diff, 2)^2 / 2)
        end
        @objective(model, Max, x[idim])
        optimize!(model)
        push!(Xs, value.(x))
        return Xs
    end

    return reduce(vcat, map(i->solve(i), 1:N); init=Vector{Vector{Float64}})
end
