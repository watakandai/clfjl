function verify_unsafe(solver, rect, N, lfs, γmax)
    """
    Check if ∃x ∈ U.
    """
    N = 2

    for i in 1:N
        #************** Lower Bound **************#
        model = solver()
        # variables
        x = @variable(model, [1:N])
        # γ = @variable(model, [1:N])
        γ = @variable(model, upper_bound=γmax)
        # Constraints on unsafe set
        @constraint(model, x[i] ≤ rect.lb[i])
        for (i, lf) in enumerate(lfs)
            @constraint(model, γ ≤ dot(lf[1:N], x))
        end
        @objective(model, Min, γ)
        optimize!(model)
        # Check if V(x) < 0
        println("Verifier LB Objective Value: ", objective_value(model), " at " , value.(x))
        if objective_value(model) <= 0
            return value.(x), objective_value(model)
        end

        #************** Upper Bound **************#
        model = solver()
        # variables
        x = @variable(model, [1:N])
        # γ = @variable(model, [1:N])
        γ = @variable(model, upper_bound=γmax)
        # Constraints on unsafe set
        @constraint(model, x[i] ≥ rect.ub[i])
        for (i, lf) in enumerate(lfs)
            @constraint(model, γ ≤ dot(lf[1:N], x))
        end
        @objective(model, Min, γ)
        optimize!(model)
        # Check if V(x) < 0
        # println("Verifier UB Objective Value: ", objective_value(model))
        println("Verifier UB Objective Value: ", objective_value(model), " at " , value.(x))
        if objective_value(model) <= 0
            return value.(x), objective_value(model)
        end
    end
    return [0, 0, 0], 1
    # return value.(x), 1
end

function verify_piece(wit_cls, flows, rect, lfs, M, N, Θv, Θd, γmax, solver)
    """
    """
    N = 2

    # For each witness region, we check if all states in that region
    # Lyapnov function satisfies by following its strategy ....
    witnesses = []
    flows = []
    for (i, wit_cl) in enumerate(wit_cls)
        for (j, wit) in enumerate(wit_cl)
            for q in 1:M
                for img in wit.img_cls[q]
                    push!(witnesses, wit.x)
                    push!(flows, img.flow)
                end
            end
        end
    end

    # TODO: Must check if the current Polyhedron@x intersects with obstacles
    for (witness, flow) in zip(witnesses, flows)
        for lf′ in lfs
            model = solver()
            x = @variable(model, [1:N])
            γ = @variable(model, upper_bound=γmax)

            @constraint(model, x .≥ rect.lb[1:N])
            @constraint(model, x .≤ rect.ub[1:N])

            # Voronoi Constraints s.t. x is contained in the same area as the witness
            for otherWitness in filter(x->x!=witness, witnesses)
                diff = (otherWitness-witness)[1:N]
                @constraint(model, dot(diff, x) ≤ norm(diff, Inf) / 2)
            end

            # The Last Important Constraint on Lyapnov Function
            for lf in lfs
                x′ = flow.A[1:N,1:N]*x + flow.b[1:N]
                @constraint(model, dot(lf′[1:N], x′) - dot(lf[1:N], x) ≥ γ)
            end

            @objective(model, Max, γ)
            optimize!(model)
            println(value.(x), objective_value(model))

            if value.(γ) ≥ 0
                return value.(x), objective_value(model), true
            end

        end
    end

    return [0, 0, 0], -1, false
end

function verify(
        wit_cls::Vector{<:Vector{<:Witness}},
        pieces::Vector{<:Piece},
        lfs::Vector{<:AbstractVector},
        M, N, Θv, Θd, γmax, solver
    )
    xopt::Vector{Float64} = fill(NaN, N)
    γopt::Float64 = -Inf
    kopt::Int = 0
    for (k, piece) in enumerate(pieces)

        # Check if
        # x, γ = verify_unsafe(solver, piece.rect, N, lfs, γmax)
        # found = γ < 0
        # println("∃x ∈ U, V(x)≥γ,  γ≥0 = $γ")
        # if found
        #     return x, γ, k
        # end

        x, γ, found = verify_piece(
            wit_cls, piece.flows, piece.rect, lfs, M, N, Θv, Θd, γmax, solver
        )
        if γ > γopt
            xopt = x
            γopt = γ
            kopt = k
        end
    end
    return xopt, γopt, kopt
end


function verifyCandidateCLF(
        counterExamples::Vector{CounterExample},
        hybridSystem::HybridSystem,
        lfs::Vector{<:AbstractVector},
        params::Parameters,
        workspace::Workspace,
        solver
    )

    N = 2

    # TODO: Must check if the current Polyhedron@x intersects with obstacles
    for counterExample in counterExamples
        for lf′ in lfs
            model = solver()
            x = @variable(model, [1:N])
            gap = @variable(model, upper_bound=params.maxLyapunovGapForVerifier)

            @constraint(model, x .≥ workspace.lb[1:N])
            @constraint(model, x .≤ workspace.ub[1:N])

            # Voronoi Constraints s.t. x is contained in the same area as the witness
            for otherCounterExample in filter(x->x!=counterExample, counterExamples)
                diff = (otherCounterExample.x-counterExample.x)[1:N]
                @constraint(model, dot(diff, x) ≤ norm(diff, Inf) / 2)
            end

            # The Last Important Constraint on Lyapnov Function
            A = counterExample.dynamics.A
            b = counterExample.dynamics.b
            for lf in lfs
                x′ = A[1:N,1:N]*x + b[1:N]
                @constraint(model, dot(lf′[1:N], x′) - dot(lf[1:N], x) ≥ gap)
            end

            @objective(model, Max, gap)
            optimize!(model)
            println(value.(x), objective_value(model))

            if value.(gap) ≥ 0
                return value.(x), objective_value(model)
            end

        end
    end

    return [0, 0, 0], -1
end

