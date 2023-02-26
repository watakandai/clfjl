function compute_lfs(
        wit_cls::Vector{<:Vector{<:Witness}},
        lfs_init::Vector{<:AbstractVector},
        M, N, Θ, rmax, solver
    )
    N = 2

    model = solver()
    lfs = [
        @variable(model, [1:N], lower_bound=-1, upper_bound=1)
        for wit_cl in wit_cls
    ]
    r = @variable(model, upper_bound=rmax)

    for (i, wit_cl) in enumerate(wit_cls)
        for (j, wit) in enumerate(wit_cl)
            # valx = dot(lfs[i], wit.x)
            valx = dot(lfs[i], wit.x[1:N])
            @constraint(model, valx ≤ 0)
            for q in 1:M
                for img in wit.img_cls[q]
                    α = img.α
                    # do not use Iterators.flatten because type-unstable
                    for lf in lfs
                        # valy = dot(lf, img.y)
                        valy = dot(lf, img.y[1:N])
                        # @constraint(model, valy + r*α ≤ valx)
                        @constraint(model, valy + r ≤ valx)
                        @constraint(model, valy ≤ 0)
                    end
                    for lf in lfs_init
                        # valy = dot(lf, img.y)
                        valy = dot(lf, img.y[1:N])
                        # @constraint(model, valy + r*α ≤ valx)
                        @constraint(model, valy + r ≤ valx)
                        @constraint(model, valy ≤ 0)
                    end
                end
            end
        end
    end

    @objective(model, Max, r)
    optimize!(model)

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    obj = value(r)
    println("Generator: objective=$obj")
    for (lf, wit_cl) in zip(lfs, wit_cls)
        for (j, wit) in enumerate(wit_cl)
            println(map(value, lf), wit.x)
        end
    end
    # display(map(bins_cl -> map(bins -> value.(bins), bins_cl), bins_cls))

    return map(lf -> map(value, lf), lfs), objective_value(model)
end

function generateCandidateCLF(
    counterExamples::Vector{CounterExample}, params::Parameters, N::Int64, solver
    )

    N = 2 # hybridSystem.numDim

    model = solver()
    lfs = [
        @variable(model, [1:N], lower_bound=-1, upper_bound=1)
        for _ in counterExamples
    ]
    gap = @variable(model, upper_bound=params.maxLyapunovGapForGenerator)

    for (i, counterExample) in enumerate(counterExamples)
        # valx = dot(lfs[i], wit.x)
        x = counterExample.x
        α = counterExample.α
        y = counterExample.y

        valx = dot(lfs[i], x[1:N])
        @constraint(model, valx ≤ 0)

        for lf in lfs
            valy = dot(lf, y[1:N])
            @constraint(model, valy ≤ 0)

            @constraint(model, valy + gap ≤ valx)
            # @constraint(model, valy + gap*α ≤ valx)
        end
    end

    @objective(model, Max, gap)
    optimize!(model)

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    obj = value(gap)
    println("Generator: objective=$obj")
    for (lf, counterExample) in zip(lfs, counterExamples)
        println(map(value, lf), ", " , counterExample)
    end

    return map(lf -> map(value, lf), lfs), objective_value(model)
end
