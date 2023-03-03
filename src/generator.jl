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
    counterExamples::Vector{CounterExample}, config, params::Parameters, N::Int64, solver
    )

    N = 2 # hybridSystem.numDim

    model = solver()

    # Constraint V(x)<=0 in the safe set (but V(x)>0 in the unsafe set).
    # We use the lagrange multipliers such that λ(-x+lb)<0, λ(x-ub)<0 holds.
    # This can be rewritten in the form a*x+b. e.g.) λ(x-ub) => λ[1, 0] - λ⋅ub
    # So we can express it as a=λ[1, 0] & b=λ⋅ub & a*x+b≥0
    λs = [@variable(model, lower_bound=0) for _ in 1:4]
    # as = [@variable(model, [1:N], lower_bound=-1, upper_bound=1) for _ in 1:4]
    as = [@variable(model, [1:N]) for _ in 1:4]
    bs = [@variable(model) for _ in 1:4]
    bound = [config["xBound"][1],
             config["xBound"][2],
             config["yBound"][1],
             config["yBound"][2]]
    # constraint on x bound
    @constraint(model, as[1] .== [-λs[1], 0])
    @constraint(model, bs[1] .== bound[1] * λs[1])
    @constraint(model, as[2] .== [λs[2], 0])
    @constraint(model, bs[2] .== -bound[2] * λs[2])
    # constraint on y bound
    @constraint(model, as[3] .== [0, -λs[3]])
    @constraint(model, bs[3] .== bound[3] * λs[3])
    @constraint(model, as[4] .== [0, λs[4]])
    @constraint(model, bs[4] .== -bound[4] * λs[4])

    lfs = [(@variable(model, [1:N], lower_bound=-1, upper_bound=1),
            @variable(model))
          for _ in counterExamples]
    gap = @variable(model, lower_bound=0, upper_bound=params.maxLyapunovGapForGenerator)

    for (i, counterExample) in enumerate(counterExamples)

        x = counterExample.x
        α = counterExample.α
        y = counterExample.y

        if counterExample.isUnsafe
            a, b = lfs[i]
            @constraint(model, dot(a, x[1:N]) + b ≥ 0)
            continue
        end

        ax, bx = lfs[i]
        valx = dot(ax, x[1:N]) + bx
        @constraint(model, valx ≤ 0)

        for (ay, by) in lfs
            valy = dot(ay, y[1:N]) + by
            @constraint(model, valy ≤ 0)
            # -20 + 2 <= -18
            # -20 + 2 <= -10
            @constraint(model, valy + gap ≤ valx)
            # @constraint(model, valy + gap*α ≤ valx)
        end

        @constraint(model, dot(as[1], y[1:N]) + bs[1] + gap ≤ valx)
        @constraint(model, dot(as[2], y[1:N]) + bs[2] + gap ≤ valx)
        @constraint(model, dot(as[3], y[1:N]) + bs[3] + gap ≤ valx)
        @constraint(model, dot(as[4], y[1:N]) + bs[4] + gap ≤ valx)
    end

    @objective(model, Max, gap)
    optimize!(model)

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    λs = value.(λs)
    as = map(a -> value.(a), as)
    bs = value.(bs)

    lfs = map(lf -> Tuple([value.(lf[1]), value(lf[2])]), lfs)
    lfs = reduce(push!, zip(as, bs), init=lfs)

    gap = value(gap)

    testCLF(counterExamples, λs, as, bs, lfs, gap)

    return lfs, objective_value(model)
end


function testCLF(counterExamples, λs, as, bs, lfs, gap)

    f(x, y) = maximum(map(lf -> dot(lf[1], [x,y]) + lf[2], lfs))
    indf(x, y) = argmax(map(lf -> dot(lf[1], [x,y]) + lf[2], lfs))

    for c in counterExamples

        x = c.x
        Vx = f(x[1], x[2])
        ix = indf(x[1], x[2])
        ax, bx = lfs[ix]

        y = c.y
        Vy = f(y[1], y[2])
        iy = indf(y[1], y[2])
        ay, by = lfs[iy]

        println("V(y)<V(x) => V($y)<V($x) \t $Vy + $gap < $Vx")
        # println($ay*y + ($by) + $gap < $ax*x + ($bx))
        @assert Vy <= Vx
    end

    X = [[-105, 0], [105, 0], [0, -105], [0, 105]]
    for i in 1:4
        a = as[i]
        b = bs[i]
        Vs = map(x -> dot(a, x)+b, X)
        println("$a * x + $b \t Vs = $Vs")
    end

    @assert length(filter(λ -> λ > 0, λs)) == length(λs)
    @assert f(-105, 0) > 0
    @assert f(105, 0) > 0
    @assert f(0, -105) > 0
    @assert f(0, 105) > 0
end
