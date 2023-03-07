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

    lfs = [(@variable(model, [1:N], lower_bound=-100, upper_bound=100),
            @variable(model))
          for _ in counterExamples]
    gap = @variable(model, lower_bound=0, upper_bound=params.maxLyapunovGapForGenerator)

    # Terminal Region
    lb = (config["goal"] .- config["goalThreshold"])[1:N]
    ub = (config["goal"] .+ config["goalThreshold"])[1:N]

    for (i, counterExample) in enumerate(counterExamples)

        x = counterExample.x
        α = counterExample.α
        y = counterExample.y

        if counterExample.isUnsafe
            a, b = lfs[i]
            @constraint(model, dot(a, x[1:N]) + b ≥ 0)
            continue
        end

        if all(lb .≤ x[1:N]) && all(x[1:N] .≤ ub)
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
        Vx = round(Vx, digits=3)

        y = c.y
        Vy = f(y[1], y[2])
        iy = indf(y[1], y[2])
        ay, by = lfs[iy]
        Vy = round(Vy, digits=3)

        # println("V(y)<V(x) => V($y)<V($x) \t $Vy + $gap < $Vx")
        # println("$ay*y + ($by) + $gap < $ax*x + ($bx)")
        @assert Vy <= Vx
    end

    # println("λs: ", λs)
    # println("LB: f(-100, -100)=", f(-100, -100))
    # println("RB: f(100, -100)=", f(100, -100))
    # println("RT: f(100, 100)=", f(100, 100))
    # println("LT: f(-100, 100)=", f(-100, 100))
    # println("LB: f(-101, -101)=", f(-101, -101))
    # println("RB: f(101, -101)=", f(101, -101))
    # println("RT: f(101, 101)=", f(101, 101))
    # println("LT: f(-101, 101)=", f(-101, 101))

    @assert length(filter(λ -> λ > 0, λs)) == length(λs)

    # Not always smaller.
    # @assert f(-99, -99) < 0
    # @assert f(99, -99) < 0
    # @assert f(99, 99) < 0
    # @assert f(-99, 99) < 0

    @assert f(-101, -101) > 0
    @assert f(101, -101) > 0
    @assert f(101, 101) > 0
    @assert f(-101, 101) > 0

    @assert f(-100, -100) >= -1E-9
    @assert f(100, -100) >= -1E-9
    @assert f(100, 100) >= -1E-9
    @assert f(-100, 100) >= -1E-9
end
