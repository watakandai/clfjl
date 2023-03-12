function generateCandidateCLF(counterExamples::Vector{CounterExample},
                              params::Parameters,
                              env::Env,
                              solver)::Tuple{LyapunovFunctions, Real}

    N = 2 # hybridSystem.numDim

    model = solver()
    λb, lfsBounds = addBoundaryLFs(model, env, N)
    # λo, lfsObstacles = addObstacleLFs(model, env, N)

    lfs = [JuMPLyapunovFunction(@variable(model, [1:N], lower_bound=-1, upper_bound=1),
                                @variable(model)) for _ in counterExamples]
    gap = @variable(model, lower_bound=0,
                           upper_bound=params.maxLyapunovGapForGenerator)

    termLB = env.termSet.lb[1:N]
    termUB = env.termSet.ub[1:N]

    for (i, counterExample) in enumerate(counterExamples)

        lfx = lfs[i]
        x = counterExample.x[1:N]
        y = counterExample.y[1:N]
        # α = counterExample.α

        # For the Unsafe Region V(x)>0
        # if counterExample.isUnsafe
        #     for lf in lfs
        #         @constraint(model, takeImage(lf, x) ≥ 0)
        #     end
        #     continue
        # end

        # if x∈T, then skip (Which would never happen, hopefully ...)
        if all(termLB .< x) && all(x .< termUB)
            # throw(DomainError(x, "A counter example is observed in the terminal state!"))
            continue
        end

        valx = takeImage(lfx, x)
        @constraint(model, valx ≤ 0)

        for lf in lfs
            valy = takeImage(lf, y)
            @constraint(model, valy ≤ 0)
            # -20 + 2 <= -18
            # -20 + 2 <= -10
            @constraint(model, valy + gap ≤ valx)
            # @constraint(model, valy + gap*α ≤ valx)
        end

        # V(y)+gap≤V(x) must also hold for the Lyapunov functions on the boundary
        for i in 1:N^2
            @constraint(model, takeImage(lfsBounds[i], x) ≤ 0)
            @constraint(model, takeImage(lfsBounds[i], y) + gap ≤ valx)
        end

    end

    bounds = map(p -> collect(p), zip(env.initSet.lb, env.initSet.ub))
    for corner in Iterators.product(bounds...)
        x = collect(corner)
        for lf in lfs
            @constraint(model, takeImage(lf, x) <= 0)
        end
    end

    @objective(model, Max, gap)
    optimize!(model)

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    lfs = map(lf -> clfFromJuMP(lf), vcat(lfs, lfsBounds))
    # lfs = map(lf -> clfFromJuMP(lf), vcat(lfs, lfsBounds, lfsObstacles))
    # λo = map(λ -> value.(λ), λo)
    λo = 0

    testCLF(params, counterExamples, value.(λb), λo, lfs, value(gap))

    return lfs, objective_value(model)
end


function addBoundaryLFs(model, env, N)
    """
    Constraint V(x)<=0 in the safe set (but V(x)>0 in the unsafe set).
    We use the lagrange multipliers such that λ(-x+lb)<0, λ(x-ub)<0 holds.
    This can be rewritten in the form a*x+b. e.g.) λ(x-ub) => λ[1, 0] - λ⋅ub
    So we can express it as a=λ[1, 0] & b=λ⋅ub & a*x+b≥0
    """

    λb = [@variable(model, lower_bound=0) for _ in 1:N^2]
    lfsBounds = [JuMPLyapunovFunction(@variable(model, [1:N], lower_bound=-1, upper_bound=1),
                                      @variable(model)) for _ in 1:N^2]

    # For each dimension x, y and ...
    for idim in 1:N

        # Values less than lb λ(- x + lb) > 0
        lb = env.workspace.lb[idim]
        i = 2*(idim-1) + 1  # evaluates to 1 and 3

        a = map(d -> d == idim ? -λb[i] : 0, 1:N)
        b = λb[i] * lb

        @constraint(model, lfsBounds[i].a .== a)
        @constraint(model, lfsBounds[i].b .== b)

        # Values greater than b λ(x - ub) > 0
        ub = env.workspace.ub[idim]
        i = 2*idim # evaluate to 2 and 4

        a = map(d -> d == idim ? λb[i] : 0, 1:N)
        b = -λb[i] * ub

        @constraint(model, lfsBounds[i].a .== a)
        @constraint(model, lfsBounds[i].b .== b)
    end
    return λb, lfsBounds
end


function addObstacleLFs(model, env, N)
    """
    For each Unsafe Obstacle Region Uo
    Enforce a_o^T x + b_o >= 0 ∀x ∈ Uo = { x : Ax ≤ β }
    where A and β are,
        -x[1] <= -o.lb[1]
        x[1] <= o.ub[1]
        -x[2] <= -o.lb[2]
        x[2] <= o.ub[2]

    Thus, A = [-1 0;
                1 0;
                0, -1;
                0 1]
          β = [-o.lb[1], o.ub[1], -o.lb[2], o.ub[2]]

    To enforce a_o^T x + b_o >= 0, we formulate it as an optimization problem
            min a_o^T x
            s.t.
                -Ax ≥ -β
                  x ⩾ 0 <- NOT GREATER THAN EQUAL TO

    Dual problem of this is,
            max -β^T y
            s.t.
                -A^T y = a  <- Equality comes from the face x ⩾ 0
                     y ≥ 0

    If there exists a feasible problem a_o^T x == -β^T y and
    the objective value must be >= -b_o.

    Therefore, we constraint a_o and b_o as follows,
         a_o == -A^T y
        -b_o ≤ -β^T y      -> b_o ≥ β^T y
           y ≥ 0
    we use λ in replacement for y
    """

    λo = [@variable(model, [1:N^2], lower_bound=0) for _ in env.obstacles]
    lfsObstacles = [JuMPLyapunovFunction(@variable(model, [1:N]),
                                         @variable(model)) for _ in env.obstacles]

    for (i, o) in enumerate(env.obstacles)
        A = [-1 0; 1 0; 0 -1; 0 1]
        β = [-o.lb[1], o.ub[1], -o.lb[2], o.ub[2]]
        a = -A' * λo[i]
        b = dot(β', λo[i])
        @constraint(model, lfsObstacles[i].a .== a)
        @constraint(model, lfsObstacles[i].b .== b)
    end
    return λo, lfsObstacles
end


function addInitialSetLF(model, env, N)
    """
    For the initial region I
    Enforce a_o^T x + b_o <= 0 ∀x ∈ I = { x : Ax ≤ β }
    where A and β are,
        -x[1] <= -lb[1]
        x[1] <= ub[1]
        -x[2] <= -lb[2]
        x[2] <= ub[2]

    Thus, A = [-1 0;
                1 0;
                0, -1;
                0 1]
          β = [-lb[1], ub[1], -lb[2], ub[2]]

    -(a_o^T x + b_o) >= 0

    To enforce a_o^T x + b_o <= 0, we formulate it as an optimization problem
            min -a_o^T x
            s.t.
                -Ax ≥ -β
                 x ⩾ 0 <- NOT GREATER THAN EQUAL TO

    Dual problem of this is,
            max -β^T y
            s.t.
                -A^T y = -a  <- Equality comes from the face x ⩾ 0
                    y ≥ 0

    If there exists a feasible problem -a_o^T x == -β^T y and
    the objective value must be <= -b_o.

    Therefore, we constraint a_o and b_o as follows,
         a_o == A^T y
        -b_o >= β^T y      -> b_o ≥ -β^T y
           y >= 0
    we use λ in replacement for y

    y=[y1, y2, y3, y4]^T
    a_o == [[-1, 1, 0, 0],
            [0, 0, -1, 1]] * y
    b_o >= [-lb[1], ub[1], -lb[2], ub[2]] * y

    a_o^T * x + b_0 =
        λ1(-x - lb[1]) + λ2(x + ub[1]) + λ3(-y - lb[2]) + λ4(y + ub[2]) <= 0???
    where λ1, λ2, λ3, λ4 >= 0
    if let x be lb <= x <= ub, e.g.) -1x<=≤1 && x=0, then

    a_o^T * x + b_0 =
        λ1(1) + λ2(x + ub[1]) + λ3(-y - lb[2]) + λ4(y + ub[2]) <= 0???
    """
    λi = @variable(model, [1:N^2], lower_bound=0)
    lfsInit = JuMPLyapunovFunction(@variable(model, [1:N]), @variable(model))

    # A = [-1 0; 1 0; 0 -1; 0 1]
    # β = [-env.initSet.lb[1], env.initSet.ub[1], -env.initSet.lb[2], env.initSet.ub[2]]
    # a = A' * λi
    # b = -dot(β', λi)
    # @constraint(model, lfsInit.a .== a)
    # @constraint(model, lfsInit.b .== b)

    return λi, lfsInit
end


function testCLF(params, counterExamples, λb, λo, lfs, gap)
    N = 2

    infeasible = gap < params.thresholdLyapunovGapForGenerator

    for c in counterExamples

        x = c.x[1:N]
        @assert length(x) == 2
        Vx = V(x, lfs)
        ix = iV(x, lfs)
        ax, bx = lfs[ix].a, lfs[ix].b
        Vx = round(Vx, digits=3)

        y = c.y[1:N]
        Vy = V(y, lfs)
        iy = iV(y, lfs)
        ay, by = lfs[iy].a, lfs[iy].b
        Vy = round(Vy, digits=3)

        if infeasible
            println("x=$x, y=$y \t=>\t Vy<Vx => $Vy + $gap < $Vx")
            println("\t$ay*y + ($by) + $gap < ")
            println("\t$ax*x + ($bx)")
        end
        @assert Vy <= Vx "$Vy <= $Vx, @x=$x and @y=$y"
    end

    @assert all(map(λ -> λ > 0, λb)) λb
    # @assert all(map(λ -> any(map(e -> e > 0, λ)), λo)) λo

    @assert V([-101, -101], lfs) > 0
    @assert V([101, -101], lfs) > 0
    @assert V([101, 101], lfs) > 0
    @assert V([-101, 101], lfs) > 0

    @assert V([-100, -100], lfs) >= -1E-9
    @assert V([100, -100], lfs) >= -1E-9
    @assert V([100, 100], lfs) >= -1E-9
    @assert V([-100, 100], lfs) >= -1E-9
end
