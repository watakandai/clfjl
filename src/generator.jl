function emptyCLgenerator(counterExamples::Vector{CounterExample},
                          env::Env,
                          solver,
                          N::Integer,
                          maxLyapunovGapForGenerator::Real,
                          thresholdLyapunovGapForGenerator::Real
                          )::Tuple{LyapunovFunctions, Real}
    a = zeros(N)
    b = 0
    lfs = [LyapunovFunction(a, b)]
    positiveGap = 1
    return lfs, positiveGap
end


function generateCandidateCLF(counterExamples::Vector{CounterExample},
                              env::Env,
                              solver,
                              N::Integer,
                              maxLyapunovGapForGenerator::Real,
                              thresholdLyapunovGapForGenerator::Real
                              )::Tuple{LyapunovFunctions, Real}

    model = solver()
    λb, lfsBounds = addBoundaryLFs(model, env, N)
    λo, lfsObstacles = addObstacleLFs(model, env, N)

    ith = length(counterExamples)==0 ? 0 : counterExamples[end].ith
    # lfExamples = [JuMPLyapunovFunction(@variable(model, [1:N], lower_bound=-1, upper_bound=1),
    #                                    @variable(model)) for _ in counterExamples]
    lfExamples = [JuMPLyapunovFunction(@variable(model, [1:N], lower_bound=-1, upper_bound=1),
                                       @variable(model)) for _ in 1:ith]
    gap = @variable(model, lower_bound=0,
                           upper_bound=maxLyapunovGapForGenerator)
    initBounds = map(p -> collect(p), zip(env.initSet.lb, env.initSet.ub))

    lfs = vcat(lfExamples, lfsBounds, lfsObstacles)

    # Initial Set must be V(x)<=0. So we ensure all starting points are V(x)<=0
    for corner in Iterators.product(initBounds...)
        x = collect(corner)
        for lf in lfs
            @constraint(model, takeImage(lf, x) <= 0)
        end
    end

    # Now, we want ot ensure that Lyapunov Functions decreases at each step.
    for counterExample in counterExamples
        ith = counterExample.ith
        lfx = lfs[ith]

        x = counterExample.x
        y = counterExample.y
        α = counterExample.α

        valx = takeImage(lfx, x)

        if counterExample.isUnsafe
            @constraint(model, valx ≥ gap)
        end

        # Recall V(x) := max_j a^T_j x + b_j
        # To ensure V(y) < V(x), there ∃lfx for x V(y, lf) < V(x, lfx) for any lf ∈ lfs
        for lf in lfs
            valy = takeImage(lf, y)
            # -20 + 2 <= -18
            # -20 + 2 <= -10
            if !counterExample.isUnsafe
                # @constraint(model, valy + gap ≤ valx)
                @constraint(model, valy + gap*α ≤ valx)
            end
        end
    end

    println("Before optimize!")
    @objective(model, Max, gap)
    optimize!(model)
    println("After optimize!")

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    lfs = map(lf -> clfFromJuMP(lf), lfs)
    λo = map(λ -> value.(λ), λo)

    if value(gap) < thresholdLyapunovGapForGenerator
        testCLF(counterExamples, lfs, value(gap))
    end

    obj = objective_value(model)

    # gc()
    GC.gc()

    return lfs, obj
end


function addBoundaryLFs(model, env, N)
    """
    Constraint V(x)<=0 in the safe set (but V(x)>0 in the unsafe set).
    We use the lagrange multipliers such that λ(-x+lb)<0, λ(x-ub)<0 holds.
    This can be rewritten in the form a*x+b. e.g.) λ(x-ub) => λ[1, 0] - λ⋅ub
    So we can express it as a=λ[1, 0] & b=λ⋅ub & a*x+b≥0
    """
    λb = [@variable(model, lower_bound=0) for _ in 1:2*N]
    # lfsBounds = [JuMPLyapunovFunction(@variable(model, [1:N], lower_bound=-1, upper_bound=1),
    lfsBounds = [JuMPLyapunovFunction(@variable(model, [1:N]),
                                      @variable(model)) for _ in 1:2*N]

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
    λo = [@variable(model, [1:2*N], lower_bound=0) for _ in env.obstacles]

    lfsObstacles = [JuMPLyapunovFunction(@variable(model, [1:N]),
    # lfsObstacles = [JuMPLyapunovFunction(@variable(model, [1:N], lower_bound=-1, upper_bound=1),
                                         @variable(model)) for _ in env.obstacles]

    for (iObs, o) in enumerate(env.obstacles)
        A = zeros(2*N, N)
        β = zeros(2*N)
        for idim in 1:N
            # lb
            i = 2*(idim-1) + 1
            A[i, idim] = -1
            β[i] = -o.lb[idim]

            # ub
            i = 2*idim
            A[i, idim] = 1
            β[i] = o.ub[idim]
        end

        a = -A' * λo[iObs]
        b = dot(β', λo[iObs])
        @constraint(model, lfsObstacles[iObs].a .== a)
        @constraint(model, lfsObstacles[iObs].b == b)
    end

    # λu = [[@variable(model, lower_bound=0) for _ in lfs] for lfs in unreachableRegions]
    # lfsUnreach = [[JuMPLyapunovFunction(@variable(model, [1:length(lf.a)]),
    #                                     @variable(model)) for lf in lfs]
    #                                                       for lfs in unreachableRegions]

    # # Constraint each unreachable region using duality
    # # Unreachable region is defined as a set of hyperplanes lfs
    # # each hyperplane a^Tx + b <= 0
    # # Thus, in reverse, the safe set is defined as a^Tx + b >= 0
    # # Therefore, to translate into <= expression, we constrain the model to -(a^Tx + b) <= 0
    # for (lfs, jlfs, λs) in zip(unreachableRegions, lfsUnreach, λu)
    #     for (lf, jlf, λ) in zip(lfs, jlfs, λs)
    #         @constraint(model, jlf.a .== -λ * lf.a)
    #         @constraint(model, jlf.b == -λ * lf.b)
    #     end
    # end

    # # Flatten lfsUnreach
    # lfsUnreach = collect(Iterators.flatten(lfsUnreach))

    # return vcat(λo, λu), vcat(lfsObstacles, lfsUnreach)
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
    lfsInit = JuMPLyapunovFunction(@variable(model, [1:N], lower_bound=-1, upper_bound=1),
                                   @variable(model))

    # A = [-1 0; 1 0; 0 -1; 0 1]
    # β = [-env.initSet.lb[1], env.initSet.ub[1], -env.initSet.lb[2], env.initSet.ub[2]]
    # a = A' * λi
    # b = -dot(β', λi)
    # @constraint(model, lfsInit.a .== a)
    # @constraint(model, lfsInit.b .== b)

    return λi, lfsInit
end


function testCLF(counterExamples, lfs, gap)

    for c in counterExamples

        x = c.x
        # @assert length(x) == 2
        Vx = V(x, lfs)
        ix = iV(x, lfs)
        ax, bx = lfs[ix].a, lfs[ix].b
        Vx = round(Vx, digits=4)

        y = c.y
        Vy = V(y, lfs)
        iy = iV(y, lfs)
        ay, by = lfs[iy].a, lfs[iy].b
        Vy = round(Vy, digits=4)

        println("x=$x, y=$y \t=>\t Vy<Vx => $Vy + $gap < $Vx")
        println("\t$ay*y + ($by) + $gap < ")
        println("\t$ax*x + ($bx)")
        @assert Vy <= Vx "$Vy <= $Vx, @x=$x and @y=$y"
    end
end
