
function getSupportingHyperPlane(obstacle::HyperRectangle,
                                 params::Parameters,
                                 env::Env,
                                 solver,
                                 ϵ,
                                 separateInit::Bool=true)::LyapunovFunctions

    Nd = params.optDim
    Ns = env.numSpaceDim
    model = solver()

    xT = @variable(model, [1:Nd])
    xI = @variable(model, [1:Nd])
    xO = @variable(model, [1:Nd])
    lf = JuMPLyapunovFunction(@variable(model, [1:Nd]),
                              @variable(model))

    termBounds = map(p -> collect(p), zip(env.termSet.lb, env.termSet.ub))
    initBounds = map(p -> collect(p), zip(env.initSet.lb, env.initSet.ub))
    obsBounds = map(p -> collect(p), zip(obstacle.lb, obstacle.ub))

    termCorners = vec(collect.(Iterators.product(termBounds...)))
    initCorners = vec(collect.(Iterators.product(initBounds...)))
    obsCorners = vec(collect.(Iterators.product(obsBounds...)))

    # Initial Set must be V(x)<=0. So we ensure all starting points are V(x)<=0
    termVal::Integer = Int(!separateInit)
    initVal::Integer = Int(separateInit)

    for xT in termCorners
        # @constraint(model, takeImage(lf, xT) ≥ termVal)
        @constraint(model, takeImage(lf, xT) ≥ 1)
    end
    for xI in initCorners
        # @constraint(model, takeImage(lf, xI) ≥ initVal)
        @constraint(model, takeImage(lf, xI) ≥ 1)
    end
    for xO in obsCorners
        @constraint(model, -1 * takeImage(lf, xO) ≥ 1)
   end

    # Solve the feasibility problem
    @objective(model, Min, 0)

    optimize!(model)

    if primal_status(model) == FEASIBLE_POINT
        return [clfFromJuMP(lf)]
    else
        throw(DomainError(model, "Could not find a supporting hyperplane"))
    end
end


function getHyperPlanesExcludingObstacles(obstacle::HyperRectangle,
                                          params::Parameters,
                                          env::Env,
                                          solver)::LyapunovFunctions
    """
    There must be a (or multiple) hyperplanes that separate O from T.
    Thus, these following conditions must satisfy:
        ∀x ∈ O, f(x) = a^T⋅x + β ≤ 0
        ∀x ∈ T, f(x) = a^T⋅x + β > 0
    For example, consider one face of the obstacle, upper bound for x[1].
        if x1 ≤ ub, then f(x) ≤ 0
        if x1 > ub, then f(x) > 0
    in other words,
        (x - ub) ≤ for x ∈ O
        (x - ub) > for x ∈ T

    For such a hyperplane, we wish to find if following conditions hold,
        ∀x ∈ O, f(x) ≤ 0
        ∀x ∈ T, f(x) > 0

    If so, we keep it, else we ignore.
    We check for four sides ...
        (-x + lb) > 0
        (x - ub) > 0
        (-y + lb) > 0
        (y - ub) > 0
    """
    N = 2

    A = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    β = [obstacle.lb[1], -obstacle.ub[1], obstacle.lb[2], -obstacle.ub[2]]

    termBounds = map(p -> collect(p), zip(env.termSet.lb, env.termSet.ub))
    termCorners = vec(collect.(Iterators.product(termBounds...)))

    feasibleLfs = []

    for i in 1:N^2
        model = solver()
        lf = JuMPLyapunovFunction(@variable(model, [1:N], lower_bound=-1, upper_bound=1),
                                  @variable(model))

        # Constraint 1: f(x) > 0, ∀x ∈ O
        @constraint(model, lf.a .== A[i])
        @constraint(model, lf.b .== β[i])

        # Constraint 2: f(x) < 0, ∀x ∈ T
        for x in termCorners
            @constraint(model, takeImage(lf, x) ≥ 0)
        end

        # Solve the feasibility problem
        @objective(model, Min, 0)
        optimize!(model)
        if primal_status(model) == FEASIBLE_POINT
            push!(feasibleLfs, lf)
        end
    end
    return map(lf -> clfFromJuMP(lf), feasibleLfs)
end


function getUnreachableRegion(obstacle::HyperRectangle,
                              params::Parameters,
                              env::Env,
                              solver,
                              ϵ)::LyapunovFunctions
    return vcat(getSupportingHyperPlane(obstacle, params, env, solver, false))
end

"Get Hyperplans that separate obstacles from initial and terminal sets"
function getUnreachableRegions(params::Parameters,
                               env::Env,
                               solver,
                               ϵ)::Vector{LyapunovFunctions}
    """
    Hyperplanes Ax + b separates obstacles from initial and terminal sets.
        ∀x ∈ O, Ax ≥ b
        ∀x ∈ I ∩ T, Ax < b
    """
    return map(o -> getUnreachableRegion(o, params, env, solver, ϵ), env.obstacles)
end

# First, identify unreachable regions
# regions = clfjl.getUnreachableRegions(params,
#                                       env,
#                                       solver,
#                                       0.0)
# for lfs in regions
#     A = map(lf->round.(lf.a[1:env.numSpaceDim], digits=2), lfs) #vec{vec}
#     A = reduce(hcat, A)' # matrix
#     b = map(lf->round(lf.b, digits=2), lfs)
#     convexObstacleDict = Dict("type" => "Convex", "A" => A, "b" => b)
#     push!(params.config["obstacles"], convexObstacleDict)
# end

