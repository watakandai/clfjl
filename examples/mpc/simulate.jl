using SparseArrays, OSQP
using Plots

# Utility function
speye(N) = spdiagm(ones(N))

"Solve Linear MPC Problem. State X=[x, y, θ], U=[a, ω]"
function solve(x0, xr, xmin, xmax, umin, umax; numHorizon=10, numStep=100)
    """
    Arguments
    ---------
    numHorizon: Prediction horizon
    v: Car Velcoity
    θ: The angle the car can take at each step
    ω: Angular Velocity

    Dynamics
        x' = x + v
        y' = y + vθ
        v' = v + a
        θ' = θ + ω

    State
        x ∈ [-1, 1]
        y ∈ [-0.2, 0.2]
        θ ∈ [-π, π]
    Input:
        a ∈ [-0.1, 0.1]
        ω ∈ [-0.1, 0.1]
    """
    N = numHorizon

    # Discrete time model of a quadcopter
    Ad = [1 0 1 0;
          0 1 1/2 1/2;
          0 0 1 0;
          0 0 0 1;] |> sparse
    Bd = [0 0;
          0 0;
          1 0;
          0 1] |> sparse
    (nx, nu) = size(Bd)

    # Objective function
    Q = spdiagm(ones(nx))           # Weights for Xs from 0:N-1
    QN = Q                          # Weights for the terminal state X at N (Xn or xr)
    R = 0.1 * speye(nu)

    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    """
    Minimize
        (xN-xr)^T QN (xN-xr) + ∑{k=0:N-1} (xk-xr)^T Q (xk-xr) + uk^T R uk
    s.t.
        x{k+1} = A xk + B uk
        xmin ≤ xk ≤ xmax
        umin ≤ uk ≤ umax
        x0 = \bar{x} (Initial state)
    Refer to https://osqp.org/docs/examples/mpc.html
    """
    # - quadratic objective
    P = blockdiag(kron(speye(N), Q), QN, kron(speye(N), R))
    # - linear objective
    q = [repeat(-Q * xr, N); -QN * xr; zeros(N*nu)]
    # - linear dynamics
    Ax = kron(speye(N + 1), -speye(nx)) + kron(spdiagm(-1 => ones(N)), Ad)
    Bu = kron([spzeros(1, N); speye(N)], Bd)
    Aeq = [Ax Bu]
    leq = [-x0; zeros(N * nx)]
    ueq = leq
    # - input and state constraints
    Aineq = speye((N + 1) * nx + N * nu)
    lineq = [repeat(xmin, N + 1); repeat(umin, N)]
    uineq = [repeat(xmax, N + 1); repeat(umax, N)]
    # - OSQP constraints
    A, l, u = [Aeq; Aineq], [leq; lineq], [ueq; uineq]

    # Create an OSQP model
    m = OSQP.Model()

    # Setup workspace
    OSQP.setup!(m; P=P, q=q, A=A, l=l, u=u, warm_start=true)

    # Simulate in closed loop
    X::Vector{Vector{<:Real}} = [x0]
    U = []
    @time for _ in 1 : numStep
        # Solve
        res = OSQP.solve!(m)

        # Check solver status
        if res.info.status != :Solved
            error("OSQP did not solve the problem!")
        end

        # Apply first control input to the plant
        ctrl = res.x[(N+1)*nx+1:(N+1)*nx+nu]
        push!(U, ctrl)
        println(typeof(Ad))
        println(typeof(x0))
        println(typeof(Bd))
        println(typeof(ctrl))
        xNext = Ad * x0 + Bd * ctrl
        push!(X, xNext)
        x0 = xNext
        @assert length(xNext) == length(xr)

        # Update initial state
        l[1:nx], u[1:nx] = -x0, -x0
        OSQP.update!(m; l=l, u=u)
    end

    Xs = [x[1] for x in X]
    Ys = [x[2] for x in X]
    Θs = [x[3] for x in X]
    plot(Xs, Ys, aspect_ratio=:equal, markerstyle=:dot)
    savefig(joinpath(@__DIR__, "img.png"))
    return X, U
end

# Initial and reference states
# X=[x, y, v, θ] , U=[a, ω]
x0 = [0, 0.3, 0, 0]
xr = [1, 0, 0, 0]

# Constraints
xmin = [-2, -0.5, -1.0, -pi]
xmax = [2, 0.5, 1.0, pi]
umin = [-0.1, -0.1]
umax = [0.1, 0.1]
solve(x0, xr, xmin, xmax, umin, umax; numHorizon=10, numStep=100)
