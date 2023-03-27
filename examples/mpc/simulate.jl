using SparseArrays, OSQP
using Plots
include("../../src/clfjl.jl")


# Utility function
speye(N) = spdiagm(ones(N))

# Initial and reference states
# X=[x, y, v, θ] , U=[a, ω]
x0 = [0, 0.3, 0, 0]
xr = [1, 0, 0, 0]

# Constraints
xmin = [-2, -0.5, -1.0, -pi]
xmax = [2, 0.5, 1.0, pi]
umin = [-0.1, -0.1]
umax = [0.1, 0.1]
clfjl.simulateMPCSet(Ad, Bd, Q, R, QN, x0, xTmin_, xTmax_, xmin, xmax, umin, umax;
                    numHorizon=10, numStep=100, stopCondition=nothing)
