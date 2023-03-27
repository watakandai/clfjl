using JLD2
using Suppressor
const GUROBI_ENV = Gurobi.Env()

include("../src/clfjl.jl")

@load  "examples/mpc/learnedCLFs.jld2" lfs counterExamples env


solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
"OutputFlag"=>false))
N = length(env.workspace.lb)

# bestCounterExamplePoint, maxGap = clfjl.verifyCandidateCLF(counterExamples, lfs, env, solver, N, 0.0)

# model = solver()
# x = @variable(model, [1:N])
# gap = @variable(model)

# for lf in lfs
#     @constraint(model, clfjl.takeImage(lf, x) <= gap)
# end
# @objective(model, Min, gap)
# optimize!(model)
# x0 = value.(x)

x0 = env.initSet.ub

@suppress_err begin # Plotting gives warnings, so I added the supress command.
    trajectory = clfjl.simulateWithCLFs(x0, lfs, counterExamples, env; numStep=100, withVoronoiControl=true)
    clfjl.plotTrajectories([trajectory], lfs, env; imgFileDir=pwd(), filename="withVoronoiControl")
    trajectory = clfjl.simulateWithCLFs(x0, lfs, counterExamples, env; numStep=100, withVoronoiControl=false)
    clfjl.plotTrajectories([trajectory], lfs, env; imgFileDir=pwd(), filename="withCLFControl")
end
