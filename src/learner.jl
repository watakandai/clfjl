@enum StatusCode begin
    CONTROLLER_FOUND = 0
    CONTROLLER_INFEASIBLE = 1
    MAX_ITER_REACHED = 2
end

const VT_ = Vector{Float64}
const IT_ = Image{Float64,VT_,Flow{Matrix{Float64}, Vector{Float64}}}
const WT_ = Witness{VT_,Vector{Vector{IT_}}}


function synthesizeCLF(
    config,
    params,
    hybridSystem,
    workspace,
    solver,
    filedir,
    callback_fcn=(args...) -> nothing,
    callback_fcn2=(args...) -> nothing)

    counterExamples::Vector{CounterExample} = []
    # Start with either Sampling or Generate a candidate Lyapunov function from workspace bounds
    sampleTrajectory(counterExamples, params.startPoint, config, params, hybridSystem, workspace)

    iter = 0
    while true

        iter += 1
        params.do_print && println("Iter: ", iter, " - ncl: ", length(counterExamples))
        if iter > params.maxIteration
            println("Max iter exceeded: ", iter)
            break
        end

        lfs, genLyapunovGap = generateCandidateCLF(counterExamples, config, params, hybridSystem.numDim, solver)
        params.do_print && println("|-- Generator Lyapunov Gap: ", genLyapunovGap)
        if genLyapunovGap < params.thresholdLyapunovGapForGenerator
            println("Controller infeasible")
            return CONTROLLER_INFEASIBLE, []
        end

        x, verLyapunovGap = verifyCandidateCLF(
            counterExamples::Vector{CounterExample},
            hybridSystem::HybridSystem,
            lfs::Vector{Tuple{Vector{Float64}, Float64}},
            params::Parameters,
            workspace::Workspace,
            solver
        )

        callback_fcn(iter, config, counterExamples, hybridSystem, lfs, x, "", filedir)

        # @assert norm(x, Inf) ≤ params.maxXNorm

        params.do_print && println("|-- CE: ", x, ", ", verLyapunovGap)
        # if verLyapunovGap ≤ params.maxLyapunovGapForVerifier
        if verLyapunovGap < params.thresholdLyapunovGapForVerifier
            println("Valid controller: terminated")
            return CONTROLLER_FOUND, lfs
        end

        sampleTrajectory(counterExamples, x, config, params, hybridSystem, workspace)

        callback_fcn2(iter, config, counterExamples, hybridSystem, lfs, x, "afterSampling", filedir)
    end
    return MAX_ITER_REACHED, lfs_init_f
end
