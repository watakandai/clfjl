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

function learn_controller(
        execPath, config,
        Dpieces::Vector{<:Piece},
        lfs_init::Vector{<:AbstractVector},
        M, N, xmax, iter_max, solver;
        γmax=2, rmax=2, Θg=4, tol_r=1e-5, tol_γ=-1e-5,
        do_print=true, callback_fcn=(args...) -> nothing
    )

    lfs_init_f = map(lf -> Float64.(lf), lfs_init)
    wit_cls = Vector{WT_}[]

    Dpieces = map(
        piece -> Piece(
            map(flow -> Flow(Float64.(flow.A), Float64.(flow.b)), piece.flows),
            Rectangle(Float64.(piece.rect.lb), Float64.(piece.rect.ub))
        ), Dpieces
    )
    nDpieces = map(
        piece -> map(flow -> opnorm(flow.A, 1), piece.flows), Dpieces
    )

    Θv = N*xmax
    Θd = 2*γmax

    x = config["start"]
    sample(wit_cls, x, Dpieces, nDpieces, M, execPath, config)

    iter = 0
    while true
        iter += 1
        do_print && println("Iter: ", iter, " - ncl: ", length(wit_cls))
        if iter > iter_max
            println("Max iter exceeded: ", iter)
            break
        end

        lfs::Vector{VT_}, r::Float64 = compute_lfs(
            wit_cls, lfs_init_f, M, N, Θg, rmax, solver
        )

        do_print && println("|-- r generator: ", r)

        if r < tol_r
            println("Controller infeasible")
            return CONTROLLER_INFEASIBLE, lfs_init_f
        end

        append!(lfs, lfs_init_f)

        x::VT_, γ::Float64, kopt::Int = verify(
            wit_cls, Dpieces, lfs, M, N, Θv, Θd, γmax, solver)

        @assert norm(x, Inf) ≤ xmax

        do_print && println("|-- CE: ", x, ", ", γ)

        callback_fcn(iter, wit_cls, lfs, x, kopt)

        if γ ≤ tol_γ
            println("Valid controller: terminated")
            return CONTROLLER_FOUND, lfs
        end

        sample(wit_cls, x, Dpieces, nDpieces, M, execPath, config)

        # do_print && println("|-- Witnesses: ", map(wit_cl -> map(w -> w.x, wit_cl), wit_cls))
    end
    return MAX_ITER_REACHED, lfs_init_f
end

