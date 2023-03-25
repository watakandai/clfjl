using JLD2
@enum StatusCode begin
    CONTROLLER_FOUND = 0
    CONTROLLER_INFEASIBLE = 1
    MAX_ITER_REACHED = 2
end

"Synthesize Polyhedral Control Lyapunov functions through Sampling Counter Examples"
function synthesizeCLF(x0::Vector{<:Real},
                       params::Parameters,
                       env::Env,
                       solver,
                       sampleFunc,
                       plotFunc=(args...)->nothing)::Tuple{StatusCode, Vector}
    counterExamples::Vector{CounterExample} = []
    x = x0
    lfs::LyapunovFunctions = []

    iter::Integer = 0
    while true

        iter += 1
        params.print && println("Sampling ...")
        sampleFunc(counterExamples, x, env)

        params.print && println("Iter: ", iter, " - ncl: ", length(counterExamples))
        if iter > params.maxIteration
            println("Max iter exceeded: ", iter)
            break
        end

        (lfs,
         genLyapunovGap::Real) = generateCandidateCLF(counterExamples,
                                                      env,
                                                      solver,
                                                      params.optDim,
                                                      params.maxLyapunovGapForGenerator,
                                                      params.thresholdLyapunovGapForGenerator)

        println("Plotting ...")
        plotFunc(iter, counterExamples, env, params, lfs)

        params.print && println("|-- Generator Lyapunov Gap: ", genLyapunovGap)
        if genLyapunovGap < params.thresholdLyapunovGapForGenerator
            println("Controller infeasible")
            return CONTROLLER_INFEASIBLE, []
        end

        (x::Vector{<:Real},
         verLyapunovGap::Real) = verifyCandidateCLF(counterExamples,
                                                    lfs,
                                                    env,
                                                    solver,
                                                    params.optDim,
                                                    params.thresholdLyapunovGapForVerifier)

        params.print && println("|-- CE: ", x, ", ", verLyapunovGap)
        if verLyapunovGap < params.thresholdLyapunovGapForVerifier
            println("Valid controller: terminated")
            @save "learnedCLFs" lfs counterExamples env
            return CONTROLLER_FOUND, lfs
        end

    end
    iter = Inf
    plotFunc(iter, counterExamples, env, params, lfs)

    return MAX_ITER_REACHED, []
end

