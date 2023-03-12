@enum StatusCode begin
    CONTROLLER_FOUND = 0
    CONTROLLER_INFEASIBLE = 1
    MAX_ITER_REACHED = 2
end

"Synthesize Polyhedral Control Lyapunov functions through Sampling Counter Examples"
function synthesizeCLF(params::Parameters,
                       env::Env,
                       solver,
                       callback_fcn::Function=(args...) -> nothing)::Tuple{StatusCode, Vector}

    counterExamples::Vector{CounterExample} = []
    totalSamplingTime = 0.0
    totalPlotTime = 0.0
    x = params.startPoint

    iter::Integer = 0
    while true

        iter += 1
        params.print && println("Sampling ...")
        t = @elapsed sampleTrajectory(counterExamples, x, params, env)
        totalSamplingTime += t

        params.print && println("Iter: ", iter, " - ncl: ", length(counterExamples))
        if iter > params.maxIteration
            println("Max iter exceeded: ", iter)
            break
        end

        (lfs::Vector,
         genLyapunovGap::Real) = generateCandidateCLF(counterExamples,
                                                      params,
                                                      env,
                                                      solver)

        t = @elapsed callback_fcn(iter, counterExamples, env, params, lfs)
        totalPlotTime += t

        params.print && println("|-- Generator Lyapunov Gap: ", genLyapunovGap)
        if genLyapunovGap < params.thresholdLyapunovGapForGenerator
            println("Controller infeasible")
            return CONTROLLER_INFEASIBLE, []
        end

        (x::Vector{Real},
         verLyapunovGap::Real) = verifyCandidateCLF(counterExamples,
                                                    lfs,
                                                    params,
                                                    env,
                                                    solver)

        params.print && println("|-- CE: ", x, ", ", verLyapunovGap)
        if verLyapunovGap < params.thresholdLyapunovGapForGenerator
            println("Valid controller: terminated")
            return CONTROLLER_FOUND, lfs
        end

    end
    return MAX_ITER_REACHED, []
end
