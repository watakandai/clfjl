using JLD2
@enum StatusCode begin
    CONTROLLER_FOUND = 0
    CONTROLLER_INFEASIBLE = 1
    MAX_ITER_REACHED = 2
end

"Synthesize Polyhedral Control Lyapunov functions through Sampling Counter Examples"
function synthesizeCLF(lines::Vector{Tuple{Vector{Float64}, Vector{Float64}}},
                       params::Parameters,
                       env::Env,
                       solver,
                       sampleFunc,
                       plotFunc=(args...)->nothing,
                       counterExamples::Vector{CounterExample} = CounterExample[])::Tuple{StatusCode, Vector}

    lfs::LyapunovFunctions = []
    for (x0, xT) in lines
        println("Sampling ...")
        sampleFunc(counterExamples, x0, env; xT=xT)
    end

    iter::Integer = 0
    while true

        iter += 1

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
            @save joinpath(params.lfsFileDir, "learnedCLFs.jld2") lfs counterExamples env
            trajectories = simulateWithCLFs(lfs, counterExamples, env; numStep=1000)
            plotTrajectories(trajectories, lfs, env; imgFileDir=params.imgFileDir)
            return CONTROLLER_FOUND, lfs
        end

        params.print && println("Sampling ...")
        sampleFunc(counterExamples, x, env; xT=Vector{Float64}())
    end
    iter = Inf
    plotFunc(iter, counterExamples, env, params, lfs)

    return MAX_ITER_REACHED, []
end

