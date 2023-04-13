using JLD2
@enum StatusCode begin
    CONTROLLER_FOUND = 0
    CONTROLLER_INFEASIBLE = 1
    MAX_ITER_REACHED = 2
    PROB_SAFE_LYAP_FOUND = 3
end

"Synthesize Polyhedral Control Lyapunov functions through Sampling Counter Examples"
function synthesizeCLF(lines::Vector{Tuple{Vector{Float64}, Vector{Float64}}},
                       params::Parameters,
                       env::Env,
                       solver,
                       sampleFunc;
                       plotFunc=(args...)->nothing,
                       verifyCandidateCLFFunc=verifyCandidateCLF,
                       counterExamples::CounterExamples=CounterExample[])::Tuple{StatusCode, Vector}

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

        println("Simulating with the Voronoi Controller if the candidate Lyapunov is safe ...")
        trajectories = simulateWithCLFs(lfs, counterExamples, env;
                                        numSample=10000, numStep=50, withVoronoiControl=true)
        numSafe = sum([traj.status == SIM_TERMINATED for traj in trajectories])
        allSafe = all([traj.status == SIM_TERMINATED for traj in trajectories])
        clfjl.plotTrajectories(trajectories, lfs, env; imgFileDir=params.imgFileDir, filename="$(iter)withVoronoiControl", numTraj=10)
        params.print && println("|-- LEARNING: Voronoi Controller Safety: ", numSafe / length(trajectories))
        if allSafe && length(counterExamples) > 0
            @save joinpath(params.lfsFileDir, "learnedCLFs.jld2") lfs counterExamples env
            return PROB_SAFE_LYAP_FOUND, lfs
        end

        params.print && println("|-- Generator Lyapunov Gap: ", genLyapunovGap)
        if genLyapunovGap < params.thresholdLyapunovGapForGenerator
            println("Controller infeasible")
            return CONTROLLER_INFEASIBLE, []
        end

        (x::Vector{<:Real},
         verLyapunovGap::Real) = verifyCandidateCLFFunc(counterExamples,
                                                    lfs,
                                                    env,
                                                    solver,
                                                    params.optDim)

        params.print && println("|-- CE: ", x, ", ", verLyapunovGap)
        if verLyapunovGap < params.thresholdLyapunovGapForVerifier
            if length(lines) > 0 || length(counterExamples) > 0
                println("Valid controller: terminated")
                @save joinpath(params.lfsFileDir, "learnedCLFs.jld2") lfs counterExamples env
                trajectories = simulateWithCLFs(lfs, counterExamples, env; numStep=30)
                plotTrajectories(trajectories, lfs, env; imgFileDir=params.imgFileDir)
                return CONTROLLER_FOUND, lfs
            end
        end

        params.print && println("Sampling ...")
        sampleFunc(counterExamples, x, env)
    end
    iter = Inf
    plotFunc(iter, counterExamples, env, params, lfs)

    return MAX_ITER_REACHED, []
end
