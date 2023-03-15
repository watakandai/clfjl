@enum StatusCode begin
    CONTROLLER_FOUND = 0
    CONTROLLER_INFEASIBLE = 1
    MAX_ITER_REACHED = 2
end

"Synthesize Polyhedral Control Lyapunov functions through Sampling Counter Examples"
function synthesizeCLF(params::Parameters,
                       env::Env,
                       solver)::Tuple{StatusCode, Vector}
    # First, identify unreachable regions
    regions = clfjl.getUnreachableRegions(params, env, solver)
    for lfs in regions
        A = map(lf->round.(lf.a, digits=2), lfs) #vec{vec}
        A = reduce(hcat, A)' # matrix
        b = map(lf->round(lf.b, digits=2), lfs)
        convexObstacleDict = Dict("type" => "Convex", "A" => A, "b" => b)
        push!(params.config["obstacles"], convexObstacleDict)
    end

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

        t = @elapsed plotCLF(iter, counterExamples, regions, env, params, lfs)
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
        if verLyapunovGap < params.thresholdLyapunovGapForVerifier
            println("Valid controller: terminated")
            return CONTROLLER_FOUND, lfs
        end

    end
    return MAX_ITER_REACHED, []
end
