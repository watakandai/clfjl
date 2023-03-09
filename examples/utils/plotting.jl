using LinearAlgebra
using PyCall
const spatial = pyimport_conda("scipy.spatial", "scipy")
const optimize = pyimport_conda("scipy.optimize", "scipy")

function compute_vertices_hrep(lfs, β, lims, N)
    nlf = length(lfs)
    M = fill(NaN, nlf + 2*N, N + 1)
    b = fill(NaN, nlf + 2*N)
    for (i, lf) in enumerate(lfs)
        for k = 1:N
            M[i, k] = float(lf[k])
        end
        M[i, N + 1] = norm(lf)
        b[i] = float(β)
    end
    for ki = 1:N
        for kj = 1:N
            M[nlf + ki*2 - 1, kj] = ki == kj ? 1.0 : 0.0
            M[nlf + ki*2, kj] = ki == kj ? -1.0 : 0.0
        end
        M[nlf + ki*2 - 1, N + 1] = 1.0
        M[nlf + ki*2, N + 1] = 1.0
        b[nlf + ki*2 - 1] = float(lims[2][ki])
        b[nlf + ki*2] = -float(lims[1][ki])
    end
    c_obj = zeros(N + 1)
    c_obj[N + 1] = -1
    bounds = ((nothing, nothing), (nothing, nothing), (nothing, 1))
    res = optimize.linprog(c_obj, A_ub=M, b_ub=b, bounds=bounds)
    @assert res["success"] && res["status"] == 0
    res["fun"] > 0 && return Vector{Float64}[]
    x = res["x"][1:N]
    for i = 1:nlf+2*N
        M[i, N + 1] = -b[i]
    end
    hs = spatial.HalfspaceIntersection(M, x)
    points = collect.(eachrow(hs.intersections))
    ch = spatial.ConvexHull(points)
    return [ch.points[i + 1, :] for i in ch.vertices]
end

function plot_level2D!(
        ax, lfs, β, lims;
        fc="green", fa=0.5, ec="green", ew=1.0
    )
    verts = compute_vertices_hrep(lfs, β, lims, 2)
    isempty(verts) && return
    polylist = matplotlib.collections.PolyCollection((verts,))
    fca = matplotlib.colors.colorConverter.to_rgba(fc, alpha=fa)
    polylist.set_facecolor(fca)
    polylist.set_edgecolor(ec)
    polylist.set_linewidth(ew)
    ax.add_collection(polylist)
end

function plot_point!(ax, point; mc="blue", ms=15)
    ax.plot(point..., marker=".", ms=ms, c=mc)
end