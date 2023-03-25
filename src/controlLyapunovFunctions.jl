
function clfFromReal(a, b)::LyapunovFunction
    return LyapunovFunction(a, b)
end


function clfFromJuMP(a::Vector{VariableRef}, b::VariableRef)::LyapunovFunction
    return LyapunovFunction(value.(a), value(b))
end


function clfFromJuMP(lf::Vector{VariableRef})::LyapunovFunction
    return clfFromJuMP(lf...)
end


function clfFromJuMP(lf::JuMPLyapunovFunction)::LyapunovFunction
    return clfFromJuMP(lf.a, lf.b)
end


function takeImage(lf::Union{LyapunovFunction, JuMPLyapunovFunction},
                   x::AbstractVector)::Union{Real, AffExpr}
    @assert length(x) == length(lf.a) "$x, $(lf.a)"
    return dot(lf.a, x) + lf.b
end


function V(x::Vector{<:Real}, lfs::LyapunovFunctions)::Real
    return maximum(map(lf -> takeImage(lf, x), lfs))
end


function iV(x::Vector{<:Real}, lfs::LyapunovFunctions)::Real
    return argmax(map(lf -> takeImage(lf, x), lfs))
end

