using Match, DelimitedFiles
using YAML


function createFile(filename, header::Vector{String})
    open(filename, "w") do f
        writedlm(f, [header], ",")
    end
end


function addRow(filename, row::AbstractVector)

    if !isfile(filename)
        throw(ArgumentError("File does not exist: $(filename)"))
    end

    rows = readdlm(filename, ',', Any, '\n')
    header = rows[1, :]

    n = length(header)
    if length(row) != n
        throw(ArgumentError("Row length does not match header length: $(length(row)) != $(n)"))
    end

    open(filename, "a+") do f
        writedlm(f, [row], ",")
    end
end


function addRow(filename, row::AbstractVector, header::Vector{String})

    if not isfile(filename)
        createFile(filename, header)
    end

    addRow(filename, row)
end


function selectCounterExampleAttributes(counterExample::CounterExample, selectedAttributes::Vector{String})::Vector{String}
    if isnothing(selectedAttributes)
        selectedAttributes = map(string, fieldnames(typeof(counterExample)))
    end

    function convertToString(counterExample, attr)::Vector{String}
        d = counterExample.dynamics
        @match attr begin
            "x" => map(i->"x$i", 1:length(counterExample.x))
            "α" => ["α"]
            "A" => vec(["A$i$j" for i ∈ 1:size(d.A,1), j ∈ 1:size(d.A,2)])
            "b" => map(i->"b$i", 1:length(d.b))
            "y" => map(i->"y$i", 1:length(counterExample.y))
            "isTerminal" => ["isTerminal"]
            "isUnsafe" => ["isUnsafe"]
            "ith" => ["ith"]
            _ => throw(ArgumentError("Invalid attribute: $(attr)"))
        end
    end

    vecs = selectedAttributes .|> attr -> convertToString(counterExample, attr)
    return collect(Iterators.flatten(vecs))
end


function translateCounterExampleVector(counterExample::CounterExample, selectedAttributes::Vector{String})::Vector{Float64}
   if isnothing(selectedAttributes)
        selectedAttributes = map(string, fieldnames(typeof(counterExample)))
    end

    function convertToNum(counterExample, attr)
        @match attr begin
            "x" => counterExample.x
            "α" => counterExample.α
            "A" => vec(counterExample.dynamics.A)
            "b" => counterExample.dynamics.b
            "y" => counterExample.y
            "isTerminal" => counterExample.isTerminal
            "isUnsafe" => counterExample.isUnsafe
            "ith" => counterExample.ith
            _ => throw(ArgumentError("Invalid attribute: $(attr)"))
        end
    end

    vecs = selectedAttributes .|> attr -> convertToNum(counterExample, attr)
    return collect(Iterators.flatten(vecs))
end



function exportCounterExamples(filename, counterExamples::CounterExamples, selectedAttributes::Vector{String}=nothing)
    header = selectCounterExampleAttributes(counterExamples[1], selectedAttributes)
    createFile(filename, header)
    for ce in counterExamples
        row = translateCounterExampleVector(ce, selectedAttributes)
        addRow(filename, row)
    end
end



function exportEnv(filename::String, env::Env; attrs::Dict=nothing)

    function convertStructToDict(str::Any; attrs::Union{Dict, Vector}=nothing)
        """
        Convert a struct to a Dict.
            str: the struct to be converted
            attr: the attribute to be selected
            attrs: a dictionary of attributes to be converted to a Dict
        """
        if isnothing(attrs)
            attrs = map(string, fieldnames(typeof(str)))
        end

        @match str begin
            x::Number => x
            x::Vector{<:Number} => x
            x::Vector{String} => x
            _ =>
                if isa(attrs, Dict)
                    Dict(string(attr) => convertStructToDict(getproperty(str, attr);
                                        attrs=attrs[string(attr)])
                        for attr in fieldnames(typeof(str)) if string(attr) in keys(attrs))
                elseif isa(attrs, Vector)
                    Dict(string(attr) => convertStructToDict(getproperty(str, attr))
                        for attr in fieldnames(typeof(str)) if string(attr) in attrs)
                else
                    throw(ArgumentError("Invalid attrs: $(attrs)"))
                end
        end
    end

    d = convertStructToDict(env; attrs=attrs)
    YAML.write_file(filename, d)
end
