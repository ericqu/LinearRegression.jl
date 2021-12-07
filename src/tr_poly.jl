
# The content of this file is copied from https://github.com/JuliaStats/StatsModels.jl/blob/master/test/extension.jl (in December 2021)
# The license hereafter is associated with this content

# Copyright (c) 2016: Dave Kleinschmidt.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

poly(x, n) = x^n

abstract type PolyModel end
struct PolyTerm <: AbstractTerm
    term::Symbol
    deg::Int
end
PolyTerm(t::Term, deg::ConstantTerm) = PolyTerm(t.sym, deg.n)

StatsModels.apply_schema(t::FunctionTerm{typeof(poly)}, sch, ::Type{<:PolyModel}) =
    PolyTerm(t.args_parsed...)

StatsModels.modelcols(p::PolyTerm, d::NamedTuple) =
    reduce(hcat, [d[p.term].^n for n in 1:p.deg])
