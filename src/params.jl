struct Params
  order::Vector{Any}
  params::IdSet{Any}
  Params() = new([], IdSet())
end

@forward Params.order Base.iterate, Base.length

function Base.push!(ps::Params, x)
  if !(x in ps.params)
    push!(ps.order, x)
    push!(ps.params, x)
  end
  return ps
end

Base.push!(ps::Params, x...) = (foreach(x -> push!(ps, x), x); ps)

Params(xs) = push!(Params(), xs...)

function Base.show(io::IO, ps::Params)
  print(io, "Params([")
  join(io, ps.order, ", ")
  print(io, "])")
end

struct Grads
  grads::IdDict{Any,Any}
end

Base.show(io::IO, ps::Grads) = println(io, "Grads(...)")

Grads() = Grads(IdDict())

@forward Grads.grads Base.setindex!, Base.haskey, Base.length, Base.iterate

function Grads(ps::Params)
  return Grads(IdDict(tracker(p) => init_grad(data(p)) for p in ps))
end

function Base.getindex(g::Grads, x::Tracked)
  haskey(g.grads, x) || return init_grad(x.grad)
  return g.grads[x]
end

function Base.getindex(g::Grads, x)
  istracked(x) || error("Object not tracked: $x")
  g[tracker(x)]
end

accum!(g::Grads, x, Δ) = g[x] = haskey(g, x) ? g[x] .+ Δ : Δ
