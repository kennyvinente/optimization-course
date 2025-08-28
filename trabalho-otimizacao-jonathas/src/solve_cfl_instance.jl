# src/solve_cfl_instance.jl

using JuMP, HiGHS

function solve_cfl(m::Int, n::Int, f::Vector{Float64}, c::Matrix{Float64}, a::Vector{Float64}, q::Vector{Float64})
    model = Model(HiGHS.Optimizer)
    set_silent(model)  # Desativa saída do solver (opcional)

    @variable(model, y[1:n], Bin)
    @variable(model, x[1:m, 1:n], Bin)

    # Função objetivo: custo fixo + custo de transporte
    @objective(model, Min, 
        sum(f[j] * y[j] for j in 1:n) + 
        sum(c[i,j] * x[i,j] for i in 1:m, j in 1:n)
    )

    # Cada cliente é atendido por exatamente uma facilidade
    @constraint(model, [i=1:m], sum(x[i,j] for j in 1:n) == 1)

    # Restrição de capacidade: não pode exceder q[j] se y[j] = 1
    @constraint(model, [j=1:n], sum(a[i] * x[i,j] for i in 1:m) <= q[j] * y[j])

    # Resolver
    optimize!(model)

    # Coletar resultados
    status = termination_status(model)
    obj_value = has_values(model) ? objective_value(model) : NaN
    y_val = has_values(model) ? value.(y) : zeros(n)
    n_open = count(>=(0.5), y_val)

    return (status, obj_value, n_open)
end