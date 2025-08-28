using JuMP, HiGHS

function solve_ufl(m::Int, n::Int, f::Vector{Float64}, c::Matrix{Float64})
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    @variable(model, y[1:n], Bin)
    @variable(model, x[1:m, 1:n], Bin)

    @objective(model, Min, sum(f[j] * y[j] for j in 1:n) + sum(c[i,j] * x[i,j] for i in 1:m, j in 1:n))

    @constraint(model, [i=1:m], sum(x[i,j] for j in 1:n) == 1)
    @constraint(model, [i=1:m, j=1:n], x[i,j] <= y[j])

    optimize!(model)

    status = termination_status(model)
    obj_value = objective_value(model)
    y_val = value.(y)
    n_open = count(>=(0.5), y_val)

    return (status, obj_value, n_open)
end