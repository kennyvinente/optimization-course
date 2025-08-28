using JuMP
using HiGHS

println("### 1ª QUESTÃO LETRA A ###")

modelo = Model(HiGHS.Optimizer)

@variable(modelo, x >= 0)
@variable(modelo, y >= 0)

@objective(modelo, Min, 12x + 20y)

@constraint(modelo, 6x + 8y >= 100)
@constraint(modelo, 7x + 12y >= 120)

optimize!(modelo)

println("Status: ", JuMP.termination_status(modelo))

if JuMP.termination_status(modelo) == MOI.OPTIMAL
    println("Solução ótima encontrada:")
    println("x = ", round(value(x), digits=2))
    println("y = ", round(value(y), digits=2))
    println("Valor da função objetivo = ", round(objective_value(modelo), digits=2))
else
    println("Não foi possível encontrar uma solução ótima.")
end