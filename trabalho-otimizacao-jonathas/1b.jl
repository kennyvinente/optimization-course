using JuMP
using HiGHS

println("### 1ª QUESTÃO LETRA B ###")

# Criação do modelo com o solver HiGHS
modelo = Model(HiGHS.Optimizer)

# Definição das variáveis de decisão (não negativas)
@variable(modelo, x >= 0)
@variable(modelo, y >= 0)

# Definição da função objetivo: minimizar 12x + 20y
@objective(modelo, Min, 12x + 20y)

# Definição das restrições
@constraint(modelo, restricao_1, 6x + 8y >= 100)
@constraint(modelo, restricao_2, 7x + 12y >= 120)

# Execução do solver para encontrar a solução ótima
optimize!(modelo)

# Verificação do status da solução
println("Status da solução: ", termination_status(modelo))

# Exibição dos resultados
if termination_status(modelo) == MOI.OPTIMAL
    println("Solução ótima encontrada!")
    println("Valor ótimo da função objetivo: ", round(objective_value(modelo), digits=2))
    println("Valor ótimo de x: ", round(value(x), digits=2))
    println("Valor ótimo de y: ", round(value(y), digits=2))
else
    println("Não foi possível encontrar uma solução ótima.")
end