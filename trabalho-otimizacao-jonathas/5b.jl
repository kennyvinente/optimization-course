using JuMP
using HiGHS

function solve_uc(g_max::Vector{Int}, g_min::Vector{Int}, c_g::Vector{Int}, c_w::Int, d::Int, w_f::Int)
    n = length(g_max)

    model = Model(HiGHS.Optimizer)

    # Variáveis de decisão
    @variable(model, 0 <= g[i=1:n] <= g_max[i])
    @variable(model, u[i=1:n], Bin)
    @variable(model, 0 <= w <= w_f)

    # Restrições
    @constraint(model, [i=1:n], g[i] >= g_min[i] * u[i])   # Geração mínima se ligado
    @constraint(model, sum(g) + w == d)                    # Atende à demanda

    # Função objetivo
    @objective(model, Min, sum(c_g[i] * g[i] for i in 1:n) + c_w * w)

    # Resolução
    optimize!(model)

    # Resultados
    println("\n--- Resultado: Unit Commitment ---")
    println("Status: ", termination_status(model))
    println("Custo total: ", objective_value(model))

    for i in 1:n
        println("Gerador $i: ", value(g[i]), " MW | Ligado: ", value(u[i]) == 1 ? "Sim" : "Não")
    end
    println("Geração Eólica: ", value(w), " MW")
end

# Exemplo de dados (mesmos do item a)
g_max = [100, 80, 50]
g_min = [20, 20, 10]
c_g = [50, 60, 80]
c_w = 30
d = 130
w_f = 50

solve_uc(g_max, g_min, c_g, c_w, d, w_f)