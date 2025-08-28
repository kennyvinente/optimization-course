using JuMP
using HiGHS

function solve_ed(g_max::Vector{Int}, g_min::Vector{Int}, c_g::Vector{Int}, c_w::Int, d::Int, w_f::Int)
    model = Model(HiGHS.Optimizer)

    n = length(g_max)

    @variable(model, g_min[i] <= g[i=1:n] <= g_max[i])
    @variable(model, 0 <= w <= w_f)

    # Demanda deve ser satisfeita: soma da geração + água = demanda
    @constraint(model, sum(g) + w == d)

    # Função objetivo: minimizar custo total de geração térmica e uso de água
    @objective(model, Min, sum(c_g[i] * g[i] for i in 1:n) + c_w * w)

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        println("Custo total ótimo: ", objective_value(model))
        for i in 1:n
            println("Geração térmica da usina ", i, ": ", value(g[i]), " MW")
        end
        println("Uso da água: ", value(w), " MW")
    else
        println("Solução não ótima ou problema infeasível.")
    end
end

# Exemplo de chamada
g_max = [200, 150, 180]   # MW
g_min = [50, 30, 60]      # MW
c_g = [100, 150, 120]     # R$/MWh
c_w = 30                  # R$/MWh (custo da água)
d = 400                   # Demanda total (MW)
w_f = 100                 # Limite de uso da água (MW)

solve_ed(g_max, g_min, c_g, c_w, d, w_f)