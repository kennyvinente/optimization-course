using JuMP, HiGHS, CSV, DataFrames

println("### 2ª QUESTÃO LETRA D ###")
println("Subconjunto: países que iniciam com a letra 'A' (11 primeiras linhas e colunas)")

# Caminho do CSV
csv_path = "passport-index-matrix.csv"

# Função para carregar e transformar a matriz
function load_and_filter_data(csv_path::String)
    full_data = CSV.read(csv_path, DataFrame, copycols=true)

    # Selecionar 11 primeiras linhas (países que começam com 'A')
    # E as 11 primeiras colunas associadas a esses países + 'from'
    filtered_data = full_data[1:11, 1:12]  # 11 linhas, 11 países + 'from'

    # Pré-processamento: -1 e 3 indicam acesso sem visto → 1
    for i in 1:nrow(filtered_data), j in 2:ncol(filtered_data)
        filtered_data[i,j] = (filtered_data[i,j] == -1 || filtered_data[i,j] == 3) ? 1 : 0
    end

    return filtered_data
end

# LETRA A: Modelo básico direto
function run_model_a(data::DataFrame)
    println("\n--- Letra D(a): Modelo direto ---")
    n = ncol(data) - 1

    model = Model(HiGHS.Optimizer)
    @variable(model, pass[1:n], Bin)
    @constraint(model, [j = 2:n+1], sum(data[i,j] * pass[i] for i in 1:n) >= 1)
    @objective(model, Min, sum(pass))

    optimize!(model)

    println("Status da solução: ", termination_status(model))
    println("Quantidade mínima de passaportes: ", objective_value(model))

    selected = findall(value.(pass) .== 1)
    names_selected = names(data)[2:end]
    println("Países selecionados: ", join(names_selected[selected], ", "))

    println("Variáveis (passaportes): ", n)
    println("Restrições (países): ", n)  # porque j = 2:n+1 → n valores
end

# LETRA B: Modularizado (com funções separadas)
function solve_set_covering(data::DataFrame)
    n = ncol(data) - 1
    model = Model(HiGHS.Optimizer)

    @variable(model, pass[1:n], Bin)
    @constraint(model, [j = 2:(n+1)], sum(data[i,j] * pass[i] for i in 1:n) >= 1)
    @objective(model, Min, sum(pass))

    optimize!(model)
    return model, pass
end

function print_results(model, pass, data::DataFrame)
    println("\n--- Letra D(b): Solução modularizada ---")
    println("Status da solução: ", termination_status(model))
    println("Número mínimo de passaportes: ", objective_value(model))

    selected = findall(value.(pass) .≈ 1.0)
    country_names = names(data)[2:end]

    println("Países selecionados:")
    for idx in selected
        println("- ", country_names[idx])
    end

    println("Total de variáveis: ", length(pass))
    println("Total de restrições: ", length(country_names))
end

# LETRA C: Somente informações estruturais
function print_problem_dimensions(data::DataFrame)
    println("\n--- Letra D(c): Estrutura do modelo ---")
    n = ncol(data) - 1
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, pass[1:n], Bin)
    @constraint(model, [j = 2:(n+1)], sum(data[i,j] * pass[i] for i in 1:n) >= 1)
    @objective(model, Min, sum(pass))

    println("Número de variáveis (passaportes): ", num_variables(model))
    println("Número de restrições (países): ", n)
end

# Execução principal da letra D
function main()
    data = load_and_filter_data(csv_path)
    run_model_a(data)
    model, pass = solve_set_covering(data)
    print_results(model, pass, data)
    print_problem_dimensions(data)
end

main()