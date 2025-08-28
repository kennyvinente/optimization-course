using JuMP, HiGHS, CSV, DataFrames

println("### 2ª QUESTÃO LETRA B ###")

# 1. Carregar e pré-processar os dados
function load_and_preprocess_data(csv_path::String)
    data = CSV.read(csv_path, DataFrame, copycols=true)
    for i in 1:nrow(data), j in 2:ncol(data)
        data[i,j] = (data[i,j] == -1 || data[i,j] == 3) ? 1 : 0
    end
    return data
end

# 2. Resolver o modelo de Set Covering
function solve_set_covering(passportdata::DataFrame)
    n = ncol(passportdata) - 1  # Ignorar coluna 'from'
    model = Model(HiGHS.Optimizer)
    
    @variable(model, pass[1:n], Bin)
    @constraint(model, [j in 2:(n+1)], sum(passportdata[i,j] * pass[i] for i in 1:n) >= 1)
    @objective(model, Min, sum(pass))
    
    optimize!(model)
    return model, pass
end

# 3. Exibir resultados formatados
function print_results(model, pass, passportdata)
    println("\n=== RESULTADOS ===")
    println("Status da solução: ", termination_status(model))
    println("Número mínimo de passaportes: ", objective_value(model))
    
    selected = findall(value.(pass) .≈ 1.0)
    countries = names(passportdata)[2:end]
    
    println("\nPaíses selecionados:")
    foreach(idx -> println("- ", countries[idx]), selected)
    println("\nTotal de variáveis: ", length(pass))
    println("Total de restrições: ", ncol(passportdata) - 2)
end

# Execução principal
function main()
    csv_path = "passport-index-matrix.csv"  # Ajuste o caminho conforme necessário
    passportdata = load_and_preprocess_data(csv_path)
    model, pass = solve_set_covering(passportdata)
    print_results(model, pass, passportdata)
end

main()