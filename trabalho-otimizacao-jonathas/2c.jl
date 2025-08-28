using JuMP, HiGHS, CSV, DataFrames

println("### 2ª QUESTÃO LETRA C ###")

function load_data(csv_path::String)
    data = CSV.read(csv_path, DataFrame, copycols=true)
    for i in 1:nrow(data), j in 2:ncol(data)
        data[i,j] = (data[i,j] == -1 || data[i,j] == 3) ? 1 : 0
    end
    return data
end

function get_problem_dimensions(data::DataFrame)
    n = ncol(data) - 1  # Ignora coluna 'from'
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, pass[1:n], Bin)
    @constraint(model, [j in 2:(n+1)], sum(data[i,j] * pass[i] for i in 1:n) >= 1)
    @objective(model, Min, sum(pass))

    println("\n=== DIMENSÕES DO PROBLEMA ===")
    println("Número de variáveis (passaportes): ", num_variables(model))
    println("Número de restrições (países): ", n)  # pois é o tamanho de 2:(n+1)

    return model
end

function main()
    data = load_data("passport-index-matrix.csv")
    model = get_problem_dimensions(data)
end

main()
