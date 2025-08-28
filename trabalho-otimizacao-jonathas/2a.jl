using JuMP, HiGHS, CSV, DataFrames

println("### 2ª QUESTÃO LETRA A ###")

# Caminho relativo do arquivo CSV (mesma pasta do script)
csv_path = "passport-index-matrix.csv"

# Ler o arquivo CSV
passportdata = CSV.read(csv_path, DataFrame, copycols=true)

# Transformar os dados: -1 e 3 → 1 (sem visto); outros → 0
for i in 1:nrow(passportdata)
    for j in 2:ncol(passportdata)
        if passportdata[i,j] == -1 || passportdata[i,j] == 3
            passportdata[i,j] = 1
        else
            passportdata[i,j] = 0
        end
    end
end

# Número total de países (ignorando a coluna 'from')
n_total = ncol(passportdata) - 1

# Criação do modelo completo
modelo_total = Model(HiGHS.Optimizer)

# Variáveis binárias para cada país
@variable(modelo_total, pass[1:n_total], Bin)

# Restrições: garantia de acesso a todos os países
@constraint(modelo_total, [j = 2:n_total], sum(passportdata[i,j] * pass[i] for i in 1:n_total) >= 1)

# Função objetivo: minimizar número de passaportes
@objective(modelo_total, Min, sum(pass))

# Resolver o problema
optimize!(modelo_total)

# Resultados finais
println("Status da solução (total): ", termination_status(modelo_total))
println("Quantidade mínima de passaportes necessários (total): ", objective_value(modelo_total))

# Países escolhidos (índices e nomes)
selected_indices_total = findall(value.(pass) .== 1)
country_names_total = names(passportdata)[2:end]

print("Países selecionados (total): ")
for idx in selected_indices_total
    print(country_names_total[idx], " ")
end
println("\n")

# Quantidade de variáveis e restrições (total)
println("Quantidade de variáveis (total): ", n_total)
println("Quantidade de restrições (total): ", n_total - 1)
println("-------------------------------")

# ————————————————————————————————
# d) Análise com subconjunto: apenas países que iniciam com 'A' (primeiras 11 linhas e colunas)

# Filtrando os primeiros 11 países
new_passportdata = passportdata[1:11, 1:12]  # 11 linhas + coluna 'from'

# Atualizando número de países no subconjunto
n_small = ncol(new_passportdata) - 1

# Novo modelo para subconjunto
modelo_reduzido = Model(HiGHS.Optimizer)

# Variáveis binárias
@variable(modelo_reduzido, pass[1:n_small], Bin)

# Restrições
@constraint(modelo_reduzido, [j = 2:n_small], sum(new_passportdata[i,j] * pass[i] for i in 1:n_small) >= 1)

# Função objetivo
@objective(modelo_reduzido, Min, sum(pass))

# Resolver
optimize!(modelo_reduzido)

# Resultados do subconjunto
println("\nStatus da solução (subconjunto): ", termination_status(modelo_reduzido))
println("Mínimo de passaportes necessários (subconjunto): ", objective_value(modelo_reduzido))

# Países selecionados no subconjunto
selected_indices_small = findall(value.(pass) .== 1)
country_names_small = names(new_passportdata)[2:end]

print("Países selecionados (subconjunto): ")
for idx in selected_indices_small
    print(country_names_small[idx], " ")
end
println("\n")

# Quantidade de variáveis e restrições (subconjunto)
println("Quantidade de variáveis (subconjunto): ", n_small)
println("Quantidade de restrições (subconjunto): ", n_small - 1)