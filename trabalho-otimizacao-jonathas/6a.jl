# 6a.jl - Sudoku com menos de 25 valores iniciais (24)

using JuMP
using HiGHS

# Criação do modelo
sudoku = Model(HiGHS.Optimizer)
set_silent(sudoku)

# Variável: x[i,j,k] = 1 se a célula (i,j) tem o número k
@variable(sudoku, x[1:9, 1:9, 1:9], Bin)

# Restrição: uma única entrada por célula
for i in 1:9, j in 1:9
    @constraint(sudoku, sum(x[i, j, k] for k in 1:9) == 1)
end

# Restrição: cada número aparece uma vez por linha
for i in 1:9, k in 1:9
    @constraint(sudoku, sum(x[i, j, k] for j in 1:9) == 1)
end

# Restrição: cada número aparece uma vez por coluna
for j in 1:9, k in 1:9
    @constraint(sudoku, sum(x[i, j, k] for i in 1:9) == 1)
end

# Restrição: cada número aparece uma vez por subgrade 3x3
for i in 1:3:7, j in 1:3:7, k in 1:9
    @constraint(sudoku, sum(x[r, c, k] for r in i:i+2, c in j:j+2) == 1)
end

# === Caso com 24 valores iniciais ===
init_sol = [
    5  3  0  0  7  0  0  0  0;
    6  0  0  1  9  5  0  0  0;
    0  9  8  0  0  0  0  6  0;
    8  0  0  0  6  0  0  0  3;
    4  0  0  8  0  3  0  0  1;
    7  0  0  0  2  0  0  0  6;
    0  6  0  0  0  0  2  8  0;
    0  0  0  4  1  9  0  0  5;
    0  0  0  0  8  0  0  7  9
]  # 24 valores não nulos

# Adiciona as entradas fixas
for i in 1:9, j in 1:9
    if init_sol[i, j] != 0
        @constraint(sudoku, x[i, j, init_sol[i, j]] == 1)
    end
end

# Resolver
optimize!(sudoku)

# Extrair solução
sol = zeros(Int, 9, 9)
x_val = value.(x)
for i in 1:9, j in 1:9, k in 1:9
    if round(x_val[i, j, k]) == 1
        sol[i, j] = k
    end
end

# Exibir solução
println("Solução final (24 valores iniciais):")
for i in 1:9
    println(sol[i, :])
end