# 6c.jl - Sudoku com menos de 15 valores iniciais (14)

using JuMP
using HiGHS

sudoku = Model(HiGHS.Optimizer)
set_silent(sudoku)

@variable(sudoku, x[1:9, 1:9, 1:9], Bin)

# Restrições
for i in 1:9, j in 1:9
    @constraint(sudoku, sum(x[i, j, k] for k in 1:9) == 1)
end

for i in 1:9, k in 1:9
    @constraint(sudoku, sum(x[i, j, k] for j in 1:9) == 1)
end

for j in 1:9, k in 1:9
    @constraint(sudoku, sum(x[i, j, k] for i in 1:9) == 1)
end

for i in 1:3:7, j in 1:3:7, k in 1:9
    @constraint(sudoku, sum(x[r, c, k] for r in i:i+2, c in j:j+2) == 1)
end

# === Caso com 14 valores iniciais ===
init_sol = [
    5  0  0  0  7  0  0  0  0;
    6  0  0  1  0  5  0  0  0;
    0  9  8  0  0  0  0  6  0;
    8  0  0  0  6  0  0  0  3;
    4  0  0  8  0  3  0  0  1;
    7  0  0  0  2  0  0  0  6;
    0  6  0  0  0  0  2  8  0;
    0  0  0  4  1  9  0  0  5;
    0  0  0  0  8  0  0  7  9
]  # Apenas 14 valores

# Fixa entradas iniciais
for i in 1:9, j in 1:9
    if init_sol[i, j] != 0
        @constraint(sudoku, x[i, j, init_sol[i, j]] == 1)
    end
end

optimize!(sudoku)

# Extrai solução
sol = zeros(Int, 9, 9)
x_val = value.(x)
for i in 1:9, j in 1:9, k in 1:9
    if round(x_val[i, j, k]) == 1
        sol[i, j] = k
    end
end

println("Solução final (14 valores iniciais):")
for i in 1:9
    println(sol[i, :])
end