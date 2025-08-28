# 4c.jl - Análise do benchmark Bilde-Krarup (formato correto)

include("src/solve_ufl_instance.jl")  # Já que é UFL, não CFL

using Printf, HiGHS

"""
    read_ufl_instance(filepath)

Lê uma instância do benchmark Bilde-Krarup no formato descrito:
- Linha 1: FILE: nome
- Linha 2: n m 0
- Próximas n linhas: [id] [f_j] [c_j1] [c_j2] ... [c_jm]
"""
function read_ufl_instance(filepath)
    try
        lines = readlines(filepath)
        idx = 1

        # Pular linha de FILE:
        startswith(strip(lines[idx]), "FILE:") && (idx += 1)

        # Linha 2: n m 0
        parts = split(strip(lines[idx]))
        length(parts) >= 3 || return nothing
        n = parse(Int, parts[1])
        m = parse(Int, parts[2])

        idx += 1

        # Inicializar vetores
        f = zeros(n)           # custos fixos
        c = zeros(m, n)        # custos de conexão: cliente i para facilidade j

        # Próximas n linhas
        for j in 1:n
            idx > length(lines) && return nothing
            data = parse.(Float64, split(strip(lines[idx])))
            idx += 1

            # Primeiro valor: id (pode ser ignorado, assume-se ordem)
            # Segundo: custo fixo da facilidade j
            f[j] = data[2]

            # Restante: custos de conexão com os m clientes
            for i in 1:m
                c[i, j] = data[2 + i]  # índice 3 a m+2
            end
        end

        return m, n, f, c
    catch
        return nothing
    end
end

"""
    solve_and_print(filepath, name)

Resolve e imprime resultado formatado.
"""
function solve_and_print(filepath, name)
    instance = read_ufl_instance(filepath)
    if instance === nothing
        return  # Silently skip
    end

    m, n, f, c = instance
    t1 = time()
    (status, obj, open_count) = solve_ufl(m, n, f, c)
    t2 = time()
    solve_time = round(t2 - t1, digits=3)

    if status != MOI.OPTIMAL
        @printf("║ %-18s ║ %3d ║ %3d ║ %14s ║ %13s ║ %12.3f ║ (status: %s)\n",
            name, m, n, "–", "–", solve_time, status)
    else
        @printf("║ %-18s ║ %3d ║ %3d ║ %14.2f ║ %13d ║ %12.3f ║\n",
            name, m, n, obj, open_count, solve_time)
    end
end

# === Cabeçalho da tabela ===
println("╔════════════════════════╦═════╦═════╦════════════════╦════════════════╦════════════╗")
println("║        Instância         ║  m  ║  n  ║  Custo Total   ║ Nº Abertas     ║ Tempo (s)  ║")
println("╠════════════════════════╬═════╬═════╬════════════════╬════════════════╬════════════╣")

base = "BildeKrarup"

# Função para identificar arquivos de instância (ignorar .opt, .lst)
is_instance(file) = !(endswith(file, ".opt") || endswith(file, ".lst") || occursin("files?", lowercase(file)))

# Processar B e C
for folder in ["B", "C"]
    dir = joinpath(base, folder)
    isdir(dir) || continue
    for file in readdir(dir)
        is_instance(file) || continue
        path = joinpath(dir, file)
        isfile(path) || continue
        solve_and_print(path, "$folder/$file")
    end
end

# Processar Dq e Eq (subpastas 1 a 10)
for outer in ["Dq", "Eq"]
    outer_dir = joinpath(base, outer)
    isdir(outer_dir) || continue
    for sub in 1:10
        dir = joinpath(outer_dir, string(sub))
        isdir(dir) || continue
        for file in readdir(dir)
            is_instance(file) || continue
            path = joinpath(dir, file)
            isfile(path) || continue
            solve_and_print(path, "$outer/$sub/$file")
        end
    end
end

# === Rodapé da tabela ===
println("╚════════════════════════╩═════╩═════╩════════════════╩════════════════╩════════════╝")