# 4b.jl - Análise do Capacitated Facility Location (CFL)

include("src/generate_instance.jl")
include("src/solve_cfl_instance.jl")

using Printf

# Parâmetros: (número de clientes m, número de facilidades n)
params = [
    (10, 4),
    (20, 5),
    (50, 10),
    (100, 15),
    (200, 30),
    (500, 50)
]

# Cabeçalho da tabela
println("╔════════╦════════╦════════════════════════╦═══════════════╦══════════════╗")
println("║   m    ║   n    ║ Nº Instalações Abertas ║  Custo Total  ║ Tempo (seg)  ║")
println("╠════════╬════════╬════════════════════════╬═══════════════╬══════════════╣")

# Executar para cada combinação
for (m, n) in params
    (m, n, f, c, a, q) = generate_instance(m, n)
    
    t1 = time()
    (status, obj, open) = solve_cfl(m, n, f, c, a, q)
    t2 = time()
    dt = round(t2 - t1, digits=3)

    if status != MOI.OPTIMAL
        @printf("║ %6d ║ %6d ║ %22s ║ %13s ║ %12.3f ║ (status: %s)\n", 
            m, n, "–", "–", dt, status)
    else
        @printf("║ %6d ║ %6d ║ %22d ║ %13.2f ║ %12.3f ║\n", 
            m, n, open, obj, dt)
    end
end

# Rodapé da tabela
println("╚════════╩════════╩════════════════════════╩═══════════════╩══════════════╝")