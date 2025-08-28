include("src/generate_instance.jl")
include("src/solve_ufl_instance.jl")

using Printf, Dates

# Parâmetros a testar
params = [
    (10, 4),
    (20, 5),
    (50, 10),
    (100, 15),
    (200, 30),
    (500, 50)
]

println("╔════════╦════════╦════════════════════════╦═══════════════╦══════════════╗")
println("║   m    ║   n    ║ Nº Instalações Abertas ║  Custo Total  ║ Tempo (seg)  ║")
println("╠════════╬════════╬════════════════════════╬═══════════════╬══════════════╣")

for (m, n) in params
    (m, n, f, c) = generate_instance(m, n)
    t1 = time()
    (status, obj, open) = solve_ufl(m, n, f, c)
    t2 = time()
    dt = round(t2 - t1, digits=3)
    @printf("║ %6d ║ %6d ║ %22d ║ %13.2f ║ %12.3f ║\n", m, n, open, obj, dt)
end

println("╚════════╩════════╩════════════════════════╩═══════════════╩══════════════╝")