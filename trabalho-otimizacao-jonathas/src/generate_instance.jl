# src/generate_instance.jl

function generate_instance(m::Int, n::Int)
    facilities = rand(2, n)      # coordenadas (x,y) das facilidades
    clients = rand(2, m)         # coordenadas (x,y) dos clientes

    opening_cost = fill(1.0, n)  # custo fixo, Float64

    # Matriz de custos: distância euclidiana
    allocation_cost = zeros(m, n)
    for i in 1:m
        for j in 1:n
            allocation_cost[i, j] = norm(clients[:, i] - facilities[:, j])
        end
    end

    # Demandas dos clientes (Float64)
    demand = Float64.(rand(1:3, m))

    # Capacidades das facilidades (Float64)
    capacity = Float64.(rand(25:100, n))

    return (m, n, opening_cost, allocation_cost, demand, capacity)
end