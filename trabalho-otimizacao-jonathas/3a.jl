using JuMP, Ipopt
using Random, Statistics

println("### 3ª QUESTÃO LETRA A ###")

function executar_mle(seed::Int, n::Int)
    println("\n========== SEED = $seed | n = $n ==========")
    Random.seed!(seed)
    data = randn(n)  # dados ~ N(0,1)

    # Criar modelo com Ipopt
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

    # Variáveis de decisão
    μ0 = randn()
    σ0 = rand() + 1
    @variable(model, μ, start = μ0)
    @variable(model, σ >= 0.0, start = σ0)

    # Parâmetro: dados fixos
    @NLparameter(model, x[i=1:n] == data[i])

    # Função de log-verossimilhança negativa (para minimizar)
    @NLexpression(model, log_likelihood,
        (n / 2) * log(2π) + n * log(σ) + (1 / (2 * σ^2)) * sum((x[i] - μ)^2 for i in 1:n)
    )

    @NLobjective(model, Min, log_likelihood)

    optimize!(model)

    println("μ (MLE)       = ", round(value(μ), digits=6))
    println("mean(data)    = ", round(mean(data), digits=6))
    println("σ² (MLE)      = ", round(value(σ)^2, digits=6))
    println("var(data)     = ", round(var(data), digits=6))
    println("log L         = ", -objective_value(model))
    println("exp(log L)    = ", exp(-objective_value(model)))
end

# 🔁 Rodar com diferentes seeds e tamanhos de n
seeds = [1234, 4321, 2468]
ns = [1000, 10_000, 100_000]

for n in ns
    for s in seeds
        executar_mle(s, n)
    end
end