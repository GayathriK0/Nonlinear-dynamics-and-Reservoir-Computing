# -------------------------------------------------------------
# Echo State Network (ESN) demo on Mackey–Glass chaotic series
# Original source: https://mantas.info/code/simple_esn/
# Cleaned & documented version by Gayathri K.
# -------------------------------------------------------------

using DelimitedFiles, Random, Plots, LinearAlgebra

# -------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------
data_path = "path\to\MackeyGlass_t17.txt"
data = readdlm(data_path)

train_len = 5000
test_len = 2000
init_len = 100

# Plot a sample of the series
plot(data[1:1000], title = "Mackey–Glass time series (sample)", legend = false)

# -------------------------------------------------------------
# 2. Initialize ESN parameters
# -------------------------------------------------------------
in_size  = 1
out_size = 1
res_size = 1000
leak_rate = 0.3               # Leaking rate (a)

Random.seed!(42)

# Input weight matrix (including bias term)
Win = (rand(res_size, 1 + in_size) .- 0.5)

# Internal reservoir weight matrix
W = rand(res_size, res_size) .- 0.5

# Normalize to desired spectral radius
println("Computing spectral radius...")
ρW = maximum(abs.(eigvals(W)))
W .*= (1.25 / ρW)
println("Spectral radius scaled to 1.25")

# -------------------------------------------------------------
# 3. Run the reservoir on training data
# -------------------------------------------------------------
X = zeros(1 + in_size + res_size, train_len - init_len)  # Collected reservoir states
Y_target = transpose(data[init_len + 2 : train_len + 1]) # Target outputs

x = zeros(res_size, 1)  # Initial reservoir state

for t in 1:train_len
    u = data[t]
    x = (1 - leak_rate) .* x .+ leak_rate .* tanh.(Win * [1; u] .+ W * x)
    if t > init_len
        X[:, t - init_len] = [1; u; x]
    end
end

# -------------------------------------------------------------
# 4. Train output weights (ridge regression)
# -------------------------------------------------------------
λ = 1e-8  # Regularization coefficient
Wout = transpose((X * transpose(X) + λ * I) \ (X * transpose(Y_target)))

# -------------------------------------------------------------
# 5. Test the trained ESN (generative mode)
# -------------------------------------------------------------
Y = zeros(out_size, test_len)
u = data[train_len + 1]

for t in 1:test_len
    x = (1 - leak_rate) .* x .+ leak_rate .* tanh.(Win * [1; u] .+ W * x)
    y = Wout * [1; u; x]
    Y[:, t] = y
    u = y  # Generative mode (use output as next input)
end

# -------------------------------------------------------------
# 6. Compute mean squared error
# -------------------------------------------------------------
error_len = 500
mse = mean((data[train_len + 2 : train_len + error_len + 1] .- Y[1, 1:error_len]).^2)
println("MSE = $mse")

# -------------------------------------------------------------
# 7. Plot results
# -------------------------------------------------------------
p1 = plot(data[train_len + 2 : train_len + test_len + 1],
          label = "Target signal", color = RGB(0, 0.75, 0))
plot!(transpose(Y), label = "Predicted signal", color = :blue)
title!("Mackey–Glass: target vs. generated ESN output")

p2 = plot(transpose(X[1:20, 1:200]), legend = false, title = "Sample reservoir activations")
p3 = bar(transpose(Wout), legend = false, title = "Output weights Wout")

plot(p1, p2, p3, layout = (3, 1), size = (900, 800))
