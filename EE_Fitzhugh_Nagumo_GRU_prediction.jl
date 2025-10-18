### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# The @bind macro is used for Pluto interactivity. 
# If running outside Pluto, a mock version is defined to avoid errors.
# This allows interactive sliders to work in Pluto notebooks.

import Pkg;

# Package installation
# Pkg.add([
#     "Graphs",
#     "DifferentialEquations",
#     "Plots",
#     "DynamicalSystems",
#     "OrdinaryDiffEq",
#     "GraphPlot",
#     "GraphRecipes",
#     "Statistics",
#     "LinearAlgebra",
#     "Measures",
#     "Random",
#     "Distributions",
#     "StatsBase",
#     "CSV",
#     "DataFrames",
#     "Gnuplot",
#     "Compose",
#     "NetworkLayout",
# ])

# Load all required libraries
using Graphs,
    DifferentialEquations,
    Plots,
    DynamicalSystems,
    OrdinaryDiffEq,
    GraphPlot,
    GraphRecipes,
    Statistics,
    LinearAlgebra,
    Measures,
    Random,
    Distributions,
    StatsBase,
    CSV,
    DataFrames,
    Gnuplot,
    Compose

# Set number of neurons in the network
begin
    n = 90
end

# Set plotting backend
plotly()

# Create a Watts-Strogatz small-world network
begin
    l = watts_strogatz(n, 2, 1, is_directed = false, seed = 3)
    aml = adjacency_matrix(l) # sparse adjacency matrix
    Aml = Matrix(adjacency_matrix(l)) # dense adjacency matrix
end

# Plot the network with circular layout
gp = gplot(l, layout = circular_layout, nodefillc = "black", edgestrokec = colorant"brown")

# Define FitzHugh–Nagumo dynamics function
function lo!(du, u, p, t)
    n, a, b, ϵ, fpa, γ,R = p

    for i = 1:n
        # Membrane potential equation with coupling
        du[1, i] =
            (-u[1, i] * (u[1, i] - a) * (u[1, i] - 1)) - (γ * u[2, i]) +
            (ϵ / (2)) * sum(j -> (Aml[i, j] * ((u[2, j] - u[1, i]))), 1:n)
        # Recovery variable equation
        du[2, i] = b * ((γ * u[1, i]) - fpa * u[2, i])
    end
end

# Set simulation parameters and initial conditions
begin
    tspan = (0.0, 10000.0)
    ϵ = 0.0180 # Coupling Strength
    a = -0.025
    b = 0.0065
    fpa = 0.02
    γ = 1
    R = trunc(Int, n / 2)
    
    p = (n, a, b, ϵ, fpa, γ, R)
    u0 = rand(2, n) # random initial conditions
    prob = ODEProblem(lo!, u0, tspan, p)
    sol = solve(prob) # solve the ODE
end
sol # solution object

# Extract membrane potentials for further analysis
ringer = sol[1, :, :]
u = ringer

# Quick plot of one neuron’s activity
plot(ringer[1,200:end], xlims=(0.0, 2000.0))

# Define lags for synchronization analysis
τ = range(1.0, stop=100.0, length = 100)

# Prepare arrays to store synchronization metrics
sa = zeros(n, length(τ))
ss = zeros(n)
s_t = zeros(length(τ))

# Compute pairwise lagged synchronization between consecutive neurons
for i in 1:n
    for j in 1:length(τ)
        if i < n
            if j == 1
                s_t[j] = sqrt(mean((u[i+1,j:end] .- u[i,1:end]).^2) / 
                             sqrt(mean(u[i,1:end].^2) * mean(u[i+1,1:end].^2)))
            else
                s_t[j] = sqrt(mean((u[i+1,j:end] .- u[i,1:end-round(Int, τ[j-1])]).^2) / 
                             sqrt(mean(u[i,1:end].^2) * mean(u[i+1,1:end].^2)))
            end
        else # wrap-around for last neuron
            if j == 1
                s_t[j] = sqrt(mean((u[1,j:end] .- u[i,1:end]).^2) / 
                             sqrt(mean(u[i,1:end].^2) * mean(u[1,1:end].^2)))
            else
                s_t[j] = sqrt(mean((u[1,j:end] .- u[i,1:end-round(Int, τ[j-1])]).^2) / 
                             sqrt(mean(u[i,1:end].^2) * mean(u[1,1:end].^2)))
            end
        end
    end
    sa[i, :] = s_t # store all lags for neuron i
    ss[i] = minimum(s_t) # store best synchronization measure
end

# Plot synchronization of first neuron over lags
plot(sa[1,:])
# Overall neuron synchronization
plot(ss)
n_a = argmax(ss) # neuron with maximal minimum lagged distance

# Prepare training and testing data for ESN prediction
data = ringer
shift = 100
train_len = 4900
predict_len = 408
training_input = data[:, 1:train_len]
training_target = data[:, 2:train_len+1]
testing_input = data[:, train_len+1:train_len+predict_len]
testing_target = data[:, train_len+2:train_len+predict_len+1]

# Initialize and train Echo State Network (ESN)
res_size = 1000
using Random
using ReservoirComputing
Random.seed!(1234)
esn = ESN(training_input; 
          reservoir = RandSparseReservoir(res_size, radius=0.9, sparsity=6/res_size),
          input_layer = WeightedLayer(),
          reservoir_driver = GRU()
)
training_method = StandardRidge(0.3)
output_layer = train(esn, training_target, training_method) # train ESN
output = esn(Predictive(testing_input), output_layer) # predict future states

# Plot actual vs predicted membrane potential of first neuron
plot([testing_target[1,:] output[1,:]], label=["actual" "predicted"],
    titlefontsize=20,
    legendfontsize=12,
    linewidth=2.5,
    xtickfontsize = 12,
    ytickfontsize = 12,
    size=(1080, 720))

# Heatmaps of actual vs predicted activity across neurons
p1 = heatmap(testing_target', clim = (-0.4, 1.2), xlabel="j", ylabel="t", zlabel="v_j", title="ϵ = 0.0211")
p2 = heatmap(output', clim = (-0.4, 1.2), xlabel="j", ylabel="t", zlabel="v_j", title="ϵ = 0.0211")
plot(p1, p2)

# Scatter plot of membrane potential at a specific time point
plot(sol[1, :, 100], leg=false, mc=:red, ms=5.5, dpi=1500, size=(490, 300), lw=3.5, seriestype=:scatter, framestyle=:box, grid=false)
