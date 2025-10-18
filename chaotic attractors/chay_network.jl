# ------------------------------------------------------------
# Load required Julia packages
# ------------------------------------------------------------
using DifferentialEquations, Plots, DynamicalSystems, OrdinaryDiffEq, DataFrames, CSV,
    GraphPlot, Graphs, Statistics, AbstractFFTs

# ------------------------------------------------------------
# Initialize plotting backend (Plotly for interactive plots)
# ------------------------------------------------------------
begin
    plotly()
    GC  # Just a reference to the garbage collector (not necessary here)
end

# ------------------------------------------------------------
# Create a Watts–Strogatz small-world network
# ------------------------------------------------------------
begin
    g = watts_strogatz(100, 2, 0.0, is_directed=false, seed=3)  # 100 nodes, 2 nearest neighbors, no rewiring
    am = adjacency_matrix(g)  # Sparse adjacency matrix
    Am = Matrix(adjacency_matrix(g))  # Convert to dense matrix for calculations
end

# ------------------------------------------------------------
# Visualize the network structure
# ------------------------------------------------------------
gplot(
    g,
    layout=circular_layout,     # Arrange nodes in a circle
    nodefillc="red",            # Node color
    edgestrokec=colorant"blue", # Edge color
)

# ------------------------------------------------------------
# Define the neuron dynamics function (modified Chay-type model)
# dx = f(x, p, t)
# ------------------------------------------------------------
function ls!(dx, x, p, t)
    g1, v1, gkv, vk, gkc, vl, gl, rh, vc, kc, rn = p  # Unpack parameters

    for i = 1:n
        # --------------------------------------------------------
        # Hodgkin–Huxley type gating kinetics for m, h, n
        # --------------------------------------------------------
        alm = (0.1 * (25.0 + x[1, i])) / (1.0 - exp(-(x[1, i] + 25.0) / 10.0))
        βm = 4.0 * exp(-(x[1, i] + 50.0) / 18.0)
        alh = 0.07 * exp(-(x[1, i] + 50.0) / 20.0)
        βh = 1.0 / (1.0 + exp(-(x[1, i] + 20.0) / 10.0))
        aln = 0.01 * (20.0 + x[1, i]) / (1.0 - exp(-(x[1, i] + 20.0) / 10.0))
        βn = 0.125 * exp(-(x[1, i] + 30.0) / 80.0)
        τn = 1.0 / (rn * (aln + βn))

        # Steady-state activation/inactivation values
        mint = alm / (alm + βm)
        nint = aln / (aln + βn)
        hint = alh / (alh + βh)

        # --------------------------------------------------------
        # Network coupling term (diffusive coupling)
        # --------------------------------------------------------
        eps = 0.3 * sum(j -> (Am[i, j] * (x[1, j] - x[1, i])), 1:n)

        # --------------------------------------------------------
        # Differential equations
        # x[1,i] → Membrane potential (V)
        # x[2,i] → Activation variable (n)
        # x[3,i] → Slow variable (e.g., intracellular Ca²⁺)
        # --------------------------------------------------------
        dx[1, i] = g1 * (mint^3) * hint * (v1 - x[1, i]) + 
                   gkv * (x[2, i]^4) * (vk - x[1, i]) + 
                   gkc * (x[3, i] / (1.0 + x[3, i])) * (vk - x[1, i]) + 
                   gl * (vl - x[1, i]) + eps

        dx[2, i] = (nint - x[2, i]) / τn
        dx[3, i] = rh * ((mint^3) * hint * (vc - x[1, i]) - kc * x[3, i])
    end
end

# ------------------------------------------------------------
# Set up and solve the ODE system
# ------------------------------------------------------------
begin
    GC.gc()  # Trigger garbage collection (optional)
    tspan = (0.0, 60.0)  # Time interval
    n = 100  # Number of neurons

    # Model parameters
    g1 = 1750.0
    v1 = 100.0
    gkv = 1700.0
    vk = -75.0
    gkc = 12.0
    vl = -40.0
    gl = 7.0
    rh = 0.27
    vc = 100.0
    kc = 3.3 / 18.0
    rn = 230.0

    p = [g1, v1, gkv, vk, gkc, vl, gl, rh, vc, kc, rn]  # Parameter vector

    # Initial conditions
    x0 = [rand(1, n); 0.01 * ones(1, n); 0.01 * ones(1, n)]

    # Define ODE problem
    lsl = ODEProblem(ls!, x0, tspan, p)

    # Solve using 4th-order Runge–Kutta with dt = 0.01
    sol = solve(lsl, alg=RK4(), dt=0.01)
end

# ------------------------------------------------------------
# Plot behavior of one neuron’s membrane potential over time
# ------------------------------------------------------------
plot(sol[1, 1, 900:end], xlabel="Time", ylabel="Membrane Potential (mV)",
     title="Neuron 1 Membrane Potential Dynamics", lw=2)

# ------------------------------------------------------------
# Extract steady-state behavior matrix (after transients)
# ------------------------------------------------------------
chay = sol[1, :, 1000:end]'

# ------------------------------------------------------------
# Visualize activity across neurons as a heatmap
# ------------------------------------------------------------
begin
    heatmap(
        chay,
        xticks=(0:20:100),
        size=(500, 300),
        clim=(-50, -20),
        color=cgrad(:gist_ncar, rev=true),
        xlabel="Neuron Index",
        ylabel="Time Steps",
        title="Network Voltage Activity Heatmap",
    )
end

# ------------------------------------------------------------
# Plot final network state (voltage at final time point)
# ------------------------------------------------------------
plot(sol[1, :, end],
     seriestype=:scatter,
     leg=false,
     mc=:red,
     ms=5.5,
     lw=3.5,
     dpi=1500,
     size=(490, 300),
     framestyle=:box,
     grid=false,
     xlabel="Neuron Index",
     ylabel="Voltage (mV)",
     title="Final Network State")

# ------------------------------------------------------------
# Print adjacency matrix and coupling terms for debugging
# ------------------------------------------------------------
print(Am)
print(eps)

# ------------------------------------------------------------
# Recompute and print coupling (eps) for each neuron
# ------------------------------------------------------------
n = 100
for i = 1:n
    eps = 0.3 * sum(j -> (Am[i, j] * (x[1, j] - x[1, i])), 1:n)
    print(eps)
end
