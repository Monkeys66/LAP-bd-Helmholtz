using Gridap
using GridapMakie, CairoMakie
using HelmholtzLAP

fem = HelmholtzLAP.assemble_fem(joinpath(@__DIR__, "geo-rectangle.msh"))

eigenpairs = HelmholtzLAP.compute_eigenpairs(fem; nev=4)
show_eigenvalues(eigenpairs)

indice = HelmholtzLAP.find_resonant_indices(eigenpairs, 3)

eigenvalue = eigenpairs.λ[indice]
Φ = eigenpairs.Φ[:, indice]

k = sqrt(eigenvalue[1])

F(x, k) =
    -x[1] * x[2] -
    (2 * k / sqrt(eigenvalue[1])) * (
        sin(x[1]) * sin(2 * x[2]) +
        sin(2 * x[1]) * sin(x[2])
    )
F_k(x, k) =
    -(2 / sqrt(eigenvalue[1])) * (
        sin(x[1]) * sin(2 * x[2]) +
        sin(2 * x[1]) * sin(x[2])
    )

source_loads = HelmholtzLAP.assemble_source_loads(fem, F, F_k, k)

is_orthogonal = check_source_orthogonality(source_loads, Φ; atol=1e-6, rtol=1e-4)

solution = solve_lap(
    fem,
    source_loads,
    Φ,
    k;
    restart=50,
    maxiter=25000,
    reltol=1e-10,
)

u_special = FEFunction(fem.U, solution.u_special)
u_final = FEFunction(fem.U, solution.u_lap)

fig = Figure(; size=(820, 440), fontsize=16)
ax = Axis(
    fig[1, 1];
    title="special solution-rectangle case",
    xlabel="x",
    ylabel="y",
    aspect=DataAspect(),
)
plt = plot!(ax, fem.Ω, u_special; colormap=:RdBu)
Colorbar(fig[1, 2], plt)
display(fig)

fig = Figure(; size=(820, 440), fontsize=16)
ax = Axis(
    fig[1, 1];
    title="LAP solution-rectangle case",
    xlabel="x",
    ylabel="y",
    aspect=DataAspect(),
)
plt = plot!(ax, fem.Ω, u_final; colormap=:RdBu)
Colorbar(fig[1, 2], plt)

display(fig)
