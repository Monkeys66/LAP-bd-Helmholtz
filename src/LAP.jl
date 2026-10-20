# Orthogonalize the load and preserve the manually supplied k-derivative.
function prepare_source_extension(fem, source_loads, Φres)
    b = project_load_to_range(fem, source_loads.b, Φres)
    b_der = source_loads.b_der

    return (; b, b_der)
end


# Solve the Helmholtz system using an already orthogonalized load.
function compute_particular_solution(
    fem,
    source_loads,
    k,
    Φres;
    restart=50,
    maxiter=25000,
    reltol=1e-10,
    atol=1e-6,
    rtol=1e-4,
)
    if !check_source_orthogonality(source_loads, Φres; atol=atol, rtol=rtol)
        error(
            "The source is not sufficiently orthogonal to the resonant eigenspace.",
        )
    end

    b = source_loads.b
    A = fem.K - k^2 * fem.M

    T = promote_type(eltype(A), eltype(b))
    u_special = zeros(T, length(b))

    return gmres!(u_special, A, b; restart=restart, maxiter=maxiter, reltol=reltol)
end


# Compute the LAP correction and the final LAP solution.
function compute_lap_solution(fem, source_loads, u_special, Φres, k)
    G = Φres' * fem.M * Φres

    derivative_coupling = Φres' * source_loads.b_der

    rhs = -derivative_coupling - 2k * (Φres' * fem.M * u_special)

    coefficients = (2k * G) \ rhs

    return u_special + Φres * coefficients
end


"""
    solve_lap(fem, source_loads, Φres, k; kwargs...)

Orthogonalize the load and return `(; u_special, u_lap)` as solution vectors.
The manually supplied `source_loads.b_der` must describe the intended source
extension at `k`; it is used unchanged, without projection or differentiation.
"""
function solve_lap(
    fem,
    source_loads,
    Φres,
    k;
    restart=50,
    maxiter=25000,
    reltol=1e-10,
    atol=1e-6,
    rtol=1e-4,
)
    prepared_loads = prepare_source_extension(fem, source_loads, Φres)

    u_special = compute_particular_solution(
        fem,
        prepared_loads,
        k,
        Φres;
        restart=restart,
        maxiter=maxiter,
        reltol=reltol,
        atol=atol,
        rtol=rtol,
    )

    u_lap = compute_lap_solution(fem, prepared_loads, u_special, Φres, k)

    return (; u_special, u_lap)
end
