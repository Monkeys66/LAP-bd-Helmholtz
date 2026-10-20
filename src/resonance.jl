# Compute the eigenvalues and eigenvectors.
function compute_eigenpairs(fem; nev=6, which=:SM)
    λ, Φ = eigs(fem.K, fem.M; nev=nev, which=which)

    p = sortperm(real.(λ))
    λ = λ[p]
    Φ = Φ[:, p]

    return (; λ, Φ)
end

# Print the eigenvalues.
function show_eigenvalues(eigenpairs)
    λ = eigenpairs.λ
    len = length(λ)
    for i in 1:len
        @printf("Eigenvalue λ_%d = %.4f\n", i, λ[i])
    end
end


function find_resonant_indices(eigenpairs, resonance_index; rtol=1e-3, atol=1e-10)
    λ = eigenpairs.λ
    n = length(λ)

    if resonance_index < 1 || resonance_index > n
        throw(
            ArgumentError(
                "resonance_index=$resonance_index is outside the computed " *
                "eigenvalue range 1:$n.",
            ),
        )
    end

    λstar = λ[resonance_index]

    left = resonance_index
    while left > 1 && isapprox(λ[left-1], λstar; rtol=rtol, atol=atol)
        left -= 1
    end

    right = resonance_index
    while right < n && isapprox(λ[right+1], λstar; rtol=rtol, atol=atol)
        right += 1
    end

    if right == n
        throw(
            ArgumentError(
                "The resonant eigenvalue cluster reaches the end of the " *
                "computed spectrum. Increase `nev` in `compute_eigenpairs` " *
                "and recompute the eigenvalues before determining the " *
                "resonant eigenspace.",
            ),
        )
    end

    return left:right
end


# Return whether the load is orthogonal to the resonant eigenspace.
function check_source_orthogonality(source_loads, Φres; atol=1e-6, rtol=1e-4)
    b = source_loads.b
    coupling = Φres' * b
    coupling_norm = norm(coupling)

    scale = max(norm(Φres) * norm(b), 1.0)

    return coupling_norm <= atol + rtol * scale
end


# Orthogonal projection of the load vector.
function project_load_to_range(fem, b, Φres)
    G = Φres' * fem.M * Φres

    coefficients = G \ (Φres' * b)

    return b - fem.M * Φres * coefficients
end
