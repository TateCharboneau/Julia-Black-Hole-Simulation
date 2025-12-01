using DifferentialEquations          # ODE integration
using StaticArrays                   # Small fixed-size vectors for speed
using Printf                         # Formatted printing
using Makie, GLMakie                 # Visualization
using Observables                    # trail updates


function schwarzschild_geodesic(u, p, λ)
    # u: state vector [t, r, θ, φ, p_t, p_r, p_θ, p_φ]
    # p: parameter (here the black hole mass M)
    # λ: affine parameter along the geodesic
    M = p

    #  commonly used components from the state vector
    r = u[2]        # radial coordinate
    pr = u[6]       # canonical radial momentum p_r

    # conserved quantities 
    E = 1.0         # photon energy (conserved)
    L = u[8]        # angular momentum (conserved)

    # shorthand terms
    r2 = r^2
    one_minus_2M_on_r = (1 - 2 * M / r)

    # Geodesic equations (Schwarzschild, equatorial plane θ = π/2)
    # dt/dλ uses the redshift factor from g_tt = -(1-2M/r)
    dt_dλ = E / one_minus_2M_on_r

    # dr/dλ is related to the canonical momentum p_r by the metric factor
    dr_dλ = one_minus_2M_on_r * pr

    # restrict motion to the equatorial plane, so θ is constant
    dθ_dλ = 0.0

    # dφ/dλ follows from p_φ = L and the spatial metric r^2
    dφ_dλ = L / r2

    # Conserved p_t and p_φ mean derivatives of 0
    dpt_dλ = 0.0

    # dp_r/dλ: radial force terms from centrifugal barrier, gravitational potential, and the p_r-dependent term coming from the metric connection.
    dpr_dλ = (L^2 / (r2 * r)) - (M * E^2 / (r2 * one_minus_2M_on_r^2)) - (M * pr^2 / r2)

    dpθ_dλ = 0.0
    dpφ_dλ = 0.0

    # Return derivatives as Static Vector for efficiency
    return @SVector [dt_dλ, dr_dλ, dθ_dλ, dφ_dλ, dpt_dλ, dpr_dλ, dpθ_dλ, dpφ_dλ]
end


# REAL-TIME SIMULATION & RENDERING #

mutable struct PhotonState
    integrator                           # ODE integrator instance
    trail::Observable{Vector{Point2f}}   # Line positions (mutable vector behind Observable)
end

# PARAMETERS
const M_BH          = 1.0     # Black hole mass (geometric units)
const DT_FRAME      = 0.1     # Affine step per visual frame
const TRAIL_LENGTH  = 300     # Max points kept per trail
const ESCAPE_RADIUS = 25.0    # Phopton escape radius
const START_POS_X   = -20.0   # Emission x-position
const Y_MIN         = -6.0    # Lowest y for initial row of photons
const Y_MAX         =  6.0    # Highest y for initial row of photons
const Y_STEP        =   .02   # Spacing between photons (units)
const HORIZON_SAFETY = 0.15   # Stop integration before horizon
const R_S  = 2.0 * M_BH       # Event horizon radius

# HELPER FUNCTIONS

"""Compute initial canonical radial momentum p_r from (E,L,r)."""
function initial_pr(E, L, r, M)
    f = 1 - 2M / r
    inner = max(0.0, E^2 - (L^2 / r^2) * f)
    dr_dλ = -sqrt(inner)             
    return dr_dλ / f            
end

"""Create a photon launched from (START_POS_X, y0) with initial direction purely +x."""
function spawn_photon(y0::Float64)
    x_launch = START_POS_X
    r0 = sqrt(x_launch^2 + y0^2)
    phi0 = atan(y0, x_launch)
    E = 1.0
    f0 = 1 - 2M_BH / r0
    s = sin(phi0); c = cos(phi0)
    if abs(s) < 1e-8
        L = 0.0
        pr0 = -E / f0
    else
        denom = c*c + f0*s*s
        Lmag = r0 * abs(s) / sqrt(denom)
        L = (-sign(s)) * Lmag
        pr0 = - (L / (r0 * f0)) * (c / s)
    end
    u0 = @SVector [0.0, r0, pi/2, phi0, -E, pr0, 0.0, L]

    # cull near horizon or after escape radius
    condition(u, t, integrator) = (u[2] - (R_S + HORIZON_SAFETY)) * (u[2] - ESCAPE_RADIUS)
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    prob = ODEProblem(schwarzschild_geodesic, u0, (0.0, 1e9), M_BH)
    integ = init(prob, Tsit5(), reltol=1e-6, abstol=1e-6, callback=cb)

    trail_obs = Observable(Point2f[(x_launch, y0)])
    return PhotonState(integ, trail_obs)
end

"""Attempt one integration step; on error mark as invalid."""
function safe_step!(ph::PhotonState)
    try
        step!(ph.integrator, DT_FRAME, true)
    catch err
        @printf("Integrator error: %s -- marking photon invalid\n", err)
        ph.integrator.u = @SVector [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
    end
end

function run_realtime_lensing()
    # State container for active photons
    active_photons = PhotonState[]

    fig = Figure(size = (1000, 800))
    ax = Axis(fig[1, 1], aspect = DataAspect(),
              title = "Real-time Photon Tracing",
              xlabel = "x / M", ylabel = "y / M",
              backgroundcolor = :black,
              xgridvisible=false, ygridvisible=false)
    
    limits!(ax, -22, 22, -17, 17)

    lines!(ax, R_S * cos.(0:0.01:2pi), R_S * sin.(0:0.01:2pi), color=:red, linewidth=3)
    text!(ax, "Event Horizon", position = Point2f(R_S*0.7, R_S*0.7), color=:red, fontsize=15, align=(:center, :center))
    lines!(ax, 3.0 * M_BH * cos.(0:0.01:2pi), 3.0 * M_BH * sin.(0:0.01:2pi), color=:orange, linestyle=:dash)
    
    display(fig)

    # Spawn a row of photons along y in [Y_MIN, Y_MAX] spaced by Y_STEP
    for y0 in Y_MIN:Y_STEP:Y_MAX
        ph = spawn_photon(y0)
        push!(active_photons, ph)
        lines!(ax, ph.trail, color=RGBAf(0.6, 0.8, 1.0, 0.9), linewidth=1.2)
    end

    while isopen(fig.scene)
        photons_to_keep = PhotonState[]
        for ph in active_photons
            safe_step!(ph)
            u = ph.integrator.u
            r, phi = u[2], u[4]

            # Remove if invalid/terminated; otherwise update trail
            if isnan(r) || r <= R_S + 0.0001 || r >= ESCAPE_RADIUS
                continue
            end

            trail_vec = ph.trail[]
            push!(trail_vec, Point2f(r * cos(phi), r * sin(phi)))
            if length(trail_vec) > TRAIL_LENGTH
                popfirst!(trail_vec)
            end
            notify(ph.trail)

            push!(photons_to_keep, ph)
        end
        active_photons = photons_to_keep
        sleep(1/60)
    end
end

run_realtime_lensing()