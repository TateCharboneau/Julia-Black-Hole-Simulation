using DifferentialEquations          # ODE integration
using StaticArrays                   # Small fixed-size vectors for speed
using Printf                         # Formatted printing
using SciMLBase                      # Integrator types
using Makie, GLMakie                 # Visualization
using Observables                    # Reactive trail/color updates



function schwarzschild_geodesic(u, p, λ)
    # u: state vector [t, r, θ, φ, p_t, p_r, p_θ, p_φ]
    # p: parameter (here the black hole mass M)
    # λ: affine parameter along the geodesic
    M = p

    # Extract commonly used components from the state vector
    r = u[2]        # radial coordinate
    pr = u[6]       # canonical radial momentum p_r

    # Conserved quantities (we set E=1 as a convenient normalization)
    E = 1.0         # photon energy (conserved)
    L = u[8]        # angular momentum (conserved)

    # Precompute shorthand terms
    r2 = r^2
    one_minus_2M_on_r = (1 - 2 * M / r)

    # Geodesic equations (Schwarzschild, equatorial plane θ = π/2)
    # dt/dλ uses the redshift factor from g_tt = -(1-2M/r)
    dt_dλ = E / one_minus_2M_on_r

    # dr/dλ is related to the canonical momentum p_r by the metric factor
    dr_dλ = one_minus_2M_on_r * pr

    # We restrict motion to the equatorial plane, so θ is constant
    dθ_dλ = 0.0

    # dφ/dλ follows from p_φ = L and the spatial metric r^2
    dφ_dλ = L / r2

    # Conserved p_t and p_φ mean their derivatives vanish
    dpt_dλ = 0.0

    # dp_r/dλ: radial force terms from centrifugal barrier, gravitational potential,
    # and the p_r-dependent term coming from the metric connection.
    dpr_dλ = (L^2 / (r2 * r)) - (M * E^2 / (r2 * one_minus_2M_on_r^2)) - (M * pr^2 / r2)

    dpθ_dλ = 0.0
    dpφ_dλ = 0.0

    # Return derivatives as a Static Vector for efficiency
    return @SVector [dt_dλ, dr_dλ, dθ_dλ, dφ_dλ, dpt_dλ, dpr_dλ, dpθ_dλ, dpφ_dλ]
end


# --- SECTION 2: REAL-TIME SIMULATION & RENDERING (WITH STABILITY FIXES) ---

mutable struct PhotonState
    integrator::SciMLBase.DEIntegrator   # ODE integrator instance
    trail::Observable{Vector{Point2f}}   # Line positions (mutable vector behind Observable)
    color::Observable{RGBAf}             # Fading display color (reactive)
    age::Float32                         # Frame-based age counter
    base_color::RGBAf                    # Starting (unfaded) color
end

# ---------------------------- Configuration Constants -------------------------
const M_BH                = 1.0        # Black hole mass (geometric units)
const DT_FRAME            = 0.5        # Affine step per visual frame
const SPAWN_RATE_BASE     = 4          # Target spawns per frame before throttling
const TRAIL_LENGTH        = 160        # Max points kept per trail
const ESCAPE_RADIUS       = 25.0       # Treat beyond this as escaped
const START_POS_X         = -20.0      # Mean emission x-position
const LAUNCH_X_JITTER     = 0.5        # +/- jitter to break vertical line artifact
const MAX_PHOTONS         = 450        # Upper bound on concurrent photons
const PHOTON_CAP_SOFT_FRAC = 0.85       # Throttle threshold fraction
const FADE_TIME_FRAMES    = 320.0f0    # e-folding time for alpha fade
const ALPHA_FLOOR         = 0.02f0     # Minimum alpha before removal
const TRAIL_POINT_STRIDE  = 1          # Decimate trail updates (1 = keep all)
const HORIZON_SAFETY_INIT = 0.15       # Stop integration before horizon
const HORIZON_SAFETY_REINIT = 0.15     # Same margin on reinit

# Derived characteristic radii
const R_S  = 2.0 * M_BH                 # Event horizon radius
const B_CRIT = 3 * sqrt(3) * M_BH       # Critical impact parameter (photon sphere)

# ----------------------------- Helper Functions ------------------------------

"""Compute initial canonical radial momentum p_r from (E,L,r)."""
function initial_pr(E, L, r, M)
    f = 1 - 2M / r
    inner = max(0.0, E^2 - (L^2 / r^2) * f)
    dr_dλ = -sqrt(inner)                 # Inward pointing initial direction
    return dr_dλ / f                     # Convert to canonical p_r
end

"""Spawn a single photon state (returns PhotonState)."""
function spawn_photon()
    b = (rand() * 2 - 1) * 10.0
    y0 = b
    x_launch = START_POS_X + (rand() - 0.5) * (2*LAUNCH_X_JITTER)
    r0 = sqrt(x_launch^2 + y0^2)
    phi0 = atan(y0, x_launch)
    E = 1.0
    L = b * E
    pr0 = initial_pr(E, L, r0, M_BH)
    u0 = @SVector [0.0, r0, pi/2, phi0, -E, pr0, 0.0, L]

    # Terminate near horizon or after escape radius
    safety = HORIZON_SAFETY_INIT
    condition(u, t, integrator) = (u[2] - (R_S + safety)) * (u[2] - ESCAPE_RADIUS)
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    prob = ODEProblem(schwarzschild_geodesic, u0, (0.0, 1e9), M_BH)
    integ = init(prob, Tsit5(), reltol=1e-6, abstol=1e-6, callback=cb,
                 unstable_check = (dt, u, p, t) -> false)

    # Visual parameters
    dist_to_crit = abs(abs(b) - B_CRIT)
    brightness = 0.25 + 0.55 * exp(-0.5 * dist_to_crit^2)
    base_alpha = 0.35 + 0.35 * exp(-0.5 * dist_to_crit^2)
    base_color = RGBAf(brightness, brightness, 1.0, base_alpha)
    trail_obs = Observable(Point2f[(x_launch, y0)])
    color_obs = Observable(base_color)
    return PhotonState(integ, trail_obs, color_obs, 0.0f0, base_color)
end

"""Attempt one integration step; reinitialize on failure with relaxed tolerances."""
function safe_step!(ph::PhotonState)
    try
        step!(ph.integrator, DT_FRAME, true)
    catch err
        @printf("Integrator error: %s -- attempting reinit with relaxed tolerances\n", err)
        u_curr = ph.integrator.u
        if isnan(u_curr[2]) || u_curr[2] <= R_S + 0.05
            # Mark invalid
            ph.integrator.u = @SVector [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
            return
        end
        safety = HORIZON_SAFETY_REINIT
        condition2(u, t, integrator) = (u[2] - (R_S + safety)) * (u[2] - ESCAPE_RADIUS)
        affect2!(integrator) = terminate!(integrator)
        cb2 = ContinuousCallback(condition2, affect2!)
        prob2 = ODEProblem(schwarzschild_geodesic, u_curr, (0.0, 1e9), M_BH)
        try
            ph.integrator = init(prob2, Vern7(), reltol=1e-5, abstol=1e-5, callback=cb2,
                                 unstable_check = (dt, u, p, t) -> false)
            step!(ph.integrator, DT_FRAME, true)
        catch err2
            @printf("Reinitialization failed: %s -- dropping photon\n", err2)
            ph.integrator.u = @SVector [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
        end
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

    while isopen(fig.scene)
        # ---------------- Spawning with Throttling -----------------
        effective_spawn = min(SPAWN_RATE_BASE, max(0, MAX_PHOTONS - length(active_photons)))
        if length(active_photons) > MAX_PHOTONS
            effective_spawn = 0
        elseif length(active_photons) > floor(Int, PHOTON_CAP_SOFT_FRAC * MAX_PHOTONS)
            effective_spawn = max(1, cld(MAX_PHOTONS - length(active_photons), 30))
        end
        for _ in 1:effective_spawn
            ph = spawn_photon()
            push!(active_photons, ph)
            lines!(ax, ph.trail, color=ph.color, linewidth=1.2)
        end

        photons_to_keep = PhotonState[]

        # Helper: attempt a step safely. If the integrator errors (e.g. dt forced below eps),
        # try reinitializing the integrator from the current state with relaxed tolerances
        # and a more robust solver. If reinit fails, mark the photon as invalid.
        photon_index = 0
        for ph in active_photons
            photon_index += 1
            safe_step!(ph)
            u = ph.integrator.u
            r, phi = u[2], u[4]

            # Cull invalid / terminated photons
            if isnan(r) || r <= R_S + 0.01 || r >= ESCAPE_RADIUS
                continue
            end

            # Aging & fade
            ph.age += 1.0f0
            fade = exp(-ph.age / FADE_TIME_FRAMES)
            new_alpha = ph.base_color.alpha * fade
            if new_alpha < ALPHA_FLOOR
                continue
            end
            if mod(photon_index, 2) == 0 || ph.age < 10
                ph.color[] = RGBAf(ph.base_color.r, ph.base_color.g, ph.base_color.b, new_alpha)
            end

            # Trail update (decimation supported)
            if TRAIL_POINT_STRIDE == 1 || mod(Int(ph.age), TRAIL_POINT_STRIDE) == 0
                trail_vec = ph.trail[]
                push!(trail_vec, Point2f(r * cos(phi), r * sin(phi)))
                if length(trail_vec) > TRAIL_LENGTH
                    popfirst!(trail_vec)
                end
                notify(ph.trail)
            end
            push!(photons_to_keep, ph)
        end
        active_photons = photons_to_keep
        
        sleep(1/60)
    end
end

run_realtime_lensing()