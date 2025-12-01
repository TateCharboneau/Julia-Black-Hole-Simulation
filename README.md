# Julia-Black-Hole-Simulation

Simple Julia project that visualizes photon geodesics around a Schwarzschild black hole using Makie.

## Prerequisites

- Julia 1.9 or newer (tested with 1.11)
- OpenGL-capable GPU/driver for `GLMakie`

## Install dependencies

From the project root run:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

This installs `DifferentialEquations`, `StaticArrays`, `GLMakie`, and other dependencies declared in `Project.toml`/`Manifest.toml`.

## Run the visualizer

Start the photon simulation with:

```powershell
julia --project=. BlackHoleRender/render.jl
```

The command opens a Makie window tracing photons emitted from the left toward the black hole.

## Tweaking the scene

Adjust the constants near the top of `BlackHoleRender/render.jl` (for example `Y_MIN`, `Y_MAX`, `Y_STEP`, or `DT_FRAME`) to experiment with different emission patterns or integration speeds. Restart the command above after saving changes.