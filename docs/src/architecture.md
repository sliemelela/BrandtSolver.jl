# Internal Architecture & Flow

This section is written for developers, contributors, and future maintainers who want to understand
*how* and *why* the codebase is structured the way it is.

`BrandtSolver.jl` implements the simulation-based dynamic portfolio choice algorithm originally
proposed by Brandt et al. (2005).
The solver works backwards through time (Dynamic Programming) using cross-sectional regressions
to approximate conditional expectations.

## High-Level Execution Flow
The entire algorithm is orchestrated by `solve_portfolio_problem`.
If we have a simulation with $M$ decision steps ($t_1, t_2, \dots, t_M$),
the algorithm flows like this:

1. **Time Loop (Backwards):** Start at the last decision period $t = M$ and walk backwards to $t = 1$.
2. **Wealth Grid Loop:** At the current time $t$, evaluate the optimal policy across a predefined grid of starting wealths (e.g., $W \in [50, 100, 150]$).
3. **For each Wealth point, do the following:**
   - **Forward Simulation:** Simulate the wealth paths forward from $t$ to $T$. We assume a zero-weight allocation between $t$ and $t+1$, and apply our already-computed optimal policies for all periods after $t+1$.
   - **Evaluate Realized Moments:** Calculate the ex-post realized marginal utility components (the Taylor series terms) for all simulated paths.
   - **Cross-Sectional Regression:** Regress these realized future values onto the state variables at time $t$ to obtain *conditional expectations*.
   - **Root Finding (FOC):** Using these conditional expectations, solve the First Order Conditions (Euler equation) to find the optimal portfolio weights $\omega_t$.
4. **Interpolation:** Once $\omega_t$ is found for every point on the wealth grid, fit a linear interpolator so we can continuously evaluate $\omega_t(W)$ for the *next* step backwards.

---

## Module Breakdown

To keep the logic clean, the `src/` directory is split into distinct files representing the different mathematical phases of the algorithm.

### 1. `orchestrator.jl` (The Main Loops)
This file contains the highest-level logic and manages the dynamic programming loops.
* **`solve_portfolio_problem`**: The main entry point. Sets up the regression strategy and
    handles the backwards time loop.
* **`create_policy_interpolators`**: Loops over the `W_grid`.
    Once all weights are solved for the grid, it uses `Interpolations.jl` to build a strongly-typed
    vector of policy interpolators.
* **`compute_expectations_and_policy`**: The master function for a single time step and a
    single wealth point. It calls the simulator, asks for the regressions, and passes the results
    to the FOC optimizer.

### 2. `simulator.jl` (The Physics)
This file handles forward-time mechanics.
Because the solver works via dynamic programming, evaluating a decision at time $t$ requires
knowing the wealth outcomes at time $t+1$ and terminal time $T$.
* **`simulate_wealth_trajectory`**: Takes a starting wealth vector and rolls it forward through
    time using `calculate_next_wealth`. It applies the `future_policies` that were computed in
    previous iterations of the backwards loop.
* **`calculate_realized_term`**: Computes the actual, realized integrand $Y_n$ for the $n$-th order
    Taylor expansion term of the Euler equation. It evaluates the derivatives of the utility
    function at the terminal wealth $W_T$.

### 3. `expectations.jl` (The Regression Logic)
Because agents don't know the future, they must form expectations conditional on the
information available today.
* **`compute_and_regress_moments`**: Generates all possible cross-asset Taylor expansion terms
    using `multiexponents`. It calculates the realized terms and passes them to the regression engine.
* **`compute_conditional_expectation`**: Projects the realized future values onto the current state variables.
    Uses either `StandardOLS` (fast QR factorization) or `TrimmedOLS` (robust filtering of explosive wealth paths).

### 4. `optimizer.jl` (The Solver)
Once the expected moments are calculated, the portfolio choice problem reduces to finding the
roots of a polynomial system (the Taylor-expanded First Order Conditions).
* **`solve_2nd_order_policy`**: If the Taylor expansion is exactly 2nd-order, the
    FOC is a purely linear system ($A\omega = b$).
    This function solves it analytically using matrix inversion, injecting a small jitter (`λ * I`) to
    the diagonal to ensure the matrix remains invertible even with highly correlated assets.
* **`solve_higher_order_policy`**: For 3rd-order or higher expansions, the FOC is highly non-linear.
    This function uses `NonlinearSolve.jl` to find the roots. **Crucially**,
    it calls `solve_2nd_order_policy` first and uses the analytical linear solution as the
    initial guess, guaranteeing rapid convergence.

### 5. `utility.jl` & `types.jl` (Data Structures)
* **`types.jl`**: Defines the `SolverParams` and `RegressionStrategy` structs.
    It also holds the `Recorder` types used for cleanly logging internal solver data for
    debugging and plotting without impacting performance.
* **`utility.jl`**: Contains `create_utility_from_ad`, which uses `ForwardDiff.jl` to automatically
    generate and aggressively cache all higher-order derivatives of the user's utility function,
    along with computing the inverse utility via `Roots.jl`.