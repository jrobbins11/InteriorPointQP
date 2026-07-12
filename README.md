# InteriorPointQP

A simple C++ library for solving convex quadratic programs (QPs) using
Mehrotra's predictor-corrector primal-dual interior point method.

## Problem Formulation

`InteriorPointQP::Solver` solves quadratic programs of the form:

```
minimize    (1/2) x^T P x + q^T x
subject to  G x <= w
            A x  = b
```

where `P` must be positive semi-definite, `G`/`w` define optional linear inequality constraints, and `A`/`b` define optional linear equality constraints. Both constraint blocks are optional, so the solver also handles equality-only, inequality-only, and unconstrained problems.

## Algorithm

The solver implements Mehrotra's predictor-corrector interior point method, originally proposed in
> Mehrotra, Sanjay. "On the implementation of a primal-dual interior point method." SIAM Journal on optimization 2.4 (1992): 575-601.

The specific implementation is based on the presentation in 
> Borrelli, Francesco, Alberto Bemporad, and Manfred Morari. Predictive control for linear and hybrid systems. Cambridge University Press, 2017.

## Requirements

- C++17 (or later) compiler
- [Eigen](https://eigen.tuxfamily.org/)

## Usage

```cpp
#include "InteriorPointQP.hpp"
#include <iostream>

int main()
{
    using namespace InteriorPointQP;

    // Minimize (1/2)x^T P x + q^T x subject to G x <= w
    Eigen::SparseMatrix<double> P(2, 2);
    P.insert(0, 0) = 2.0;
    P.insert(1, 1) = 2.0;

    Eigen::VectorXd q(2);
    q << -2.0, -5.0;

    Eigen::SparseMatrix<double> G(3, 2);
    G.insert(0, 0) = 1.0;  G.insert(0, 1) = -2.0;
    G.insert(1, 0) = -1.0; G.insert(1, 1) = -2.0;
    G.insert(2, 0) = -1.0; G.insert(2, 1) = 2.0;

    Eigen::VectorXd w(3);
    w << 2.0, -2.0, 3.0;

    Solver solver(P, q, G, w);

    Result result = solver.solve();

    if (result.converged && result.feasible)
    {
        std::cout << "x* = " << result.solution.transpose() << std::endl;
        std::cout << "objective = " << result.objective << std::endl;
    }

    return 0;
}
```

### Custom settings

Solver behavior is controlled via the `Settings` struct, which can be passed
at construction or updated afterward:

```cpp
Settings settings;
settings.max_iterations = 50;
settings.feasibility_tolerance = 1e-8;

Solver solver(P, q, G, w, std::nullopt, std::nullopt, settings);
// or:
solver.update_settings(settings);
```

| Field                         | Description                                                          | Default   |
|-------------------------------|------------------------------------------------------------------------|-----------|
| `max_time_sec`                | Maximum wall-clock solve time (seconds)                                | `inf`     |
| `barrier_init`                | Initial barrier parameter                                              | `1e6`     |
| `barrier_converged`           | Barrier parameter threshold for declaring convergence                  | `1e-6`    |
| `barrier_max`                 | Barrier parameter threshold for declaring infeasibility                | `1e20`    |
| `feasibility_tolerance`       | Tolerance for satisfying constraints                                   | `1e-6`    |
| `max_iterations`               | Maximum interior point iterations                                      | `100`     |
| `step_scaling`                | Step size scaling factor, in `(0, 1)`                                  | `0.999`   |
| `line_search_step_factor`     | Step size reduction factor per line search iteration, in `(0, 1)`      | `0.9`     |
| `line_search_max_iterations`  | Maximum line search iterations                                         | `1000`    |

## Result

`solve()` returns a `Result` struct containing:

- `solution` — primal solution vector `x`
- `dual_solution_v`, `dual_solution_u` — dual variables
- `slack_solution_s` — slack variables for inequality constraints
- `objective` — objective value at the solution
- `converged` — whether the barrier parameter reached `barrier_converged`
- `feasible` — whether the solution satisfies constraints within
  `feasibility_tolerance`
- `num_iterations` — number of interior point iterations taken
- `solution_time_sec` — total solve time