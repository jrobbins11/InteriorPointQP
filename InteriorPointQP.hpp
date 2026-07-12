#pragma once

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <ostream>
#include <cmath>
#include <utility>
#include <limits>
#include <optional>

namespace InteriorPointQP
{

/**
 * @brief Settings structure for QP solver
 * 
 */
struct Settings
{
    /// max computation time in seconds
    double max_time_sec = std::numeric_limits<double>::infinity();

    /// initial barrier parameter
    double barrier_init = 1e6;

    /// barrier parameter value below which the solver is determined to be converged
    double barrier_converged = 1e-6;
    
    /// barrier parameter value above which the problem is determined to be infeasible 
    double barrier_max = 1e20;

    /// tolerance for determining that a solution satisfies constraints
    double feasibility_tolerance = 1e-6;

    /// maximum number of interior point iterations
    int max_iterations = 100;

    /// step size scaling, must be less in (0, 1)
    double step_scaling = 0.999;

    /// line search step size reduces by this factor at every iteration, must be in (0, 1)
    double line_search_step_factor = 0.9;

    /// max iterations for line search
    int line_search_max_iterations = 1000;
};

/**
 * @brief Solution structure for QP solver
 * 
 */
struct Result
{
    /// solution vector x
    Eigen::VectorXd solution {};

    /// dual solution vector v
    Eigen::VectorXd dual_solution_v {};

    /// dual solution vector u
    Eigen::VectorXd dual_solution_u {};
    
    /// slack variable solution vector s
    Eigen::VectorXd slack_solution_s {};

    /// objective function value (1/2)*x^T*P*x + q^T*x
    double objective {};

    /// convergence flag, true if optimization converged
    bool converged {};

    /// feasibility flag, true if solution x is feasible within feasibility tolerance
    bool feasible {};

    /// number of interior point iterations to get solution
    int num_iterations {};

    /// total solution time in seconds
    double solution_time_sec {};
};

/**
 * @brief Class to solve the quadratic program (1/2)*x^T*P*x + q^T*x s.t. G*x <= w and A*x = b using Mehrotra's primal-dual interior point method
 * 
 */
class Solver
{
    public:

        /** 
         * @brief Construct Solver object
         * @param P quadratic cost matrix, must be positive semi-definite
         * @param q linear cost vector
         * @param G inequality constraint matrix (optional)
         * @param w inequality constraint vector (optional)
         * @param A equality constraint matrix (optional)
         * @param b equality constraint vector (optional)
         * @throws std::invalid_argument if problem dimensions are inconsistent or settings are invalid
         */ 
        Solver(
            Eigen::SparseMatrix<double> P,
            Eigen::VectorXd q,
            std::optional<Eigen::SparseMatrix<double>> G = std::nullopt,
            std::optional<Eigen::VectorXd> w = std::nullopt,
            std::optional<Eigen::SparseMatrix<double>> A = std::nullopt,
            std::optional<Eigen::VectorXd> b = std::nullopt,
            std::optional<Settings> settings = std::nullopt    
        );

        /**
         * @brief Solve quadratic program (1/2)*x^T*P*x + q^T*x s.t. G*x <= w and A*x = b.
         * @return Result structure
         */
        Result solve();

        /**
         * @brief Update settings structure
         * 
         * @param settings updated settings structure
         * @throw std::invalid_argument if settings are not valid
         */
        void update_settings(const Settings& settings);

        /**
         * @brief Check that problem settings are valid
         * 
         * @param settings settings structure
         * @return true if settings are valid
         */
        bool validate_settings(const Settings& settings) const;

    private:

        // problem matrices / vectors
        Eigen::SparseMatrix<double> P_, A_, G_, A_T_, G_T_;
        Eigen::VectorXd q_, b_, w_;

        // settings
        Settings settings_;

        // linear system
        Eigen::SparseMatrix<double> M0_, dM_;
        Eigen::VectorXd bm_;
        Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> lu_solver_;
        Eigen::ComputationInfo lu_status_ = Eigen::ComputationInfo::Success;

        // working vars
        Eigen::VectorXd x_, v_, u_, s_;

        // problem dimensions
        int n_, m_ineq_, m_eq_;   

        // helper methods
        double objective(const Eigen::Ref<const Eigen::VectorXd> x);
        void generate_system_matrix();
        void generate_rhs();
        void update_rhs(const Eigen::Ref<const Eigen::ArrayXd> nu);
        std::pair<double, bool> line_search(const Eigen::Ref<const Eigen::VectorXd> del_s, const Eigen::Ref<const Eigen::VectorXd> del_u);
        void remove_linearly_dependent_equality_constraints();
        double compute_mu(const Eigen::Ref<const Eigen::VectorXd> s, const Eigen::Ref<const Eigen::VectorXd> u) const;
        bool check_problem_dimensions();
        void initialize_working_matrices();
        bool is_feasible(const Eigen::Ref<const Eigen::VectorXd> x) const;
        Result solve_equality_constrained();
};

} // end namespace

// printing overloads
std::ostream& operator<<(std::ostream& os, const InteriorPointQP::Settings& settings);
std::ostream& operator<<(std::ostream& os, const InteriorPointQP::Result& result);