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

struct Settings
{
    double max_time_sec = std::numeric_limits<double>::infinity();
    double barrier_init = 1e6;
    double barrier_converged = 1e-6;
    double barrier_max = 1e20;
    double feasibility_tolerance = 1e-6;
    int max_iterations = 100;
    double line_search_gamma = 0.999;
    double line_search_t = 0.9;
    int line_search_max_iterations = 1000;
};

struct Result
{
    Eigen::VectorXd solution {};
    Eigen::VectorXd dual_solution_v {};
    Eigen::VectorXd dual_solution_u {};
    Eigen::VectorXd slack_solution_s {};
    double objective {};
    bool converged {};
    bool feasible {};
    int num_iterations {};
    double solution_time_sec {};
};

class Solver
{
    public:

        // constructor
        // TODO: handle unconstrained problems
        // TODO: document what the variables mean
        // TODO: require P positive semidef
        // TODO: use std::optional
        // 
        /// Loops until the barrier parameter is less than barrier_converged and the solution is
        /// feasible within feasibility_tolerance
        Solver(
            Eigen::SparseMatrix<double> P,
            Eigen::VectorXd q,
            std::optional<Eigen::SparseMatrix<double>> G = std::nullopt,
            std::optional<Eigen::VectorXd> w = std::nullopt,
            std::optional<Eigen::SparseMatrix<double>> A = std::nullopt,
            std::optional<Eigen::VectorXd> b = std::nullopt,
            std::optional<Settings> settings = std::nullopt    
        );

        // solve method
        Result solve();

        // update methods
        void update_settings(const Settings& settings);

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