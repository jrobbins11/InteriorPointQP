#pragma once

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <ostream>
#include <vector>
#include <cmath>
#include <utility>
#include <limits>

namespace InteriorPointQP
{

struct Settings
{
    double max_time_sec = std::numeric_limits<double>::infinity();
    double barrier_init = 1e6;
    double barrier_terminal = 1e-6;
    double barrier_max = 1e20; // TODO: robustify infeasibility detection
    double feasibility_tolerance = 1e-6;
    int max_iterations = 100;
    double line_search_gamma = 0.999;
    double line_search_t = 0.9;
    int line_search_max_iterations = 1000;

    friend std::ostream& operator<<(std::ostream& os, const Settings& settings);
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

    friend std::ostream& operator<<(std::ostream& os, const Result& result);
};

class Solver
{
    public:

        // constructor
        // TODO: handle unconstrained problems
        // TODO: document what the variables mean
        // TODO: require P positive semidef
        // TODO: use std::optional
        Solver(
            const Eigen::SparseMatrix<double>& P,
            const Eigen::VectorXd& q,
            const Eigen::SparseMatrix<double>& G = Eigen::SparseMatrix<double>{},
            const Eigen::VectorXd& w = Eigen::VectorXd{},
            const Eigen::SparseMatrix<double>& A = Eigen::SparseMatrix<double>{},
            const Eigen::VectorXd& b = Eigen::VectorXd{},
            const Settings& settings = Settings{}    
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
};

} // end namespace