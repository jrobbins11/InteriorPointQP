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
    double mu_term = 1e-3;
    double mu_feas = 1e0;
    double mu_max = 1e20; // TODO: robustify infeasibility detection
    double mu_init = 1e6;
    double eps_feas = 1e-6;
    int iter_max = 100;
    double gamma = 0.999;
    double t_ls = 0.9;
    bool preprocessing_enable = true;
    double T_max = std::numeric_limits<double>::infinity();
};

struct Result
{
    Eigen::VectorXd x;
    Eigen::VectorXd v;
    Eigen::VectorXd u;
    Eigen::VectorXd s;
    double objective;
    bool converged;
    int num_iter;
    double sol_time;
    bool feasible;

    friend std::ostream& operator<<(std::ostream& os, const Result& result)
    {
        os << "InteriorPointQP Result: " << std::endl;
        if (result.x.size() < 10)
        {
            os << " x: " << result.x;
            os << " v: " << result.v;
            os << " u: " << result.u;
            os << " s: " << result.s;
        }
        os << " objective: " << result.objective << std::endl;
        os << " converged: " << result.converged << std::endl;
        os << " num_iter: " << result.num_iter << std::endl;
        os << " sol_time: " << result.sol_time << " s" << std::endl;
        os << " feasible: " << result.feasible << std::endl;
        return os;
    }
};

class Solver
{
    public:

        // constructor
        // TODO: handle unconstrained problems
        // TODO: document what the variables mean
        // TODO: require P positive semidef
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
        Eigen::SparseMatrix<double> M_, M0_, dM_;
        std::vector<Eigen::Triplet<double>> triplets_;
        Eigen::VectorXd r_C_, r_E_, r_I_, r_S_;
        Eigen::VectorXd bm_;
        Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> lu_solver_;
        Eigen::ComputationInfo lu_status_ = Eigen::ComputationInfo::Success;

        // preprocessing
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr_solver_;
        Eigen::SparseMatrix<double> P_eq_;

        // initial vars
        Eigen::VectorXd x0_, v0_, u0_, s0_;

        // working vars
        Eigen::VectorXd x_, v_, u_, s_;
        Eigen::VectorXd nu_;
        Eigen::DiagonalMatrix<double, -1> S_, Del_S_;

        // problem dimensions
        int n_, m_ineq_, m_eq_;

        // flags
        bool equalityConstrained_;        

        // helper methods
        double objective(const Eigen::Ref<const Eigen::VectorXd> x);
        void generate_system_matrix();
        void generate_rhs();
        void update_rhs();
        std::pair<double, bool> line_search(const Eigen::Ref<const Eigen::VectorXd> del_s, const Eigen::Ref<const Eigen::VectorXd> del_u);
        void get_permute_matrix(const Eigen::SparseMatrix<double> * mat_in, Eigen::SparseMatrix<double> * mat_out);
        void make_valid_equality_constraints();
        void make_valid_A();
        void make_valid_b();
        double compute_mu(const Eigen::Ref<const Eigen::VectorXd> s, const Eigen::Ref<const Eigen::VectorXd> u);
        bool compute_problem_dimensions();
        void initialize_working_matrices();
};

} // end namespace