#pragma once

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <vector>
#include <cmath>
#include <utility>
#include <limits>

namespace MehrotraQP
{

struct Settings
{
    double mu_term = 1e-3;
    double mu_feas = 1e0;
    double mu_max = 1e12;
    double mu_init = 1e7;
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
    bool feas;
};

class Solver
{
    public:

        // constructor
        // TODO: handle unconstrained problems
        Solver(
            const Eigen::SparseMatrix<double>& P,
            const Eigen::VectorXd& q,
            const Eigen::SparseMatrix<double>& A = Eigen::SparseMatrix<double>{},
            const Eigen::VectorXd& b = Eigen::VectorXd{},
            const Eigen::SparseMatrix<double>& G = Eigen::SparseMatrix<double>{},
            const Eigen::VectorXd& w = Eigen::VectorXd{},
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
        Eigen::SparseMatrix<double> M, M0, dM;
        std::vector<Eigen::Triplet<double>> tripvec_dM;
        Eigen::VectorXd r_C, r_E, r_I, r_S;
        Eigen::VectorXd bm;
        Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> LU_solver;
        Eigen::ComputationInfo LU_status = Eigen::ComputationInfo::Success;

        // preprocessing
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> QR_solver;
        Eigen::SparseMatrix<double> P_eq;

        // initial vars
        Eigen::VectorXd x0, v0, u0, s0;

        // working vars
        Eigen::VectorXd x, v, u, s;
        Eigen::VectorXd nu;
        Eigen::DiagonalMatrix<double, -1> S, Del_S;

        // duality measure
        double mu;

        // problem dimensions
        int n_, m_ineq_, m_eq_;

        // flags
        bool equalityConstrained_;
        

        // helper methods
        double objective(const Eigen::Ref<const Eigen::VectorXd> x);
        void generateSystemMatrix();
        void generateRHS();
        void updateRHS();
        std::pair<double, bool> lineSearch(const Eigen::Ref<const Eigen::VectorXd> del_s, const Eigen::Ref<const Eigen::VectorXd> del_u);
        void getLinDepPermuteAndChopMatrix(const Eigen::SparseMatrix<double> * mat_in, Eigen::SparseMatrix<double> * mat_out);
        void getValidEqualityConstraints();
        void makeValid_A();
        void makeValid_b();
        double computeMu(const Eigen::Ref<const Eigen::VectorXd> s, const Eigen::Ref<const Eigen::VectorXd> u);
        bool computeProblemDimensions();
        void initializeWorkingMatrices();
};

} // end namespace