#pragma once

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <utility>

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
    double T_max = 0;
};

struct Results
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
        Solver(const Eigen::SparseMatrix<double>& P,
            const Eigen::VectorXd& q,
            const Eigen::SparseMatrix<double>& A = Eigen::SparseMatrix<double>{},
            const Eigen::VectorXd& b = Eigen::VectorXd{},
            const Eigen::SparseMatrix<double>& G = Eigen::SparseMatrix<double>{},
            const Eigen::VectorXd& w = Eigen::VectorXd{});

        // solve method
        Results solve();

    private:

        // problem matrices / vectors
        Eigen::SparseMatrix<double> P;
        Eigen::VectorXd q;
        Eigen::SparseMatrix<double> A, A_T;
        Eigen::VectorXd b;
        Eigen::SparseMatrix<double> G, G_T;
        Eigen::VectorXd w;

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
        int n;
        int m_ineq;
        int m_eq;

        // flags
        bool equalityConstrained;
        bool A_updated;
        bool b_updated;
   
        // settings
        Settings settings;

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
        void computeProblemDimensions();
        void initializeWorkingMatrices();
};

} // end namespace