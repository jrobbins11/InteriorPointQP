#include "MehrotraQP.hpp"

#include <chrono>
#include <stdexcept>
#include <limits>

using namespace MehrotraQP;

namespace 
{
    // utilities
    void getTripletsForMatrix(const Eigen::SparseMatrix<double> * mat_ptr, std::vector<Eigen::Triplet<double>> &tripvec,
        int rowOffset, int colOffset)
    {       
        for (int i=0; i<mat_ptr->outerSize(); i++)
        {
            for (typename Eigen::SparseMatrix<double>::InnerIterator it(*mat_ptr,i); it; ++it)
                tripvec.emplace_back(it.row()+rowOffset, it.col()+colOffset, it.value());
        }
    }

    void getTripletsForMatrixDiagonal(const Eigen::Ref<const Eigen::VectorXd> d, std::vector<Eigen::Triplet<double>> &tripvec,
                int rowOffset, int colOffset)
    {
        int m = d.rows();
        for (int i=0; i<m; i++)
        {
            tripvec.emplace_back(i+rowOffset, i+colOffset, d(i));
        }
    }

    // constants
    constexpr double INFTY = std::numeric_limits<double>::infinity();
    constexpr double EPSILON = Eigen::NumTraits<double>::dummy_precision();
}

// constructor
Solver::Solver(
    const Eigen::SparseMatrix<double>& P,
    const Eigen::VectorXd& q,
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b,
    const Eigen::SparseMatrix<double>& G,
    const Eigen::VectorXd& w,
    const Settings& settings 
)
{    
    // copy in data, check for default-constructed arguments
    P_ = P;
    q_ = q;
    A_ = (A.rows() == 0) ? Eigen::SparseMatrix<double>(0, q.size()) : A;
    b_ = b;
    G_ = (G.rows() == 0) ? Eigen::SparseMatrix<double>(0, q.size()) : G;
    w_ = w;
    settings_ = settings;    

    // preprocessing (make sure A is full rank)
    if (settings_.preprocessing_enable)
    {
        getValidEqualityConstraints();
        makeValid_A();
        makeValid_b();
    }

    // compute problem dimensions
    if (!computeProblemDimensions())
    {
        throw std::invalid_argument("Inconsistent problem dimensions");
    }
}

void Solver::update_settings(const Settings& settings) 
{
    settings_ = settings;
}

// solve
Results Solver::solve()
{
    // declare
    double mu, h, mu_pred, sigma;
    Eigen::VectorXd del, del_x, del_v, del_u, del_s;
    Eigen::VectorXd s_pred, u_pred;
    bool numerical_issue;
    std::pair<double, bool> line_search_out;

    // start timer
    auto timer_init = std::chrono::high_resolution_clock::now();

    // running timer
    double running_timer = 0; // init
    const double T_max = (settings_.T_max > 0.0) ? settings_.T_max : INFTY;

    // initialize start vars
    x0 = Eigen::VectorXd::Zero(n_);
    double zeta = sqrt(settings_.mu_init);
    v0 = Eigen::VectorXd::Zero(m_eq_);
    u0 = zeta*Eigen::VectorXd::Ones(m_ineq_);
    s0 = zeta*Eigen::VectorXd::Ones(m_ineq_);

    // initialize primal and dual vars
    x = x0;
    v = v0;
    u = u0;
    s = s0;

    // initialize working matrices
    initializeWorkingMatrices();

    // outer loop init
    int k = 0;
    numerical_issue = false;

    // compute duality measure
    mu = computeMu(s, u);

    // loop
    while ((mu > settings_.mu_term) && (k < settings_.iter_max) && 
        !numerical_issue && (running_timer < T_max) && (mu <= settings_.mu_max))
    {
        // generate system matrix and decompose
        generateSystemMatrix();

        // check for numerical issues and terminate if necessary
        if (LU_status != Eigen::ComputationInfo::Success)
        {
            numerical_issue = true;
            break;
        } 

        // predictor step
        generateRHS();
        del = LU_solver.solve(bm);
        del_u = del.segment(n_+m_eq_, m_ineq_);
        del_s = del.segment(n_+m_eq_+m_ineq_, m_ineq_);

        // line search for step size
        line_search_out = lineSearch(del_s, del_u);
        h = line_search_out.first;
        if (line_search_out.second)
        {
            numerical_issue = true;
            break;
        }

        // predicted duality measure
        s_pred = s + h*del_s;
        u_pred = u + h*del_u;
        mu_pred = computeMu(s_pred, u_pred);

        // centering parameter
        sigma = pow(mu_pred/mu, 3);

        // calculate corrected nu and recompute search direction
        Del_S.diagonal() = del_s;
        nu = (sigma*mu)*Eigen::VectorXd::Ones(m_ineq_) - Del_S*del_u;
        updateRHS();

        del = LU_solver.solve(bm);
        del_x = del.segment(0, n_);
        del_v = del.segment(n_, m_eq_);
        del_u = del.segment(n_+m_eq_, m_ineq_);
        del_s = del.segment(n_+m_eq_+m_ineq_, m_ineq_);

        // line search for step size
        line_search_out = lineSearch(del_s, del_u);
        h = line_search_out.first;
        if (line_search_out.second)
        {
            numerical_issue = true;
            break;
        }

        // update
        x = x + settings_.gamma*h*del_x;
        v = v + settings_.gamma*h*del_v;
        u = u + settings_.gamma*h*del_u;
        s = s + settings_.gamma*h*del_s;

        // update running timer
        auto timer_running = std::chrono::high_resolution_clock::now();
        auto duration_running = std::chrono::duration_cast<std::chrono::microseconds>(timer_running - timer_init);
        running_timer = 1e-6 * ((double) duration_running.count());

        // compute duality measure
        mu = computeMu(s, u);

        // iterate 
        k++;
    }

    // timing
    auto timer_final = std::chrono::high_resolution_clock::now();
    auto duration_final = std::chrono::duration_cast<std::chrono::microseconds>(timer_final - timer_init);
    double time = 1e-6 * ((double) duration_final.count());

    // assemble results
    Results results;
    results.x = x;
    results.v = v;
    results.u = u;
    results.s = s;
    results.objective = objective(x);
    results.feas = mu <= settings_.mu_feas; // divergence check
    results.converged = !numerical_issue && (k < settings_.iter_max) && results.feas;
    results.num_iter = k;
    results.sol_time = time;

    // return
    return results;
}

/* helper methods */

// objective function
double Solver::objective(const Eigen::Ref<const Eigen::VectorXd> x_in)
{
    double J;
    J = 0.5*x_in.dot(P_*x_in) + q_.dot(x_in);
    return J;
}

// initialize working matrices
void Solver::initializeWorkingMatrices()
{
    // precompute transposes
    A_T_ = A_.transpose();
    G_T_ = G_.transpose();

    // initialize working matrices
    bm.resize(n_ + m_eq_ + m_ineq_ + m_ineq_);

    // precompute constant part of M matrix
    // M = [P, A', G', 0;
    //      A, 0, 0, 0;
    //      G, 0, 0, I;
    //      0, 0, S, Z]
    std::vector<Eigen::Triplet<double>> tripvec_M0;
    tripvec_M0.reserve(P_.nonZeros() + 2*A_.nonZeros() + 2*G_.nonZeros() + m_ineq_);
    M0.resize(n_ + m_eq_ + 2*m_ineq_, n_ + m_eq_ + 2*m_ineq_);

    getTripletsForMatrix(&P_, tripvec_M0, 0, 0);
    if (equalityConstrained_)
        getTripletsForMatrix(&A_T_, tripvec_M0, 0, n_);
    getTripletsForMatrix(&G_T_, tripvec_M0, 0, n_ + m_eq_);

    if (equalityConstrained_)
        getTripletsForMatrix(&A_, tripvec_M0, n_, 0);

    getTripletsForMatrix(&G_, tripvec_M0, n_ + m_eq_, 0);
    getTripletsForMatrixDiagonal(Eigen::VectorXd::Ones(m_ineq_), tripvec_M0, n_ + m_eq_, n_ + m_eq_ + m_ineq_);

    M0.setFromTriplets(tripvec_M0.begin(), tripvec_M0.end());

    // pre-allocate and initialize dM
    tripvec_dM.clear();
    tripvec_dM.reserve(2*m_ineq_);
    dM.resize(n_ + m_eq_ + 2*m_ineq_, n_ + m_eq_ + 2*m_ineq_);
    getTripletsForMatrixDiagonal(s, tripvec_dM, n_ + m_eq_ + m_ineq_, n_ + m_eq_);
    getTripletsForMatrixDiagonal(u, tripvec_dM, n_ + m_eq_ + m_ineq_, n_ + m_eq_ + m_ineq_);
    dM.setFromTriplets(tripvec_dM.begin(), tripvec_dM.end());

    // analyze pattern
    M = M0 + dM;
    LU_solver.analyzePattern(M);
}

// generate system matrix
void Solver::generateSystemMatrix()
{
    // update changing part of system matrix
    // update dM based on known sparsity pattern
    int i_s = 0;
    int i_u = 0;
    for (int k=0; k<dM.outerSize(); k++)
    {
        for (typename Eigen::SparseMatrix<double>::InnerIterator it(dM,k); it; ++it)
        {
            if (i_s < s.rows())
            {
                it.valueRef() = s(i_s);
                i_s++;
            }
            else if (i_u < u.rows())
            {
                it.valueRef() = u(i_u);
                i_u++;
            }
        }
    }
    
    // update M
    M = M0 + dM;

    // LU decomposition
    LU_solver.factorize(M);

    // get status
    LU_status = LU_solver.info();

}

// generate right hand side of linear system
void Solver::generateRHS()
{
    // compute r_E term
    if (equalityConstrained_)
    {
        // compute r_C term
        r_C = P_*x + q_ + A_T_*v + G_T_*u;

        // compute r_E term
        r_E = A_*x - b_;
    }
    else
        r_C = P_*x + q_ + G_T_*u;

    // compute r_I term
    r_I = (G_*x - w_) + s;

    // compute r_S term
    S.diagonal() = s;
    r_S = S*u; // no centering term

    // RHS
    bm.segment(0, n_) = -r_C;
    if (equalityConstrained_)
        bm.segment(n_, m_eq_) = -r_E;
    bm.segment(n_+m_eq_, m_ineq_) = -r_I;
    bm.segment(n_+m_eq_+m_ineq_, m_ineq_) = -r_S;
}

// update RHS
void Solver::updateRHS()
{
    // update r_S term
    r_S = r_S - nu;
    
    // update RHS
    bm.segment(n_+m_eq_+m_ineq_, m_ineq_) = -r_S;
}

// line search
std::pair<double, bool> Solver::lineSearch(const Eigen::Ref<const Eigen::VectorXd> del_s, const Eigen::Ref<const Eigen::VectorXd> del_u)
{
    // declare
    Eigen::VectorXd s_ds, u_du;

    // init
    double h = 1;
    bool valid = false;
    int cnt_max = 10000;
    int cnt = 0;

    // loop
    while (!valid && (cnt < cnt_max) && (h > EPSILON))
    {
        // updated s, u
        s_ds = s + h*del_s;
        u_du = u + h*del_u;

        // check for validity of step
        if ((s_ds.minCoeff() > 0) && (u_du.minCoeff() > 0))
            valid = true;
        else
            h = settings_.t_ls*h;

        // increment
        cnt++;
    }

    // check for validity
    bool numerical_issue = ((cnt == cnt_max) || (h <= EPSILON));

    // output
    return std::pair<double, bool> {h, numerical_issue};
}

// duality measure 
double Solver::computeMu(const Eigen::Ref<const Eigen::VectorXd> s_in, const Eigen::Ref<const Eigen::VectorXd> u_in)
{
    return (s_in.dot(u_in))/m_ineq_;
}

// pre-processing
void Solver::getLinDepPermuteAndChopMatrix(const Eigen::SparseMatrix<double> * mat_in, Eigen::SparseMatrix<double> * mat_out)
{
    // check for empty input matrix
    if (mat_in->rows() == 0 || mat_in->cols() == 0)
    {
        mat_out->resize(0, 0);
        return;
    }

    // matrix transpose
    Eigen::SparseMatrix<double> mat_T = mat_in->transpose();

    // compute QR decomposition
    QR_solver.analyzePattern(mat_T);
    QR_solver.factorize(mat_T);

    // get permutation matrix and its indices
    Eigen::PermutationMatrix<-1, -1> P_full = QR_solver.colsPermutation();
    Eigen::VectorXi ind_full = P_full.indices();

    // construct permute and chop matrix directly
    std::vector<Eigen::Triplet<double>> tripvec;
    tripvec.reserve(ind_full.size());
    for (int i=0; i<QR_solver.rank(); i++) // QR solver automatically puts linearly dependent rows at end
        tripvec.push_back(Eigen::Triplet<double>(ind_full[i], i, 1));

    mat_out->resize(mat_T.cols(), QR_solver.rank());
    mat_out->setFromTriplets(tripvec.begin(), tripvec.end());
}

bool Solver::computeProblemDimensions()
{
    // get dimension variables
    n_ = P_.rows();
    m_eq_ = A_.rows();
    m_ineq_ = G_.rows();
    equalityConstrained_ = (m_eq_ == 0) ? false : true;

    // check validity
    return n_ == q_.size() && n_ == P_.cols() && n_ == A_.cols() && n_ == G_.cols() 
        && m_eq_ == b_.size() && m_ineq_ == w_.size();
}

void Solver::getValidEqualityConstraints()
{
    // remove any invalid constraints (linearly dependent)
    getLinDepPermuteAndChopMatrix(&A_, &P_eq);
}

void Solver::makeValid_A()
{
    A_ = (A_.transpose()*P_eq).transpose();
}

void Solver::makeValid_b()
{
    b_ = (b_.transpose()*P_eq).transpose();
}

