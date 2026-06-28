#include "InteriorPointQP.hpp"

#include <chrono>
#include <stdexcept>
#include <limits>

using namespace InteriorPointQP;

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
    constexpr double EPSILON = Eigen::NumTraits<double>::dummy_precision();
}

// constructor
Solver::Solver(
    const Eigen::SparseMatrix<double>& P,
    const Eigen::VectorXd& q,
    const Eigen::SparseMatrix<double>& G,
    const Eigen::VectorXd& w,
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b,
    const Settings& settings 
)
{    
    // copy in data, check for default-constructed arguments
    P_ = P;
    q_ = q;
    G_ = (G.rows() == 0) ? Eigen::SparseMatrix<double>(0, q.size()) : G;
    w_ = w;
    A_ = (A.rows() == 0) ? Eigen::SparseMatrix<double>(0, q.size()) : A;
    b_ = b;
    settings_ = settings;    

    // preprocessing (make sure A is full rank)
    if (settings_.preprocessing_enable)
    {
        make_valid_equality_constraints();
        make_valid_A();
        make_valid_b();
    }

    // compute problem dimensions
    if (!compute_problem_dimensions())
    {
        throw std::invalid_argument("Inconsistent problem dimensions");
    }
}

void Solver::update_settings(const Settings& settings) 
{
    settings_ = settings;
}

// solve
Result Solver::solve()
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

    // initialize start vars
    x0_ = Eigen::VectorXd::Zero(n_);
    double zeta = sqrt(settings_.mu_init);
    v0_ = Eigen::VectorXd::Zero(m_eq_);
    u0_ = zeta*Eigen::VectorXd::Ones(m_ineq_);
    s0_ = zeta*Eigen::VectorXd::Ones(m_ineq_);

    // initialize primal and dual vars
    x_ = x0_;
    v_ = v0_;
    u_ = u0_;
    s_ = s0_;

    // initialize working matrices
    initialize_working_matrices();

    // outer loop init
    int k = 0;
    numerical_issue = false;

    // compute duality measure
    mu = compute_mu(s_, u_);

    // loop
    while ((mu > settings_.mu_term) && (k < settings_.iter_max) && 
        !numerical_issue && (running_timer < settings_.T_max) && (mu <= settings_.mu_max))
    {
        // generate system matrix and decompose
        generate_system_matrix();

        // check for numerical issues and terminate if necessary
        if (lu_status_ != Eigen::ComputationInfo::Success)
        {
            numerical_issue = true;
            break;
        } 

        // predictor step
        generate_rhs();
        del = lu_solver_.solve(bm_);
        del_u = del.segment(n_+m_eq_, m_ineq_);
        del_s = del.segment(n_+m_eq_+m_ineq_, m_ineq_);

        // line search for step size
        line_search_out = line_search(del_s, del_u);
        h = line_search_out.first;
        if (line_search_out.second)
        {
            numerical_issue = true;
            break;
        }

        // predicted duality measure
        s_pred = s_ + h*del_s;
        u_pred = u_ + h*del_u;
        mu_pred = compute_mu(s_pred, u_pred);

        // centering parameter
        sigma = pow(mu_pred/mu, 3);

        // calculate corrected nu and recompute search direction
        Del_S_.diagonal() = del_s;
        nu_ = (sigma*mu)*Eigen::VectorXd::Ones(m_ineq_) - Del_S_*del_u;
        update_rhs();

        del = lu_solver_.solve(bm_);
        del_x = del.segment(0, n_);
        del_v = del.segment(n_, m_eq_);
        del_u = del.segment(n_+m_eq_, m_ineq_);
        del_s = del.segment(n_+m_eq_+m_ineq_, m_ineq_);

        // line search for step size
        line_search_out = line_search(del_s, del_u);
        h = line_search_out.first;
        if (line_search_out.second)
        {
            numerical_issue = true;
            break;
        }

        // update
        x_ = x_ + settings_.gamma*h*del_x;
        v_ = v_ + settings_.gamma*h*del_v;
        u_ = u_ + settings_.gamma*h*del_u;
        s_ = s_ + settings_.gamma*h*del_s;

        // update running timer
        auto timer_running = std::chrono::high_resolution_clock::now();
        auto duration_running = std::chrono::duration_cast<std::chrono::microseconds>(timer_running - timer_init);
        running_timer = 1e-6 * ((double) duration_running.count());

        // compute duality measure
        mu = compute_mu(s_, u_);

        // iterate 
        k++;
    }

    // timing
    auto timer_final = std::chrono::high_resolution_clock::now();
    auto duration_final = std::chrono::duration_cast<std::chrono::microseconds>(timer_final - timer_init);
    double time = 1e-6 * ((double) duration_final.count());

    // assemble results
    Result results;
    results.x = x_;
    results.v = v_;
    results.u = u_;
    results.s = s_;
    results.objective = objective(x_);
    results.feasible = mu <= settings_.mu_feas; // divergence check
    results.converged = !numerical_issue && (k < settings_.iter_max) && results.feasible;
    results.num_iter = k;
    results.sol_time = time;

    // return
    return results;
}

/* helper methods */

// objective function
double Solver::objective(const Eigen::Ref<const Eigen::VectorXd> x_in)
{
    return 0.5*x_in.dot(P_*x_in) + q_.dot(x_in);
}

// initialize working matrices
void Solver::initialize_working_matrices()
{
    // precompute transposes
    A_T_ = A_.transpose();
    G_T_ = G_.transpose();

    // initialize working matrices
    bm_.resize(n_ + m_eq_ + m_ineq_ + m_ineq_);

    // precompute constant part of M matrix
    // M = [P, A', G', 0;
    //      A, 0, 0, 0;
    //      G, 0, 0, I;
    //      0, 0, S, Z]
    std::vector<Eigen::Triplet<double>> tripvec_M0;
    tripvec_M0.reserve(P_.nonZeros() + 2*A_.nonZeros() + 2*G_.nonZeros() + m_ineq_);
    M0_.resize(n_ + m_eq_ + 2*m_ineq_, n_ + m_eq_ + 2*m_ineq_);

    getTripletsForMatrix(&P_, tripvec_M0, 0, 0);
    if (equalityConstrained_)
        getTripletsForMatrix(&A_T_, tripvec_M0, 0, n_);
    getTripletsForMatrix(&G_T_, tripvec_M0, 0, n_ + m_eq_);

    if (equalityConstrained_)
        getTripletsForMatrix(&A_, tripvec_M0, n_, 0);

    getTripletsForMatrix(&G_, tripvec_M0, n_ + m_eq_, 0);
    getTripletsForMatrixDiagonal(Eigen::VectorXd::Ones(m_ineq_), tripvec_M0, n_ + m_eq_, n_ + m_eq_ + m_ineq_);

    M0_.setFromTriplets(tripvec_M0.begin(), tripvec_M0.end());

    // pre-allocate and initialize dM
    triplets_.clear();
    triplets_.reserve(2*m_ineq_);
    dM_.resize(n_ + m_eq_ + 2*m_ineq_, n_ + m_eq_ + 2*m_ineq_);
    getTripletsForMatrixDiagonal(s_, triplets_, n_ + m_eq_ + m_ineq_, n_ + m_eq_);
    getTripletsForMatrixDiagonal(u_, triplets_, n_ + m_eq_ + m_ineq_, n_ + m_eq_ + m_ineq_);
    dM_.setFromTriplets(triplets_.begin(), triplets_.end());

    // analyze pattern
    M_ = M0_ + dM_;
    lu_solver_.analyzePattern(M_);
}

// generate system matrix
void Solver::generate_system_matrix()
{
    // update changing part of system matrix
    // update dM based on known sparsity pattern
    int i_s = 0;
    int i_u = 0;
    for (int k=0; k<dM_.outerSize(); k++)
    {
        for (typename Eigen::SparseMatrix<double>::InnerIterator it(dM_,k); it; ++it)
        {
            if (i_s < s_.rows())
            {
                it.valueRef() = s_(i_s);
                i_s++;
            }
            else if (i_u < u_.rows())
            {
                it.valueRef() = u_(i_u);
                i_u++;
            }
        }
    }
    
    // update M
    M_ = M0_ + dM_;

    // LU decomposition
    lu_solver_.factorize(M_);

    // get status
    lu_status_ = lu_solver_.info();

}

// generate right hand side of linear system
void Solver::generate_rhs()
{
    // compute r_E term
    if (equalityConstrained_)
    {
        // compute r_C term
        r_C_ = P_*x_ + q_ + A_T_*v_ + G_T_*u_;

        // compute r_E term
        r_E_ = A_*x_ - b_;
    }
    else
        r_C_ = P_*x_ + q_ + G_T_*u_;

    // compute r_I term
    r_I_ = (G_*x_ - w_) + s_;

    // compute r_S term
    S_.diagonal() = s_;
    r_S_ = S_*u_; // no centering term

    // RHS
    bm_.segment(0, n_) = -r_C_;
    if (equalityConstrained_)
        bm_.segment(n_, m_eq_) = -r_E_;
    bm_.segment(n_+m_eq_, m_ineq_) = -r_I_;
    bm_.segment(n_+m_eq_+m_ineq_, m_ineq_) = -r_S_;
}

// update RHS
void Solver::update_rhs()
{
    // update r_S term
    r_S_ = r_S_ - nu_;
    
    // update RHS
    bm_.segment(n_+m_eq_+m_ineq_, m_ineq_) = -r_S_;
}

// line search
std::pair<double, bool> Solver::line_search(const Eigen::Ref<const Eigen::VectorXd> del_s, const Eigen::Ref<const Eigen::VectorXd> del_u)
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
        s_ds = s_ + h*del_s;
        u_du = u_ + h*del_u;

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
double Solver::compute_mu(const Eigen::Ref<const Eigen::VectorXd> s_in, const Eigen::Ref<const Eigen::VectorXd> u_in)
{
    return (s_in.dot(u_in))/m_ineq_;
}

// pre-processing
void Solver::get_permute_matrix(const Eigen::SparseMatrix<double> * mat_in, Eigen::SparseMatrix<double> * mat_out)
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
    qr_solver_.analyzePattern(mat_T);
    qr_solver_.factorize(mat_T);

    // get permutation matrix and its indices
    Eigen::PermutationMatrix<-1, -1> P_full = qr_solver_.colsPermutation();
    Eigen::VectorXi ind_full = P_full.indices();

    // construct permute and chop matrix directly
    std::vector<Eigen::Triplet<double>> tripvec;
    tripvec.reserve(ind_full.size());
    for (int i=0; i<qr_solver_.rank(); i++) // QR solver automatically puts linearly dependent rows at end
        tripvec.push_back(Eigen::Triplet<double>(ind_full[i], i, 1));

    mat_out->resize(mat_T.cols(), qr_solver_.rank());
    mat_out->setFromTriplets(tripvec.begin(), tripvec.end());
}

bool Solver::compute_problem_dimensions()
{
    // get dimension variables
    n_ = P_.rows();
    m_eq_ = A_.rows();
    m_ineq_ = G_.rows();
    equalityConstrained_ = (m_eq_ == 0) ? false : true;

    // check validity
    const bool dims_consistent = n_ == q_.size() && n_ == P_.cols() && n_ == A_.cols() && n_ == G_.cols() 
        && m_eq_ == b_.size() && m_ineq_ == w_.size();
    const bool dims_valid = n_ > 0 && m_eq_ >= 0 && m_ineq_ >= 0;
    return dims_consistent && dims_valid;
}

void Solver::make_valid_equality_constraints()
{
    // remove any invalid constraints (linearly dependent)
    get_permute_matrix(&A_, &P_eq_);
}

void Solver::make_valid_A()
{
    A_ = (A_.transpose()*P_eq_).transpose();
}

void Solver::make_valid_b()
{
    b_ = (b_.transpose()*P_eq_).transpose();
}

