#include "InteriorPointQP.hpp"

#include <chrono>
#include <stdexcept>
#include <cmath>

using namespace InteriorPointQP;

namespace 
{
    // utilities
    void get_triplets(const Eigen::SparseMatrix<double>& mat, std::vector<Eigen::Triplet<double>> &triplets,
        int row_offset, int col_offset)
    {
        if (row_offset < 0 || col_offset < 0)
        {
            throw std::invalid_argument("get_triplets: offsets out of range");
        }       
        for (int i=0; i<mat.outerSize(); i++)
        {
            for (typename Eigen::SparseMatrix<double>::InnerIterator it(mat,i); it; ++it)
                triplets.emplace_back(it.row()+row_offset, it.col()+col_offset, it.value());
        }
    }

    void get_triplets_diagonal(const Eigen::Ref<const Eigen::VectorXd> d, std::vector<Eigen::Triplet<double>> &triplets,
                int row_offset, int col_offset)
    {
        if (row_offset < 0 || col_offset < 0)
        {
            throw std::invalid_argument("get_triplets: offsets out of range");
        }     
        int m = d.rows();
        for (int i=0; i<m; i++)
        {
            triplets.emplace_back(i+row_offset, i+col_offset, d(i));
        }
    }

    // constants
    constexpr double EPSILON = Eigen::NumTraits<double>::dummy_precision();
}

std::ostream& operator<<(std::ostream& os, const Settings& settings)
{
    os << "InteriorPointQP Settings: " << std::endl;
    os << " max_time_sec: " << settings.max_time_sec << std::endl;
    os << " barrier_init: " << settings.barrier_init << std::endl;
    os << " barrier_max: " << settings.barrier_max << std::endl;
    os << " barrier_terminal: " << settings.barrier_terminal << std::endl;
    os << " feasibility_tolerance: " << settings.feasibility_tolerance << std::endl;
    os << " max_iterations: " << settings.max_iterations << std::endl;
    os << " line_search_gamma: " << settings.line_search_gamma << std::endl;
    os << " line_search_t: " << settings.line_search_t << std::endl;
    return os;
}

std::ostream& operator<<(std::ostream& os, const Result& result)
{
    os << "InteriorPointQP Result: " << std::endl;
    if (result.solution.size() < 10)
    {
        os << " solution: " << result.solution;
        os << " dual_solution_v: " << result.dual_solution_v;
        os << " dual_solution_u: " << result.dual_solution_u;
        os << " slack_solution_s: " << result.slack_solution_s;
    }
    os << " objective: " << result.objective << std::endl;
    os << " converged: " << result.converged << std::endl;
    os << " feasible: " << result.feasible << std::endl;
    os << " num_iteratins: " << result.num_iterations << std::endl;
    os << " solution_time_sec: " << result.solution_time_sec << " s" << std::endl;
    return os;
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
    remove_linearly_dependent_equality_constraints();

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
    double mu, mu_pred, sigma;

    // start timer
    auto timer_init = std::chrono::high_resolution_clock::now();

    // running timer
    double running_timer = 0.9; // init

    // initialize primal and dual vars
    const double zeta = std::sqrt(settings_.barrier_init);
    x_ = Eigen::VectorXd::Zero(n_);
    v_ = Eigen::VectorXd::Zero(m_eq_);
    u_ = zeta*Eigen::VectorXd::Ones(m_ineq_);
    s_ = zeta*Eigen::VectorXd::Ones(m_ineq_);

    // initialize working matrices
    initialize_working_matrices();

    // outer loop init
    int k = 0;
    bool numerical_issue = false;

    // compute duality measure
    mu = compute_mu(s_, u_);

    // loop
    while ((mu > settings_.barrier_terminal) && (k < settings_.max_iterations) && 
        !numerical_issue && (running_timer < settings_.max_time_sec) && (mu <= settings_.barrier_max))
    {
        // generate system matrix and decompose
        generate_system_matrix();

        // check for numerical issues and terminate if necessary
        numerical_issue |= (lu_status_ != Eigen::ComputationInfo::Success);

        // predictor step
        generate_rhs();
        const Eigen::VectorXd del_pred = lu_solver_.solve(bm_);
        const auto& del_u_pred = del_pred.segment(n_+m_eq_, m_ineq_);
        const auto& del_s_pred = del_pred.segment(n_+m_eq_+m_ineq_, m_ineq_);

        // line search for step size
        const auto [h_pred, no_progress_pred] = line_search(del_s_pred, del_u_pred);
        numerical_issue |= no_progress_pred;

        // predicted duality measure
        mu_pred = compute_mu(s_ + h_pred*del_s_pred, u_ + h_pred*del_u_pred);

        // centering parameter
        sigma = pow(mu_pred/mu, 3);

        // calculate corrected nu and recompute search direction
        Del_S_.diagonal() = del_s_pred;
        update_rhs((sigma*mu)*Eigen::VectorXd::Ones(m_ineq_) - Del_S_*del_u_pred);

        const Eigen::VectorXd del_corr = lu_solver_.solve(bm_);
        const auto& del_x_corr = del_corr.segment(0, n_);
        const auto& del_v_corr = del_corr.segment(n_, m_eq_);
        const auto& del_u_corr = del_corr.segment(n_+m_eq_, m_ineq_);
        const auto& del_s_corr = del_corr.segment(n_+m_eq_+m_ineq_, m_ineq_);

        // line search for step size
        const auto [h_corr, no_progress_corr] = line_search(del_s_corr, del_u_corr);
        numerical_issue |= no_progress_corr;

        // update
        x_ += settings_.line_search_gamma*h_corr*del_x_corr;
        v_ += settings_.line_search_gamma*h_corr*del_v_corr;
        u_ += settings_.line_search_gamma*h_corr*del_u_corr;
        s_ += settings_.line_search_gamma*h_corr*del_s_corr;

        // update running timer
        auto timer_running = std::chrono::high_resolution_clock::now();
        auto duration_running = std::chrono::duration_cast<std::chrono::microseconds>(timer_running - timer_init);
        running_timer = 1e-6 * ((double) duration_running.count());

        // compute duality measure
        mu = compute_mu(s_, u_);

        // iterate 
        ++k;
    }

    // timing
    auto timer_final = std::chrono::high_resolution_clock::now();
    auto duration_final = std::chrono::duration_cast<std::chrono::microseconds>(timer_final - timer_init);
    double time = 1e-6 * ((double) duration_final.count());

    // assemble results
    Result results;
    results.solution = x_;
    results.dual_solution_v = v_;
    results.dual_solution_u = u_;
    results.slack_solution_s = s_;
    results.objective = objective(x_);
    results.feasible = is_feasible(x_);
    results.converged = !numerical_issue && (k < settings_.max_iterations);
    results.num_iterations = k;
    results.solution_time_sec = time;

    // return
    return results;
}

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

    get_triplets(P_, tripvec_M0, 0, 0);
    if (equality_constrained_)
        get_triplets(A_T_, tripvec_M0, 0, n_);
    get_triplets(G_T_, tripvec_M0, 0, n_ + m_eq_);

    if (equality_constrained_)
        get_triplets(A_, tripvec_M0, n_, 0);

    get_triplets(G_, tripvec_M0, n_ + m_eq_, 0);
    get_triplets_diagonal(Eigen::VectorXd::Ones(m_ineq_), tripvec_M0, n_ + m_eq_, n_ + m_eq_ + m_ineq_);

    M0_.setFromTriplets(tripvec_M0.begin(), tripvec_M0.end());

    // pre-allocate and initialize dM
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.clear();
    triplets.reserve(2*m_ineq_);
    dM_.resize(n_ + m_eq_ + 2*m_ineq_, n_ + m_eq_ + 2*m_ineq_);
    get_triplets_diagonal(s_, triplets, n_ + m_eq_ + m_ineq_, n_ + m_eq_);
    get_triplets_diagonal(u_, triplets, n_ + m_eq_ + m_ineq_, n_ + m_eq_ + m_ineq_);
    dM_.setFromTriplets(triplets.begin(), triplets.end());

    // analyze pattern
    lu_solver_.analyzePattern(M0_ + dM_);
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

    // LU decomposition
    lu_solver_.factorize(M0_ + dM_);

    // get status
    lu_status_ = lu_solver_.info();

}

// generate right hand side of linear system
void Solver::generate_rhs()
{
    // compute r_E term
    Eigen::VectorXd r_C, r_E;
    if (equality_constrained_)
    {
        // compute r_C term
        r_C = P_*x_ + q_ + A_T_*v_ + G_T_*u_;

        // compute r_E term
        r_E = A_*x_ - b_;
    }
    else
    {
        r_C = P_*x_ + q_ + G_T_*u_;
    }

    // compute r_I term
    const Eigen::VectorXd r_I = (G_*x_ - w_) + s_;

    // compute r_S term
    S_.diagonal() = s_;
    r_S_ = S_*u_; // no centering term

    // RHS
    bm_.segment(0, n_) = -r_C;
    if (equality_constrained_)
        bm_.segment(n_, m_eq_) = -r_E;
    bm_.segment(n_+m_eq_, m_ineq_) = -r_I;
    bm_.segment(n_+m_eq_+m_ineq_, m_ineq_) = -r_S_;
}

// update RHS
void Solver::update_rhs(const Eigen::Ref<const Eigen::VectorXd> nu)
{
    // update r_S term
    r_S_ -= nu;
    
    // update RHS
    bm_.segment(n_+m_eq_+m_ineq_, m_ineq_) = -r_S_;
}

// line search
std::pair<double, bool> Solver::line_search(const Eigen::Ref<const Eigen::VectorXd> del_s, const Eigen::Ref<const Eigen::VectorXd> del_u)
{
    // init
    double h = 1;
    bool valid = false;
    int cnt_max = settings_.line_search_max_iterations;
    int cnt = 0;

    // loop
    while (!valid && (cnt < cnt_max) && (h > EPSILON))
    {
        // updated s, u
        const Eigen::VectorXd s_ds = s_ + h*del_s;
        const Eigen::VectorXd u_du = u_ + h*del_u;

        // check for validity of step
        if ((s_ds.minCoeff() > 0) && (u_du.minCoeff() > 0))
            valid = true;
        else
            h = settings_.line_search_t*h;

        // increment
        ++cnt;
    }

    // check for validity
    const bool no_progress = ((cnt == cnt_max) || (h <= EPSILON));

    // output
    return {h, no_progress};
}

// duality measure 
double Solver::compute_mu(const Eigen::Ref<const Eigen::VectorXd> s_in, const Eigen::Ref<const Eigen::VectorXd> u_in) const
{
    return (s_in.dot(u_in))/m_ineq_;
}

bool Solver::compute_problem_dimensions()
{
    // get dimension variables
    n_ = P_.rows();
    m_eq_ = A_.rows();
    m_ineq_ = G_.rows();
    equality_constrained_ = m_eq_ > 0;
    inequality_constrained_ = m_ineq_ > 0;

    // check validity
    const bool dims_consistent = n_ == q_.size() && n_ == P_.cols() && n_ == A_.cols() && n_ == G_.cols() 
        && m_eq_ == b_.size() && m_ineq_ == w_.size();
    const bool dims_valid = n_ > 0 && m_eq_ >= 0 && m_ineq_ >= 0;
    return dims_consistent && dims_valid;
}

void Solver::remove_linearly_dependent_equality_constraints()
{
    // check for empty input matrix
    if (A_.rows() == 0 || A_.cols() == 0)
    {
        return;
    }

    // matrix transpose
    Eigen::SparseMatrix<double> mat_T = A_.transpose();

    // compute QR decomposition
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr_solver;
    qr_solver.analyzePattern(mat_T);
    qr_solver.factorize(mat_T);

    // get permutation matrix and its indices
    Eigen::PermutationMatrix<-1, -1> P_full = qr_solver.colsPermutation();
    Eigen::VectorXi ind_full = P_full.indices();

    // construct permutation matrix
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(ind_full.size());
    for (int i=0; i<qr_solver.rank(); i++) // QR solver automatically puts linearly dependent rows at end
        triplets.emplace_back(ind_full[i], i, 1);

    Eigen::SparseMatrix<double> P_eq (mat_T.cols(), qr_solver.rank());
    P_eq.setFromTriplets(triplets.begin(), triplets.end());

    // update equality constraints
    A_ = (A_.transpose()*P_eq).transpose();
    b_ = (b_.transpose()*P_eq).transpose();
}

bool Solver::is_feasible(const Eigen::Ref<const Eigen::VectorXd> x) const
{
    const bool equality_cons_feasible = equality_constrained_ ? (A_*x - b_).cwiseAbs().maxCoeff() < settings_.feasibility_tolerance : true;
    const bool inequality_cons_feasible = inequality_constrained_ ? (G_*x - w_).maxCoeff() < settings_.feasibility_tolerance : true;
    return equality_cons_feasible && inequality_cons_feasible;
}
