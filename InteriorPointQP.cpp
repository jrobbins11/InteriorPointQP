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
        for (int i=0; i<mat.outerSize(); ++i)
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
        for (int i=0; i<static_cast<int>(d.rows()); ++i)
        {
            triplets.emplace_back(i+row_offset, i+col_offset, d(i));
        }
    }

    // constants
    constexpr double EPSILON = Eigen::NumTraits<double>::dummy_precision();
}

// constructor
Solver::Solver(
    Eigen::SparseMatrix<double> P,
    Eigen::VectorXd q,
    std::optional<Eigen::SparseMatrix<double>> G,
    std::optional<Eigen::VectorXd> w,
    std::optional<Eigen::SparseMatrix<double>> A,
    std::optional<Eigen::VectorXd> b,
    std::optional<Settings> settings    
)
{    
    // copy in data, check for default-constructed arguments
    P_ = P;
    q_ = q;
    G_ = G.has_value() ? G.value() : Eigen::SparseMatrix<double>(0, q.size());
    w_ = w.has_value() ? w.value() : Eigen::VectorXd::Zero(0);
    A_ = A.has_value() ? A.value() : Eigen::SparseMatrix<double>(0, q.size());
    b_ = b.has_value() ? b.value() : Eigen::VectorXd::Zero(0);
    settings_ = settings.has_value() ? settings.value() : Settings{};    

    // preprocessing (make sure A is full rank)
    remove_linearly_dependent_equality_constraints();

    // compute problem dimensions
    if (!check_problem_dimensions())
    {
        throw std::invalid_argument("Inconsistent problem dimensions");
    }
    if (!validate_settings(settings_))
    {
        throw std::invalid_argument("Settings invalid.");
    }
}

void Solver::update_settings(const Settings& settings) 
{
    if (!validate_settings(settings))
    {
        throw std::invalid_argument("Settings invalid.");
    }
    settings_ = settings;
}

// solve
Result Solver::solve()
{
    // start timer
    auto timer_init = std::chrono::high_resolution_clock::now();

    // check for early exit (equality constrained)
    if (G_.rows() == 0)
    {
        return solve_equality_constrained();
    }

    // running timer
    double running_timer = 0.0; // init

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
    double mu = compute_mu(s_, u_);

    // loop
    while ((mu > settings_.barrier_converged || !is_feasible(x_)) && (k < settings_.max_iterations) && 
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
        const double mu_pred = compute_mu(s_ + h_pred*del_s_pred, u_ + h_pred*del_u_pred);

        // centering parameter
        const double sigma = std::pow(mu_pred/mu, 3);

        // calculate corrected nu and recompute search direction
        update_rhs((sigma*mu)*Eigen::ArrayXd::Ones(m_ineq_) - del_s_pred.array() * del_u_pred.array());

        const Eigen::VectorXd del_corr = lu_solver_.solve(bm_);
        const auto& del_x_corr = del_corr.segment(0, n_);
        const auto& del_v_corr = del_corr.segment(n_, m_eq_);
        const auto& del_u_corr = del_corr.segment(n_+m_eq_, m_ineq_);
        const auto& del_s_corr = del_corr.segment(n_+m_eq_+m_ineq_, m_ineq_);

        // line search for step size
        const auto [h_corr, no_progress_corr] = line_search(del_s_corr, del_u_corr);
        numerical_issue |= no_progress_corr;

        // update
        x_ += settings_.step_scaling*h_corr*del_x_corr;
        v_ += settings_.step_scaling*h_corr*del_v_corr;
        u_ += settings_.step_scaling*h_corr*del_u_corr;
        s_ += settings_.step_scaling*h_corr*del_s_corr;

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
    const auto timer_final = std::chrono::high_resolution_clock::now();
    const auto duration_final = std::chrono::duration_cast<std::chrono::microseconds>(timer_final - timer_init);

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
    results.solution_time_sec = 1e-6 * static_cast<double>(duration_final.count());

    // return
    return results;
}

// special case: no inequality constraints
Result Solver::solve_equality_constrained()
{
    // start timer
    auto timer_init = std::chrono::high_resolution_clock::now();

    // initialize primal and dual vars
    x_ = Eigen::VectorXd::Zero(n_);
    v_ = Eigen::VectorXd::Zero(m_eq_);
    u_.resize(0);
    s_.resize(0);

    // solve linear system
    initialize_working_matrices();
    generate_rhs();
    lu_solver_.factorize(M0_ + dM_);
    const bool factorized = lu_status_ == Eigen::ComputationInfo::Success;

    Eigen::VectorXd x_v;
    if (factorized)
    {
        x_v = lu_solver_.solve(bm_);
        x_ = x_v.segment(0, n_);
        v_ = x_v.segment(n_, m_eq_);
    }

    // timing
    const auto timer_final = std::chrono::high_resolution_clock::now();
    const auto duration_final = std::chrono::duration_cast<std::chrono::microseconds>(timer_final - timer_init);

    // assemble results
    Result results;
    results.solution = x_;
    results.dual_solution_v = v_;
    results.dual_solution_u = u_;
    results.slack_solution_s = s_;
    results.objective = objective(x_);
    results.feasible = is_feasible(x_);
    results.converged = factorized;
    results.num_iterations = 0;
    results.solution_time_sec = 1e-6 * static_cast<double>(duration_final.count());

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
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(P_.nonZeros() + 2*A_.nonZeros() + 2*G_.nonZeros() + m_ineq_);
    M0_.resize(n_ + m_eq_ + 2*m_ineq_, n_ + m_eq_ + 2*m_ineq_);

    get_triplets(P_, triplets, 0, 0);
    get_triplets(A_T_, triplets, 0, n_);
    get_triplets(G_T_, triplets, 0, n_ + m_eq_);

    get_triplets(A_, triplets, n_, 0);

    get_triplets(G_, triplets, n_ + m_eq_, 0);
    get_triplets_diagonal(Eigen::VectorXd::Ones(m_ineq_), triplets, n_ + m_eq_, n_ + m_eq_ + m_ineq_);

    M0_.setFromTriplets(triplets.begin(), triplets.end());

    // initialize dM
    triplets.clear();
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
                ++i_s;
            }
            else if (i_u < u_.rows())
            {
                it.valueRef() = u_(i_u);
                ++i_u;
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
    bm_.segment(0, n_) = -(P_*x_ + q_ + A_T_*v_ + G_T_*u_);
    bm_.segment(n_, m_eq_) = -(A_*x_ - b_);
    bm_.segment(n_+m_eq_, m_ineq_) = -((G_*x_ - w_) + s_);
    bm_.segment(n_+m_eq_+m_ineq_, m_ineq_) = -(s_.array() * u_.array()); // elementwise multiplication, no centering term
}

// update RHS
void Solver::update_rhs(const Eigen::Ref<const Eigen::ArrayXd> nu)
{   
    // update RHS
    bm_.segment(n_+m_eq_+m_ineq_, m_ineq_) = -(s_.array() * u_.array() - nu);
}

// line search
std::pair<double, bool> Solver::line_search(const Eigen::Ref<const Eigen::VectorXd> del_s, const Eigen::Ref<const Eigen::VectorXd> del_u)
{
    // init
    double h = 1;
    bool valid = false;
    int cnt = 0;

    // loop
    while (!valid && (cnt <  settings_.line_search_max_iterations) && (h > EPSILON))
    {
        // updated s, u
        const Eigen::VectorXd s_ds = s_ + h*del_s;
        const Eigen::VectorXd u_du = u_ + h*del_u;

        // check for validity of step
        if ((s_ds.minCoeff() > 0) && (u_du.minCoeff() > 0))
            valid = true;
        else
            h = settings_.line_search_step_factor*h;

        // increment
        ++cnt;
    }

    // check for validity
    const bool no_progress = (cnt ==  settings_.line_search_max_iterations) || (h <= EPSILON);

    // output
    return {h, no_progress};
}

// duality measure 
double Solver::compute_mu(const Eigen::Ref<const Eigen::VectorXd> s_in, const Eigen::Ref<const Eigen::VectorXd> u_in) const
{
    return (s_in.dot(u_in))/m_ineq_;
}

bool Solver::check_problem_dimensions()
{
    // get dimension variables
    n_ = static_cast<int>(P_.rows());
    m_eq_ = static_cast<int>(A_.rows());
    m_ineq_ = static_cast<int>(G_.rows());

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
    qr_solver.compute(mat_T);

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
    const bool equality_cons_feasible = m_eq_ > 0 ? (A_*x - b_).cwiseAbs().maxCoeff() < settings_.feasibility_tolerance : true;
    const bool inequality_cons_feasible = m_ineq_ > 0 ? (G_*x - w_).maxCoeff() < settings_.feasibility_tolerance : true;
    return equality_cons_feasible && inequality_cons_feasible;
}

bool Solver::validate_settings(const Settings& settings) const
{
    const bool time_valid = settings.max_time_sec > 0.;
    const bool barrier_valid = settings.barrier_init > 0. &&
                               settings.barrier_converged > 0. &&
                               settings.barrier_max > 0. &&
                               settings.barrier_converged < settings.barrier_init &&
                               settings.barrier_init < settings.barrier_max;
    const bool tol_valid = settings.feasibility_tolerance > 0.;
    const bool iterations_valid = settings.max_iterations > 0;
    const bool step_valid = settings.step_scaling > 0. && settings.step_scaling < 1.;
    const bool line_search_valid = settings.line_search_step_factor > 0. &&
                                   settings.line_search_step_factor < 1. &&
                                   settings.line_search_max_iterations > 0;
    return time_valid && barrier_valid && tol_valid && iterations_valid && step_valid && line_search_valid;
}

std::ostream& operator<<(std::ostream& os, const Settings& settings)
{
    os << "InteriorPointQP Settings: " << std::endl;
    os << " max_time_sec: " << settings.max_time_sec << std::endl;
    os << " barrier_init: " << settings.barrier_init << std::endl;
    os << " barrier_max: " << settings.barrier_max << std::endl;
    os << " barrier_terminal: " << settings.barrier_converged << std::endl;
    os << " feasibility_tolerance: " << settings.feasibility_tolerance << std::endl;
    os << " max_iterations: " << settings.max_iterations << std::endl;
    os << " step_scaling: " << settings.step_scaling << std::endl;
    os << " line_search_step_factor: " << settings.line_search_step_factor << std::endl;
    os << " line_search_max_iterations: " << settings.line_search_max_iterations << std::endl;
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
    os << " num_iterations: " << result.num_iterations << std::endl;
    os << " solution_time_sec: " << result.solution_time_sec << " s" << std::endl;
    return os;
}