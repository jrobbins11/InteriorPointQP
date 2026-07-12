#include "InteriorPointQP.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/src/SparseCore/SparseMatrix.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <random>
#include <iostream>

using namespace InteriorPointQP;

// test fixture
class RandomQP : public ::testing::Test
{
protected:
    void SetUp() override 
    {
        // init fields
        distr = std::uniform_real_distribution<double>(1, 10.);

        // random unconstrained QP
        std::vector<Eigen::Triplet<double>> triplets;
        q.resize(n);
        for (int i=0; i<n; ++i) 
        {
            triplets.emplace_back(i, i, distr(gen));
            q(i) = distr(gen);
        }
        P.resize(n, n);
        P.setFromTriplets(triplets.begin(), triplets.end());
    }

    void TearDown() override
    {
    }

    std::mt19937 gen {42};
    std::uniform_real_distribution<double> distr;
    const int n = 50;
    Eigen::SparseMatrix<double> P;
    Eigen::VectorXd q;
};

// Test that problem dimension checking works
TEST(InteriorPointQP, InvalidProblemDimensions)
{
    // constructor with empty P and q matrices should throw
    EXPECT_THROW(Solver(Eigen::SparseMatrix<double>{}, Eigen::VectorXd{}), std::invalid_argument);

    // constructor with inconsistent dimensions should throw
    EXPECT_THROW(Solver(Eigen::SparseMatrix<double>(5, 5),
                        Eigen::VectorXd::Zero(5),
                        Eigen::SparseMatrix<double>(3, 4),
                        Eigen::VectorXd::Zero(3)),
                    std::invalid_argument);

    // constructor with valid dimensions should not throw
    EXPECT_NO_THROW(Solver(Eigen::SparseMatrix<double>(5, 5),
                        Eigen::VectorXd::Zero(5),
                        Eigen::SparseMatrix<double>(3, 5),
                        Eigen::VectorXd::Zero(3)));
}

// Check that perturbations around the optimal solution for a nearly unconstrained problem have higher cost than solver solution
TEST_F(RandomQP, NearlyUnconstrainedOptimal)
{
    // do-nothing inequality constraints
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.clear();
    for (int i=0; i<n; ++i)
    {
        triplets.emplace_back(i, i, 1.0);
    }
    Eigen::SparseMatrix<double> G (n, n);
    G.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::VectorXd w = 1e6 * Eigen::VectorXd::Ones(n);

    // solve QP
    Solver solver(P, q, G, w);
    const Result result = solver.solve();

    // perturb solution and get objective
    auto perturbed_objective = [&](const Eigen::VectorXd& x, double pert_size) -> double 
    {
        // perturbed point
        std::uniform_real_distribution<double> pert_dist(-pert_size, pert_size);
        Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);
        for (int i=0; i<n; ++i)
        {
            dx(i) = pert_dist(gen);
        }
        const Eigen::VectorXd x_dx = x + dx;

        // get objective
        return 0.5*x_dx.dot(P*x_dx) + q.dot(x_dx);
    };

    // check that random perturbations have higher objective
    for (int i=0; i<100; ++i)
    {
        const double result_norm = result.solution.norm();
        EXPECT_LT(result.objective, perturbed_objective(result.solution, 0.01*result_norm));
    }
}

// Unconstrained produces optimal solution
TEST(InteriorPointQP, UnconstrainedOptimal)
{
    // Problem: min 0.5 * (x-xr)^T (x-xr) -> solution is xr
    // q = xr

    // random xr
    std::mt19937 gen (42);
    std::uniform_real_distribution<double> distr(1., 10.);
    const int n = 50;
    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
    for (int i=0; i<n; ++i) 
    {
        triplets.emplace_back(i, i, 1.0);
        q(i) = distr(gen);
    }
    Eigen::SparseMatrix<double> P (n, n);
    P.setFromTriplets(triplets.begin(), triplets.end());

    // solve QP
    Solver solver(P, q);
    const Result result = solver.solve();
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.feasible);

    // check solution matches expectation
    for (int i=0; i<n; ++i)
    {
        EXPECT_NEAR(result.solution(i), q(i), 1e-6);
    }
}

// Infeasible problem produces infeasible solution
TEST_F(RandomQP, InfeasibleProblem)
{
    const double bound = 10.;

    // Add box constraints
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.clear();
    for (int i=0; i<n; ++i)
    {
        triplets.emplace_back(i, i, 1.0);
        triplets.emplace_back(i+n, i, -1.0);
    }
    Eigen::SparseMatrix<double> G (2*n, n);
    G.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::VectorXd w = bound * Eigen::VectorXd::Ones(2*n);

    // Add some incompatible equality constraints
    const int n_eq = n/4;
    triplets.clear();
    for (int i=0; i<n_eq; ++i)
    {
        triplets.emplace_back(i, i, 1.0);
    }
    Eigen::SparseMatrix<double> A (n_eq, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::VectorXd b = 1.01 * bound * Eigen::VectorXd::Ones(n_eq);

    // solve QP, check infeasible
    Solver solver(P, q, G, w, A, b);
    const Result result = solver.solve();
    EXPECT_FALSE(result.feasible);
}

// Feasible equality-constrained problem produces feasible solution
TEST_F(RandomQP, FeasibleProblem)
{
    const double bound = 10.;

    // Add box constraints
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.clear();
    for (int i=0; i<n; ++i)
    {
        triplets.emplace_back(i, i, 1.0);
        triplets.emplace_back(i+n, i, -1.0);
    }
    Eigen::SparseMatrix<double> G (2*n, n);
    G.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::VectorXd w = bound * Eigen::VectorXd::Ones(2*n);

    // Add some incompatible equality constraints
    const int n_eq = n/4;
    triplets.clear();
    for (int i=0; i<n_eq; ++i)
    {
        triplets.emplace_back(i, i, 1.0);
    }
    Eigen::SparseMatrix<double> A (n_eq, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::VectorXd b = 0.99 * bound * Eigen::VectorXd::Ones(n_eq);

    // solve QP, check infeasible
    Solver solver(P, q, G, w, A, b);
    const Result result = solver.solve();
    EXPECT_TRUE(result.feasible);
    EXPECT_TRUE(result.converged);
}

// Correctness checking: Hock–Schittkowski hs21
TEST(InteriorPointQP, ResultCorrectnessHS21)
{
    // https://apmonitor.com/wiki/uploads/Apps/hs021.apm

    // objective
    Eigen::SparseMatrix<double> P (2,2);
    P.insert(0, 0) = 1./100.;
    P.insert(1, 1) = 1.;
    P *= 2.; // correcting for 1/2 factor

    Eigen::VectorXd q = Eigen::VectorXd::Zero(2);

    // inequality constraints
    Eigen::SparseMatrix<double> G (5, 2);
    Eigen::VectorXd w (5);

    // 10 x0 - x1 >= 10
    G.insert(0, 0) = -10.;
    G.insert(0, 1) = 1.;
    w(0) = -10.;

    // 2 <= x0 <= 50
    G.insert(1, 0) = -1.;
    w(1) = -2.;
    G.insert(2, 0) = 1.;
    w(2) = 50.;

    // -50 <= x1 <= 50
    G.insert(3, 1) = -1.;
    w(3) = 50.;
    G.insert(4, 1) = 1.;
    w(4) = 50.;

    // solve QP
    Solver solver(P, q, G, w);
    const Result result = solver.solve();
    ASSERT_TRUE(result.feasible);
    ASSERT_TRUE(result.converged);

    // check objective against known value
    EXPECT_NEAR(result.objective - 100., -99.96, 1e-2);
}

// Correctness checking: Hock–Schittkowski hs35
TEST(InteriorPointQP, ResultCorrectnessHS35)
{
    // https://apmonitor.com/wiki/uploads/Apps/hs035.apm

    // objective
    Eigen::SparseMatrix<double> P (3,3);
    P.insert(0,0) = 2.;
    P.insert(1,1) = 2.;
    P.insert(2,2) = 1.;
    P.insert(0, 1) = 1.;
    P.insert(1, 0) = 1.;
    P.insert(0, 2) = 1.;
    P.insert(2, 0) = 1.;
    P *= 2.; // correcting for 1/2 factor

    Eigen::VectorXd q (3);
    q(0) = -8.;
    q(1) = -6.;
    q(2) = -4.;

    // inequality constraints
    Eigen::SparseMatrix<double> G(4, 3);
    Eigen::VectorXd w(4);

    // all variables >= 0
    G.insert(0,0) = -1.;
    w(0) = 0.;
    G.insert(1,1) = -1.;
    w(1) = 0.;
    G.insert(2,2) = -1.;
    w(2) = 0.;

    // x0 + x1 + 2x2 <= 3
    G.insert(3,0) = 1.;
    G.insert(3,1) = 1.;
    G.insert(3,2) = 2.;
    w(3) = 3.;

    // solve QP
    Solver solver(P, q, G, w);
    const Result result = solver.solve();
    ASSERT_TRUE(result.feasible);
    ASSERT_TRUE(result.converged);

    // check objective against known value
    EXPECT_NEAR(result.objective + 9., 1./9., 1e-2);
}

// Correctness checking: Hock–Schittkowski hs76
TEST(InteriorPointQP, ResultCorrectnessHS76)
{
    // https://apmonitor.com/wiki/uploads/Apps/hs076.apm

    // objective
    Eigen::SparseMatrix<double> P (4,4);
    P.insert(0,0) = 1.;
    P.insert(1,1) = 0.5;
    P.insert(2,2) = 1.;
    P.insert(3,3) = 0.5;
    P.insert(0,2) = -0.5;
    P.insert(2,0) = -0.5;
    P.insert(2,3) = 0.5;
    P.insert(3,2) = 0.5;
    P *= 2.; // correcting for 1/2 factor

    Eigen::VectorXd q (4);
    q(0) = -1.;
    q(1) = -3.;
    q(2) = 1.;
    q(3) = -1.;

    // inequality constraints
    Eigen::SparseMatrix<double> G (7, 4);
    Eigen::VectorXd w (7);

    // all variables >= 0
    G.insert(0,0) = -1.;
    w(0) = 0.;
    G.insert(1,1) = -1.;
    w(1) = 0.;
    G.insert(2,2) = -1.;
    w(2) = 0.;
    G.insert(3,3) = -1.;
    w(3) = 0.;

    // x0 + 2x1 + x2 + x3 <= 5
    G.insert(4,0) = 1.;
    G.insert(4,1) = 2.;
    G.insert(4,2) = 1.;
    G.insert(4,3) = 1.;
    w(4) = 5.;

    // 3x0 + x1 + 2x2 - x3 <= 4
    G.insert(5,0) = 3.;
    G.insert(5,1) = 1.;
    G.insert(5,2) = 2.;
    G.insert(5,3) = -1.;
    w(5) = 4.;

    // x1 + 4x2 >= 1.5
    G.insert(6,1) = -1.;
    G.insert(6,2) = -4.;
    w(6) = -1.5;

    // solve QP
    Solver solver(P, q, G, w);
    const Result result = solver.solve();
    ASSERT_TRUE(result.feasible);
    ASSERT_TRUE(result.converged);

    // check objective against known value
    EXPECT_NEAR(result.objective, -4.681818181, 1e-2);
}

// TODO: Equality constraints not linearly independent
