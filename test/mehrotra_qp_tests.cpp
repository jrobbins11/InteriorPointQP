#include "MehrotraQP.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/src/SparseCore/SparseMatrix.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <random>
#include <iostream>

using namespace MehrotraQP;

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
TEST(MehrotraQP, InvalidProblemDimensions)
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
        triplets.emplace_back(1, 1, 1.0);
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
        const double result_norm = result.x.norm();
        EXPECT_LT(result.objective, perturbed_objective(result.x, 0.01*result_norm));
    }
}

// Unconstrained produces optimal solution
TEST(MehrotraQP, UnconstrainedOptimal)
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
        EXPECT_NEAR(result.x(i), q(i), 1e-6);
    }
}

// Infeasible problem produces infeasible solution
TEST_F(RandomQP, InfeasibleProblem)
{
    // TODO
}