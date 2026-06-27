#include "MehrotraQP.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/src/SparseCore/SparseMatrix.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <random>

using namespace MehrotraQP;

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

// Check that perturbations around the optimal solution for an unconstrained problem have higher cost than solver solution
TEST(MehrotraQP, UnconstrainedOptimal)
{
    // random unconstrained QP
    std::mt19937 gen (42);
    std::uniform_real_distribution<double> distr(1., 10.);
    const int n = 50;
    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
    for (int i=0; i<n; ++i) 
    {
        triplets.emplace_back(i, i, distr(gen));
        q(i) = distr(gen);
    }
    Eigen::SparseMatrix<double> P (n, n);
    P.setFromTriplets(triplets.begin(), triplets.end());

    // solve QP
    Solver solver(P, q);
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
        return (0.5*x_dx.transpose()*P*x_dx + q.transpose()*x_dx)(0);
    };

    // check that random perturbations have higher objective
    for (int i=0; i<100; ++i)
    {
        const double result_norm = result.x.norm();
        EXPECT_LT(result.objective, perturbed_objective(result.x, 0.01*result_norm));
    }
}