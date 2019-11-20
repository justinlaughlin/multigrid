#ifndef INITIALIZE_PROBLEM_H
#define INITIALIZE_PROBLEM_H

#include <iostream>
#include <unordered_map>
#include <libraries/Eigen/Sparse>
typedef std::unordered_map<int,double> Map;
typedef Eigen::SparseMatrix<double> SparseMatrix;
typedef Eigen::Triplet<double> T;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;

#define PI 3.14159265

SparseMatrix constructLHS(int N, int idx0, int chunk, int RB);

std::tuple<MatrixXd,MatrixXd> constructXY(int N);

VectorXd constructRHS(int N, Map a, MatrixXd X, MatrixXd Y);

double eval_ua(Map a, double x, double y);

VectorXd computeAnalyticSolution(int N, Map a, MatrixXd X, MatrixXd Y);

double gaussQuad(Map a, VectorXd xel, VectorXd yel, VectorXd uel);

double computeError(int N, MatrixXd X, MatrixXd Y, VectorXd umg, Map a);

#endif // INITIALIZE_PROBLEM_H

