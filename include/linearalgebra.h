#ifndef PARALLELMULTIGRID_LINEARALGEBRA_H
#define PARALLELMULTIGRID_LINEARALGEBRA_H

#include <libraries/Eigen/Dense>
#include <libraries/Eigen/Sparse>

typedef Eigen::SparseMatrix<double> SparseMatrix;
using Eigen::VectorXd;
using Eigen::MatrixXd;

void GaussSeidelIteration(SparseMatrix& A, VectorXd& u, VectorXd& b);
void GaussSeidelIterationParallel(SparseMatrix& A, VectorXd& u, VectorXd& b, int idx0, int chunk, int RB);

#endif //PARALLELMULTIGRID_LINEARALGEBRA_H
