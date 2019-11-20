#ifndef PARALLELMULTIGRID_MULTIGRID_H
#define PARALLELMULTIGRID_MULTIGRID_H
#include <libraries/Eigen/Sparse>
#include <chrono>

typedef Eigen::SparseMatrix<double> SparseMatrix;
typedef Eigen::SparseVector<double> SparseVector;
typedef Eigen::Triplet<double> T;
using Eigen::VectorXd;
using Eigen::MatrixXd;


// data structure to store different LHS matrices
// key is the tuple of arguments to constructLHS()
typedef std::tuple<int, int, int, int> matKey;
typedef std::map<matKey, SparseMatrix> matMap;

// Getter functions for timings
double getTime();
double getComputationTime();

// Send data from the master to the workers
void master_send(int fineNdof, int numWorkers, bool killSignal,
                int N, VectorXd b, VectorXd u, int GS_ITERATIONS,
                int RB);

// Receive smoothed solution from the workers
void master_receive(int fineNdof, int numWorkers, VectorXd& u, double& computationGSTime);

// Single multigrid v-cycle iteration
VectorXd vcycle(SparseMatrix A, VectorXd b, VectorXd u, int numProc, bool RBordering, matMap& Amap);

#endif //PARALLELMULTIGRID_MULTIGRID_H