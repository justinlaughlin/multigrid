#include <linearalgebra.h>
#include <iostream>
#include <string>
#include <mpi.h>
#include <unordered_map>
#include <chrono>

using namespace std;


/* DISCLAIMER:
 Technically A should be RowMajor but Eigen's support of RowMajor seems limited right now (many things break).
 In this case we are okay just switching the order of indices because A is guaranteed to always be symmetric.
 i.e. we can send columns of A to each processor instead of rows and the resulting calculations will be equivalent.
*/

void GaussSeidelIterationParallel(SparseMatrix& A, VectorXd& u, VectorXd& b, int idx0, int chunk, int RB) {
    /*
    This code also runs in serial. Just set idx0=0 and chunk=0 or chunk=u.size()
    idx 0: Index to start from
    chunk: number of rows/columns to operate on 
    */

    // Using 0 just returns us to the serial version
    if (chunk==0){
        chunk=u.size();
    }

    for (int k=0; k<chunk; k++) { // when A is truncated
        double sum = 0;
        double val = 0;
        for (SparseMatrix::InnerIterator it(A,k); it; ++it){
            if (it.row() != (idx0+k))
                sum += it.value()*u[it.row()];
            else
                val = it.value();
        }
        // this conditional statement is only needed when using RB ordering.
        // changing input so only odd/even parity are input would eliminate this. 
        // doesn't seem to affect performance by much though...
        // A future enhancement could be to choose which indices of u to send based on parity. For
        // now this works
        if (val != 0)
            u[idx0+k] = (1.0/val) * (b[idx0+k]-sum);
    }
}
