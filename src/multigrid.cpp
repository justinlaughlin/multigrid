#include <iostream>
#include <linearalgebra.h>
#include <multigrid.h>
#include <libraries/Eigen/Sparse>
#include <initializeProblem.h>
#include <mpi.h>

using namespace std;

int DIRECT_SOLVE_THRESHOLD = 250;
double DOF_PER_CPU_MULT = 1/2;
double DOF_PER_CPU = pow(10,5);

double totalGSTime = 0;         // total time spent on GS (including communication)
double computationGSTime = 0;   // time spent on computation portion of GS (excluding communication)
double getTime() {
    return totalGSTime;
}
double getComputationTime(){
    return computationGSTime;
}

int updateWorkers(int newDim, int origProcs) {
    return min(origProcs, int(ceil(newDim/DOF_PER_CPU)));
    //return origProcs;
}

/*
Send data from the master to the workers
*/
void master_send(int fineNdof, int numWorkers, bool killSignal,
                int N, VectorXd b, VectorXd u, int GS_ITERATIONS,
                int RB){
    int chunk = std::floor(fineNdof/(numWorkers)); 
    int remainderChunk = chunk + int(fineNdof%numWorkers);
    int idx0 = 0;
 
    // loop through all workers
    for (int n=1; n<=numWorkers; n++){
        // MPI_SEND(buffer, count, datatype, destination (proc #), tag, communicator)
        MPI_Send(&killSignal,1,MPI::BOOL,n,0,MPI_COMM_WORLD);
        
        MPI_Send(&N,1,MPI::INT,n,1,MPI_COMM_WORLD);
        MPI_Send(&idx0,1,MPI::INT,n,2,MPI_COMM_WORLD);
        if (n<numWorkers)
            MPI_Send(&chunk,1,MPI::INT,n,3,MPI_COMM_WORLD);
        else 
            MPI_Send(&remainderChunk,1,MPI::INT,n,3,MPI_COMM_WORLD);
        MPI_Send(b.data(),b.size(),MPI::DOUBLE,n,4,MPI_COMM_WORLD); // rhs vector
        MPI_Send(u.data(),u.size(),MPI::DOUBLE,n,5,MPI_COMM_WORLD); 
        MPI_Send(&GS_ITERATIONS,1,MPI::INT,n,6,MPI_COMM_WORLD);
        MPI_Send(&RB,1,MPI::INT,n,13,MPI_COMM_WORLD);
        idx0 += chunk;
    }

}


/*
Receive the smoothed solution from the workers
*/
void master_receive(int fineNdof, int numWorkers, VectorXd& u, double& computationGSTime){

    MPI_Status Stat;
    int chunk = std::floor(fineNdof/(numWorkers)); 
    int remainderChunk = chunk + int(fineNdof%numWorkers);
    int idx0 = 0;
    VectorXd uchunk = VectorXd::Zero(chunk,1);
    VectorXd uchunkR = VectorXd::Zero(remainderChunk,1);
    // now receive the smoothed solution and assemble back into u
    for (int n=1; n<=numWorkers; n++){

        if (n<numWorkers){
            MPI_Recv(uchunk.data(),chunk,MPI::DOUBLE,n,10,MPI_COMM_WORLD,&Stat);
            for (int z=0; z<chunk; z++){
                u(idx0+z) = uchunk(z);
            }
            idx0 += chunk;
        }
        else {
            MPI_Recv(uchunkR.data(),remainderChunk,MPI::DOUBLE,n,10,MPI_COMM_WORLD,&Stat);
            for (int z=0; z<remainderChunk; z++) {
                u(idx0+z) = uchunkR(z);
            }
        }

        double GSTime;
        MPI_Recv(&GSTime, 1, MPI::DOUBLE, n, 12, MPI_COMM_WORLD,&Stat);
        computationGSTime += GSTime/numWorkers;
    }
}

/*
 * Performs a single V cycle on the system Au=b 
 */
VectorXd vcycle(SparseMatrix A, VectorXd b, VectorXd u, int numProc, bool RBordering, matMap& Amap) {
    /*
    RBordering: whether to use RB ordering or just normal gauss-seidel
    */
    cout.precision(12);

    int fineNdof = b.size(); // total degrees of freedom on the (current) fine grid size
    int N = int(sqrt(fineNdof)) + 2; // number of grid points in each dimension
    int Nm2 = N-2;

    // MPI
    int numWorkers = numProc-1; // Master does not participate
    cout << "vcycle() numWorkers: " << numWorkers << endl;
    bool killSignal = false;
    MPI_Status Stat;

    //Gauss-Seidel iterations
    int GS_ITERATIONS = 20;

    cout << "\nvcycle() iteration..." << "fineNdof = " << fineNdof << endl;

    // Direct solve if below threshold
    if (fineNdof <= DIRECT_SOLVE_THRESHOLD) {
        Eigen::SimplicialLDLT<SparseMatrix> solver;
        solver.compute(A);
        cout << "Size is smaller than or equal to DIRECT_SOLVE_THRESHOLD = " << DIRECT_SOLVE_THRESHOLD << "...Directly solving system for N = " << fineNdof << "\n" << endl;
        cout << "*******************" << endl;
        cout << "Beginning prolongation..." << endl;
        return solver.solve(b);
    }

    /* 
    Restriction Phase: Gauss-Seidel Smoothing
    */

    // Timing for Gauss-Seidel (including communication costs)
    auto startGS = chrono::high_resolution_clock::now();
   
    // Run in serial
    if (numWorkers <= 1){         
        int idx0=0; int chunk=0; // 0 or u.size() signals that we are running in serial 
        int RB=0;
        if (RBordering==false)
            RB=0;
        else
            RB=1; // red points

        matKey Akey = make_tuple(N,idx0,chunk,RB);
        // construct or find A
        if (Amap.find(Akey) == Amap.end()) {
            //not found
            auto Atemp = constructLHS(N,idx0,chunk,RB);
            Amap.insert(pair<matKey,SparseMatrix>(Akey,Atemp));
        }
        else {
            cout << "LHS matrix found for tuple key: ( " << N << " , ";
            cout << idx0 << " , " << chunk << " , " << RB << " ) - skipping construction..." << endl;
        }
        auto Anow = Amap.at(Akey);

        for (int i = 0; i < GS_ITERATIONS; i++) {
            GaussSeidelIterationParallel(Anow,u,b,idx0,chunk,RB);
        }

        //black points
        if (RBordering==true){
            RB=2; 

            Akey = make_tuple(N,idx0,chunk,RB);
            // construct or find A
            if (Amap.find(Akey) == Amap.end()) {
                //not found
                auto Atemp = constructLHS(N,idx0,chunk,RB);
                Amap.insert(pair<matKey,SparseMatrix>(Akey,Atemp));
            }
            else {
                cout << "LHS matrix found for tuple key: ( " << N << " , ";
                cout << idx0 << " , " << chunk << " , " << RB << " ) - skipping construction..." << endl;
            }
            auto Anow = Amap.at(Akey);

            for (int i = 0; i < GS_ITERATIONS; i++) {
                GaussSeidelIterationParallel(Anow,u,b,idx0,chunk,RB);
            }
        }

    cout << "Finished " << GS_ITERATIONS << " Gauss Seidel smoothing iterations (serial) for Ndof = " << fineNdof << endl;
    }
    // Run in parallel 
    else if (numWorkers>1 and RBordering==false){ 
        int RB = 0; // No red-black ordering
        master_send(fineNdof, numWorkers, killSignal, N, b, u, GS_ITERATIONS, RB);
        master_receive(fineNdof, numWorkers, u, computationGSTime);
    }
    // Run in parallel (gauss-seidel red-black ordering)
    else if (numWorkers>1 and RBordering==true){
        int RB = 1; // begin with red points (odd parity)
        master_send(fineNdof, numWorkers, killSignal, N, b, u, GS_ITERATIONS, RB);
        master_receive(fineNdof, numWorkers, u, computationGSTime);

        RB = 2; // now we solve black points (even parity)
        master_send(fineNdof, numWorkers, killSignal, N, b, u, GS_ITERATIONS, RB);
        master_receive(fineNdof, numWorkers, u, computationGSTime);
    }
    
    chrono::duration<double> GSduration = chrono::high_resolution_clock::now() - startGS;
    totalGSTime += double(GSduration.count());


    // Interpolation operator
    int coarseN = (N-1)/2 + 1;
    int coarseNm2 = coarseN-2;
    //int coarseNdof = (fineNdof-1)/2 + 1;
    int coarseNdof = pow(coarseNm2,2);

    cout << "coarseNdof = " << coarseNdof << endl;

    // 2d prolongation stencil (rows of P should sum to 1)
    // 1/4  1/2  1/4
    // 1/2   1   1/2
    // 1/4  1/2  1/4
    //
    // 2d restriction stencil [full weighting] (rows of R should sum to 1)
    // 1/16 1/8  1/16
    // 1/8  1/4  1/8
    // 1/16 1/8  1/16

    vector<T> tripletListP;
    tripletListP.reserve(9*coarseNdof);
    vector<T> tripletListR;
    tripletListR.reserve(9*coarseNdof);
    SparseMatrix P = SparseMatrix(fineNdof, coarseNdof);
    SparseMatrix R = SparseMatrix(coarseNdof, fineNdof);
    int idx_fine;
    for (int j=0; j < coarseNdof; j++){
        // Prolongation matrix \Omega^{2h} -> \Omega
        // what a pain to figure this one out...
        // index in \Omega^h corresponding to index j in \Omega^{2h}
        idx_fine = ((2*floor(j/coarseNm2)+1)*Nm2 + 1) + (j%coarseNm2)*2; 
        
        tripletListP.push_back(T(idx_fine,j,1.0));
        // up, down, right, left
        tripletListP.push_back(T(idx_fine+Nm2,j,0.5));
        tripletListP.push_back(T(idx_fine-Nm2,j,0.5));
        tripletListP.push_back(T(idx_fine+1,j,0.5));  
        tripletListP.push_back(T(idx_fine-1,j,0.5));  
        // diagonals
        tripletListP.push_back(T(idx_fine+Nm2+1,j,0.25));
        tripletListP.push_back(T(idx_fine+Nm2-1,j,0.25));
        tripletListP.push_back(T(idx_fine-Nm2+1,j,0.25));
        tripletListP.push_back(T(idx_fine-Nm2-1,j,0.25));

        // Restriction matrix
        tripletListR.push_back(T(j,idx_fine,0.25));
        // up, down, right, left
        tripletListR.push_back(T(j,idx_fine+Nm2,0.125));
        tripletListR.push_back(T(j,idx_fine-Nm2,0.125));
        tripletListR.push_back(T(j,idx_fine+1,0.125));  
        tripletListR.push_back(T(j,idx_fine-1,0.125));  
        // diagonals
        tripletListR.push_back(T(j,idx_fine+Nm2+1,0.0625));
        tripletListR.push_back(T(j,idx_fine+Nm2-1,0.0625));
        tripletListR.push_back(T(j,idx_fine-Nm2+1,0.0625));
        tripletListR.push_back(T(j,idx_fine-Nm2-1,0.0625));

    }
    P.setFromTriplets(tripletListP.begin(), tripletListP.end());
    R.setFromTriplets(tripletListR.begin(), tripletListR.end());


    // Residual
    auto residual = b - A*u;
    //cout << "residual norm: " << residual.norm() << endl;
    auto coarseProjectionResidual = R*residual;

    // Restriction
    auto coarseMatrix = constructLHS(coarseN,0,0,0);

    VectorXd zeroVec = VectorXd::Zero(coarseNdof,1);

    int nextNumProc = updateWorkers(coarseNdof, numProc);
    auto coarseU = vcycle(coarseMatrix, coarseProjectionResidual, zeroVec, nextNumProc, RBordering, Amap);

    u = u+P*coarseU;
    cout << "Prolongating for Ndof = " << fineNdof << endl;

    /* 
    Interpolation Phase: Gauss-Seidel Smoothing
    */

    // Timing for Gauss-Seidel (including communication costs)
    startGS = chrono::high_resolution_clock::now();
   
    // Run in serial
    if (numWorkers <= 1){         
        int idx0=0; int chunk=0; // 0 or u.size() signals that we are running in serial 
        int RB=0;
        if (RBordering==false)
            RB=0;
        else
            RB=1; // red points

        matKey Akey = make_tuple(N,idx0,chunk,RB);
        // construct or find A
        if (Amap.find(Akey) == Amap.end()) {
            //not found
            auto Atemp = constructLHS(N,idx0,chunk,RB);
            Amap.insert(pair<matKey,SparseMatrix>(Akey,Atemp));
        }
        else {
            cout << "LHS matrix found for tuple key: ( " << N << " , ";
            cout << idx0 << " , " << chunk << " , " << RB << " ) - skipping construction..." << endl;
        }
        auto Anow = Amap.at(Akey);

        for (int i = 0; i < GS_ITERATIONS; i++) {
            GaussSeidelIterationParallel(Anow,u,b,idx0,chunk,RB);
        }

        //black points
        if (RBordering==true){
            RB=2; 

            Akey = make_tuple(N,idx0,chunk,RB);
            // construct or find A
            if (Amap.find(Akey) == Amap.end()) {
                //not found
                auto Atemp = constructLHS(N,idx0,chunk,RB);
                Amap.insert(pair<matKey,SparseMatrix>(Akey,Atemp));
            }
            else {
                cout << "LHS matrix found for tuple key: ( " << N << " , ";
                cout << idx0 << " , " << chunk << " , " << RB << " ) - skipping construction..." << endl;
            }
            auto Anow = Amap.at(Akey);

            for (int i = 0; i < GS_ITERATIONS; i++) {
                GaussSeidelIterationParallel(Anow,u,b,idx0,chunk,RB);
            }
        }

    cout << "Finished " << GS_ITERATIONS << " Gauss Seidel smoothing iterations (serial) for Ndof = " << fineNdof << endl;
    }
    // Run in parallel 
    else if (numWorkers>1 and RBordering==false){ 
        int RB = 0; // No red-black ordering
        master_send(fineNdof, numWorkers, killSignal, N, b, u, GS_ITERATIONS, RB);
        master_receive(fineNdof, numWorkers, u, computationGSTime);
    }

    // Run in parallel (gauss-seidel red-black ordering)
    else if (numWorkers>1 and RBordering==true){
        int RB = 1; // begin with red points (odd parity)
        master_send(fineNdof, numWorkers, killSignal, N, b, u, GS_ITERATIONS, RB);
        master_receive(fineNdof, numWorkers, u, computationGSTime);

        RB = 2; // now we solve black points (even parity)
        master_send(fineNdof, numWorkers, killSignal, N, b, u, GS_ITERATIONS, RB);
        master_receive(fineNdof, numWorkers, u, computationGSTime);
    }
    
    GSduration = chrono::high_resolution_clock::now() - startGS;
    totalGSTime += double(GSduration.count());


    return u;
}

