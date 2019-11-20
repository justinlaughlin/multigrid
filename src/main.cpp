//TODO
// round down N to nearest "nice number"

#include <cmath>
#include <cstdlib>
#include <stdlib.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <libraries/Eigen/Core>
#include <libraries/Eigen/Sparse>
#include <libraries/Eigen/SparseCholesky>
#include <unordered_map>
#include <initializeProblem.h>
//#include <error.h>
#include <linearalgebra.h>
#include <multigrid.h>
#include <chrono>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[]){
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int numWorkers = world_size-1;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Values of N which can be easily interpolated (2^n+1):
    // 3,5,9,17,33,65,129,257,513,1025,2049 (~5gb),4098,8196
    // And corresponding degres of freedom Ndof = (N-2)^2
    // 1, 9, 49, 225, 961, 3969, 16129, 65025, 261121, 1046529, 4190209 (~5gb),16793604,67174416(-80gb)
    int N = atoi(argv[1]);
    // System will be of size (N-2)^2 (-2 due to homogeneous dirichlet bcs)

    bool writeData = true;
    bool RBordering = true; // should we use red-black ordering for Gauss-Seidel smooths?

    /*
    Construct the LHS matrix A and RHS vector b
    */
    // initialize a map for LHS matrices
    matKey Akey;
    matMap Amap;
    Akey = make_tuple(N,0,0,0);
    auto A = constructLHS(N,0,0,0);
    Amap.insert(pair<matKey,SparseMatrix>(Akey,A));

    auto XY = constructXY(N);
    auto X = get<0>(XY);
    auto Y = get<1>(XY);

    // Coefficients for b(x,y) (i.e. a_n). Keys correspond to indices (n), values are the coefficients)
    //Map a = {{1,2},{5,1},{10,10},{20,5}};
    Map a = {{1,-1}, {2,4}, {3,9}, {4,16}, {5,25}, {10,100}, {20,-200}, {50,-2000}}; 
    auto b = constructRHS(N,a,X,Y);


    /*
    Tasks for the master
    */
    if (world_rank==0){
        cout << "Total CPUs: " << world_size << endl;
        unordered_map<string,chrono::duration<double>> timings;

        /*
        Compute analytic solution.
        */
        auto start = chrono::high_resolution_clock::now();
        cout << "*******************" << endl;
        cout << "Computing analytic solution..." << endl;
        auto ua = computeAnalyticSolution(N,a,X,Y);
        double ua_norm = ua.norm();
        timings[string("analytic")] = chrono::high_resolution_clock::now() - start;

        /* 
        Multigrid vcycle
        */
        cout << "*******************" << endl;
        cout << "Computing solution via vcycle..." << endl;
        start = chrono::high_resolution_clock::now();

        VectorXd umg = VectorXd::Zero(b.size(),1); // initial guess

        // Main loop
        if (numWorkers>1){
            bool killSignal = false; // when true, tells the workers cpus to stop
            for (int n=1;n<=numWorkers;n++){
                MPI_Send(&killSignal,1,MPI::BOOL,n,0,MPI_COMM_WORLD);
            }
        }
        int Ncycle = 3; // number of vcycles
        for (int k=0; k<Ncycle; k++){
            umg = vcycle(A,b,umg,world_size,RBordering,Amap);
            auto absErrNodalMultigrid = umg-ua;
            cout << "^^^relative error multigrid (cycle " << k+1 <<"): ";
            cout << absErrNodalMultigrid.norm()/ua.norm() << endl;
        }
        if (numWorkers>1){
            bool killSignal = true; // kill the workers
            for (int n=1;n<=numWorkers;n++){
                MPI_Send(&killSignal,1,MPI::BOOL,n,0,MPI_COMM_WORLD);
            }
        }

        
        timings[string("vcycle")] = chrono::high_resolution_clock::now() - start;
        cout << "^^^Time to complete " << Ncycle << " multigrid vcycle(s): " << timings[string("vcycle")].count() << endl;
        cout << "^^^Time to compute analytic solution: " << timings[string("analytic")].count() << endl;

        auto res_mg = b - A*umg;
        cout << "^^^Norm of parallel multigrid residual: " << res_mg.norm() << "\n" << endl;

        // /*
        // Direct solve using Cholesky decomposition
        // */
        // cout << "*******************" << endl;
        // start = chrono::high_resolution_clock::now();
        // cout << "Computing solution directly via Cholesky decomp..." << endl;
        // Eigen::SimplicialLDLT<SparseMatrix> solver;
        // solver.compute(A);
        // auto ucholesky = solver.solve(b);
        // timings[string("cholesky")] = chrono::high_resolution_clock::now() - start;
        // cout << "Time to complete Cholesky Decomp: " << timings[string("cholesky")].count() << endl;

        // auto res_cholesky = b - A*ucholesky;
        // cout << "Norm of Cholesky residual: " << res_cholesky.norm() << "\n" << endl;

        /* 
        Computing errors
        */
        cout << "*******************" << endl;

       // double errorMultigrid = computeError(N,X,Y,umg,a);
       // double errorCholesky = computeError(N,X,Y,ucholesky,a);

        // Gauss quadrature will get us more accurate error estimates but for large number of
        // points is somewhat unnecessary. 
        //        cout << "\n4 point Gauss quad relative error = " << endl;
        //        cout << "relative error multigrid: " << errorMultigrid << endl;
        //        cout << "relative error cholesky: " << errorCholesky << endl;

        VectorXd absErrNodalMultigrid = umg - ua;
        // VectorXd absErrNodalCholesky = ucholesky - ua;
        // cout << "\neasy way to calculate relative error (compare vertices of solution directly): " << endl;
        // cout << "relative error multigrid: " << absErrNodalMultigrid.norm()/ua.norm() << endl;
        // cout << "relative error cholesky: " << absErrNodalCholesky.norm()/ua.norm() << endl;

        /*
        Timings
        */
        //cout << "*******************" << endl;
        //cout << "Timings:" << endl;
        
        // cout << "Cholesky decomp: " << timings[string("cholesky")].count() << endl;
        // cout << "Multigrid was " << timings[string("cholesky")].count() / timings[string("vcycle")].count() << " times faster..." << "\n" << endl;

        /*
        Write data to files
        */
        if (writeData == true){
            cout << "*******************" << endl;
            cout << "Dumping data..." << endl;
            ofstream file_X("X.txt");
            ofstream file_Y("Y.txt");
            ofstream file_umg("umg.txt");
            // ofstream file_ucholesky("ucholesky.txt");
            ofstream file_ua("ua.txt");
            ofstream file_errorMultigrid("errorMultigrid.txt");
            // ofstream file_errorCholesky("errorCholesky.txt");

            if (file_X.is_open())
                file_X << X << endl;
            if (file_Y.is_open())
                file_Y << Y << endl;
            if (file_umg.is_open())
                file_umg << umg << endl;
            // if (file_ucholesky.is_open())
            //     file_ucholesky << ucholesky << endl;
            if (file_ua.is_open())
                file_ua << ua << endl;
            if (file_errorMultigrid.is_open())
                file_errorMultigrid << absErrNodalMultigrid << endl;
            // if (file_errorCholesky.is_open())
            //     file_errorCholesky << absErrNodalCholesky << endl;

            // careful with these, A can get very large! only uncomment for debugging small problems
//            ofstream file_A("A.txt");
//            ofstream file_b("b.txt");
//            if (file_A.is_open())
//                file_A << MatrixXd(A) << endl;
//            if (file_b.is_open())
//                file_b << b << endl;
            cout << "Finished dumping data." << endl; 
        } // end if writeData==true
        // cout << "Cholesky decomp: " << timings[string("cholesky")].count() << endl;
        //cout << "Multigrid was " << timings[string("cholesky")].count() / timings[string("vcycle")].count() << " times faster..." << "\n" << endl;
        cout << "Total time spent on pre-smoothing Gauss-Seidel (including communication): " << getTime() << endl;
        cout << "Total time spent on pre-smoothing Gauss-Seidel (excluding communication): " << getComputationTime() << endl; 
        cout << "^^^Percentage of time spent on Gauss-Seidel computations out of total time: " << getComputationTime()/getTime() << endl;

    } // end if world_rank==0


    /*
    Tasks for the workers
    */
    if (world_rank!=0 and numWorkers>1){

        MPI_Status Stat;
        bool killSignal = false;
        int currentN, idx0, chunk, GS_ITERATIONS, RB;

        // MPI_Recv(buffer, count, datatype, source (proc #), tag (arbitrary int), communicator, status)
        MPI_Recv(&killSignal,1,MPI::BOOL,0,0,MPI_COMM_WORLD,&Stat);
        while (true){
            MPI_Recv(&killSignal,1,MPI::BOOL,0,0,MPI_COMM_WORLD,&Stat);
            if (killSignal==true) break;
 
            MPI_Recv(&currentN,1,MPI::INT,0,1,MPI_COMM_WORLD,&Stat);
            int currentNdof = int(pow(currentN-2,2));
            VectorXd rhs = VectorXd::Zero(currentNdof);
            VectorXd my_u = VectorXd::Zero(currentNdof);

            MPI_Recv(&idx0,1,MPI::INT,0,2,MPI_COMM_WORLD,&Stat);
            MPI_Recv(&chunk,1,MPI::INT,0,3,MPI_COMM_WORLD,&Stat);
            MPI_Recv(rhs.data(),currentNdof,MPI::DOUBLE,0,4,MPI_COMM_WORLD,&Stat); // rhs vector
            MPI_Recv(my_u.data(),currentNdof,MPI::DOUBLE,0,5,MPI_COMM_WORLD,&Stat);
            MPI_Recv(&GS_ITERATIONS,1,MPI::INT,0,6,MPI_COMM_WORLD,&Stat);
            MPI_Recv(&RB,1,MPI::INT,0,13,MPI_COMM_WORLD,&Stat);
           
            // construct A and then truncate 
            matKey Akey = make_tuple(currentN,idx0,chunk,RB);
            if (Amap.find(Akey) == Amap.end()) {
                //not found
                auto Atemp = constructLHS(currentN,idx0,chunk,RB);
                Amap.insert(pair<matKey,SparseMatrix>(Akey,Atemp));
            }
            else {
                cout << "LHS matrix found for tuple key: ( " << currentN << " , ";
                cout << idx0 << " , " << chunk << " , " << RB << " ) - skipping construction..." << endl;
            }
            auto Anow = Amap.at(Akey);

            auto startGS = chrono::high_resolution_clock::now();
            for (int i=0; i<GS_ITERATIONS; i++){
                GaussSeidelIterationParallel(Anow,my_u,rhs,idx0,chunk,RB);
            }

            chrono::duration<double> GSduration = chrono::high_resolution_clock::now() - startGS;

            double GSTime = double(GSduration.count());
            MPI_Send(&GSTime,1,MPI::DOUBLE,0,12,MPI_COMM_WORLD);

            MPI_Send(my_u(Eigen::seqN(idx0,chunk)).data(),chunk,MPI::DOUBLE,0,10,MPI_COMM_WORLD);
        }


    }

    MPI_Finalize();

}
