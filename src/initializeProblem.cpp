#include <unordered_map>
#include <iostream>
#include <fstream>
#include <libraries/Eigen/Sparse>
#include <libraries/Eigen/SparseCholesky>
#include <initializeProblem.h>

using namespace std;

SparseMatrix constructLHS(int N, int idx0, int chunk, int RB){
    /*
    idx0 is the offset in cols/rows
    RB: red [odd] (1), black [even] (2), no RB ordering scheme (0)
    For red, odd points (e.g. 1,2) we need the values of A with even parity (e.g. [1,1], [2,2], etc.)
    */ 
    //cout << "\nconstructLHS():" << endl;
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    // system has pow(N-2,2) degrees of freedom due to homogeneous dirichlet bc
    int Nm2=N-2;
    int Ndof=pow(Nm2,2);
    if (chunk==0)
        chunk = Ndof; //defualt. not truncated
    int parity;

    int i,j; // j is the index in the full A matrix, i is the corresponding index for truncated A
    
    // step size in x and y since domain is [0,1] x [0,1]
    double h = double(1)/(double(N)-double(1));
    double powhm2 = 1.0/pow(h,2);
    //cout << "step size (h) = " << h << endl;

    /* Stencil for 2d finite difference of -laplacian operator:
                     0  -1  0
                     -1  4  -1
                     0  -1  0
    */

    // construct triplets to increase performance of filling sparse matrix A
    vector<T> tripletList;
    tripletList.reserve(5*Ndof);
    // construct the discretized laplacian operator matrix row by row
    for(j=idx0;j<idx0+chunk;j++){ 
        parity = (j%Nm2 + int(floor(double(j)/double(Nm2))) )%2;
        if ((RB==1 and parity == 0) or (RB==2 and parity == 1) or (RB==0)){ 
            i = j-idx0; // truncated index
            tripletList.push_back(T(i,j,4.0*powhm2));
            // since we are indexing row by row
            if (j%Nm2 != 0){ // if we are not on the left column (x=dx)
                tripletList.push_back(T(i,j-1,-powhm2));
            }
            if ((j+1)%Nm2 != 0){ // if we are not on the right column (x=1-dx)
                tripletList.push_back(T(i,j+1,-powhm2));
            }
            if (j > Nm2-1){ // if we are not on the bottom row (y=dy)
                tripletList.push_back(T(i,j-Nm2,-powhm2));
            }
            if (j < Ndof - Nm2){ // if we are not on the top row (y=1-dy)
                tripletList.push_back(T(i,j+Nm2,-powhm2));
            }
        }
    }

    SparseMatrix tempA(chunk,Ndof); 
    tempA.setFromTriplets(tripletList.begin(), tripletList.end());

    // Unfortunately RowMajor doesn't work so well so we have to use this trick, but 
    // thankfully A is always symmetric!
    SparseMatrix A(Ndof,chunk); 
    A = tempA.transpose();

    return A;
}

tuple<MatrixXd,MatrixXd> constructXY(int N){
    //cout << "\nconstructXYGrid():" << endl;
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    // system has pow(N-2,2) degrees of freedom due to homogeneous dirichlet bc
    int Nm2=N-2;
    int Ndof=pow(Nm2,2);

    int i;

    auto X = RowVectorXd::LinSpaced(N,0,1).replicate(N,1).transpose();
    auto Y = VectorXd::LinSpaced(N,0,1).replicate(1,N).transpose();
    
    auto XY = make_tuple(X,Y);
    return XY;
}

VectorXd constructRHS(int N, unordered_map<int,double> a, MatrixXd X, MatrixXd Y){
    /* 
    b(x,y) = sum_n [a_n * sin(n pi x)*sin(n pi y)
    */
    //cout << "constructRHS():" << endl;
    int Ndof = pow(N-2,2);
    VectorXd b;
    b = MatrixXd::Zero(Ndof,1);

    //cout << "Coefficients for b(x,y):" << endl;

    int i,j;
    int idx = 0;
    for (j=1;j<N-1;j++){
        for (i=1;i<N-1;i++){
            for (pair<int,double> coeff : a){
                //if (i==1 and j==1) cout << "a_" << coeff.first << " = " << coeff.second << endl;
                b[idx] +=  coeff.second * sin(coeff.first*PI*X(i,j)) * sin(coeff.first*PI*Y(i,j));
            }
            idx += 1;
        }
    }

    return b;
}

double eval_ua(Map a, double x, double y){
    double ua = 0;
    for (pair<int,double> coeff : a){
        ua += coeff.second/(2*pow(coeff.first*PI,2))*sin(coeff.first*PI*x)*sin(coeff.first*PI*y);
    }
    return ua;
}

VectorXd computeAnalyticSolution(int N, Map a, MatrixXd X, MatrixXd Y){
    VectorXd ua;
    ua = MatrixXd::Zero(pow(N-2,2),1);

    int i,j;
    int idx = 0;
    for (j=1;j<N-1;j++){
        for (i=1;i<N-1;i++){
            ua[idx] += eval_ua(a,X(i,j),Y(i,j));
            idx += 1;
        }
    }

    cout << "Done computing analytical solution for Ndof = " << pow(N-2,2) << "\n" << endl;
    return ua;
}


double gaussQuad(Map a, VectorXd xel, VectorXd yel, VectorXd uel){
    /* 
    approximate integration via gaussian quadrature
    xel, yel: 2x1 vectors of x,y coordinates
    uel: 4x1 vector of values of u at four corners of element. index from bottom left vertex then snakes up (same as coordinate system) i.e.
    [2 3]
    [0 1]
    */
    // order 4 integration
    vector<double> weights{0.6521451548625461, 0.6521451548625461, 0.3478548451374538, 0.3478548451374538}; 
    // this implies transformation to [-1,1]x[-1,1] isoparametric domain
    vector<double> abscissa{-0.3399810435848563, 0.3399810435848563, -0.8611363115940526, 0.8611363115940526}; 
    // implies [0,1] domain. i.e. (abscissa+1)/2
    vector<double> abscissa_p{0.33000947820757187, 0.6699905217924281, 0.06943184420297371, 0.9305681557970262}; // 
    int i,j;
    int Ng = weights.size(); // number of gauss quadrature points
    double dx = xel[1] - xel[0]; 
    double dy = yel[1] - yel[0];
    double x,y;
    double ubot, utop, ueval, ua;
    double area = dx*dy;
    double uint = 0; // the integral

    // loop over all gauss quad points (2d)
    for (i=0;i<Ng;i++){
        for (j=0;j<Ng;j++){
            x = xel[0]+dx/2 + abscissa[i]*dx/2;
            y = yel[0]+dy/2 + abscissa[j]*dy/2;
            // bilinear interpolation to evaluate u
            ubot = uel[0] + (uel[1]-uel[0])*abscissa_p[i];
            utop = uel[2] + (uel[3]-uel[2])*abscissa_p[i];
            ueval = ubot + (utop-ubot)*abscissa_p[j];
            ua = eval_ua(a,x,y); // compute the analytic solution

            uint += area/4 * weights[i]*weights[j] * (ueval-ua);

        }
    } 

    return uint;
     
}


double computeError(int N, MatrixXd X, MatrixXd Y, VectorXd umg, unordered_map<int,double> a){
    // computes error for a single element
    // e = \int_{x_0}^{x_1} dx \int_{y_0}^{y_1} dy (u_a(x) - u)^2

    cout << "Computing relative L2 error..." << endl;

    double error = 0; // L2 norm of absolute error
    double denom = 0; // L2 norm of analytic solution

    int Nm1 = N-1;
    int Nel = pow(N-1,2); // Number of elements

    // indices for different corners
    int idx0 = 0;
    int idx1 = 0;
    int idx2 = 0;
    int idx3 = 0;

    // x,y coordinate indices
    int xidx, yidx;

    int i;
    VectorXd umg_block, zeroVec;
    zeroVec = VectorXd::Zero(4);

    // loop through elements
    for (i=0;i<Nel;i++){
        umg_block = VectorXd::Zero(4); // reset
        // bottom-left corner
        if (i>=Nm1 and i%Nm1 != 0){
            umg_block[0] = umg[idx0];
            ++idx0;
        }
        // bottom-right corner
        if (i>=Nm1 and (i+1)%Nm1 != 0){
            umg_block[1] = umg[idx1];
            ++idx1;
        }
        // top-left corner
        if (i<=pow(Nm1,2)-N and i%Nm1 != 0){
            umg_block[2] = umg[idx2];
            ++idx2;
        }
        // top-right corner
        if (i<=pow(Nm1,2)-N and (i+1)%Nm1 != 0){
            umg_block[3] = umg[idx3];
            ++idx3;
        }

        xidx = i%Nm1;
        yidx = floor(i/Nm1);
        //cout << "xidx: " << xidx << "yidx: " << yidx << endl;
        error += pow(gaussQuad(a, X.block<2,1>(xidx,yidx), Y.block<1,2>(xidx,yidx), umg_block), 2);
        denom += pow(gaussQuad(a, X.block<2,1>(xidx,yidx), Y.block<1,2>(xidx,yidx), zeroVec), 2);
    }

    return sqrt(error)/sqrt(denom); // return the relative error
}

