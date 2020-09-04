/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *
 */
//#include <stdio.h>
#include "1_d_conv.cu"
#include "n_tree.cuh"
#include <cuda_profiler_api.h>
//#include "utils.h"
/*
*********************************************************************
function name: main
description: test and compare
parameters:
            none
return: none
*********************************************************************
*/


int main(int argc, char const *argv[]){

    const std::string device_cuda = "cuda:0"; //officially retarded
    const std::string device_cpu = "cpu";
//    torch::manual_seed(0);

//    torch::Tensor X_train = read_csv<float>("X_train_PCA.csv",11619,3); something wrong with data probably...
//    torch::Tensor b_train = read_csv<float>("Y_train.csv",11619,1); something wrong with data probably...
    torch::Tensor X_train = torch::rand({10000,nd});//.to(device_cuda); //Something fishy going on here, probably the boxes stuff...
    torch::Tensor b_train = torch::randn({10000,1});//to(device_cuda);
    X_train = X_train.to(device_cuda);
    b_train = b_train.to(device_cuda);
    float ls = 1.0; //lengthscale
    float lambda = 1e-1; // ridge parameter
//    int T = 10;
//    int max_its = 50;
//    float tol = 1e-6;

    rbf_pointer<float> op,op_grad; //potential/kernel choice
    cudaMemcpyFromSymbol(&op, rbf_pointer_func<float>, sizeof(rbf_pointer<float>)); //regular rbf kernel
    cudaMemcpyFromSymbol(&op_grad, rbf_pointer_grad<float>, sizeof(rbf_pointer<float>)); //gradient of rbf kernel
    torch::Tensor res,res_ref;

    FFM_object<float> ffm_obj = FFM_object<float>(X_train, X_train, ls, op, lambda, device_cuda); //FMM object
//    FFM_object<float> ffm_obj_grad = FFM_object<float>(X,X,ls,op_grad,lambda,device_cuda);
    exact_MV<float> exact_ref = exact_MV<float>(X_train, X_train, ls, op, lambda, device_cuda); //Exact method reference
//    exact_MV<float> ffm_obj_grad_exact = exact_MV<float>(X,X,ls,op_grad,lambda,device_cuda);
    /*
     * Implement "skip heuristic" for full interactions..., figure out how to calculate everything on cuda at the end, little interactions as possible. for each level of granularity...
     * */

    /*
     * Evaluate performance and accuracy
     */
    auto start = std::chrono::high_resolution_clock::now();
    res_ref = exact_ref * b_train;
    auto end = std::chrono::high_resolution_clock::now();
    res = ffm_obj * b_train; //Horrendus complexity, needs to fixed now
    auto end_2 = std::chrono::high_resolution_clock::now();
    auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
//    auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2-end);
//    std::cout<<duration_1.count()<<std::endl;
//    std::cout<<duration_2.count()<<std::endl;
//    std::cout<<((res_ref-res)/res_ref).abs_().mean()<<std::endl;
    cudaProfilerStop();
    cudaDeviceReset();
    return 0;
}
