/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *
 */
#include <stdio.h>
#include "1_d_conv.cu"
#include "n_tree.cuh"
#include "utils.h"
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

//    torch::Tensor X = read_csv<float>("X.csv",1000,3);
//    torch::Tensor b = read_csv<float>("V.csv",1000,2);
    torch::Tensor X = torch::rand({100,nd});
    torch::Tensor b = torch::randn({100,1});

    float ls = 3.0;
    float lambda = 1e-2;
    rbf_pointer<float> op,op_grad;
    cudaMemcpyFromSymbol(&op, rbf_pointer_func<float>, sizeof(rbf_pointer<float>)); //rbf_pointer_func,rbf_pointer_grad
    cudaMemcpyFromSymbol(&op_grad, rbf_pointer_grad<float>, sizeof(rbf_pointer<float>)); //rbf_pointer_func,rbf_pointer_grad

    FMM_obj<float> ffm_obj = FMM_obj<float>(X,X,ls,op,lambda,device_cuda);
    FMM_obj<float> ffm_obj_grad = FMM_obj<float>(X,X,ls,op_grad,lambda,device_cuda);

//    torch::Tensor output = ffm_obj_test*b;
//    exact_MV<float> exact_obj_test = exact_MV<float>(X,X,ls,op,lambda,device_cuda);
//    torch::Tensor output_ref = exact_obj_test*b;
    torch::Tensor b_inv,tridiag_matrix,log_det,trace;
//    std::tie(b_inv,tridiag_matrix) = CG(ffm_obj_test,b,(float) 1e-6,(int) 100,true);
    std::tie(log_det,trace) = trace_and_log_det_calc(ffm_obj,ffm_obj_grad,(int)10,(int)50,(float)1e-6);
    std::cout<<log_det<<std::endl;
    std::cout<<trace<<std::endl;

//    log_det = calculate_one_lanczos_triag(tridiag_matrix);
//    std::cout<<log_det<<std::endl;
//    std::cout<<b_inv<<std::endl;
//    std::cout<<tridiag_matrix<<std::endl;


//    X_data=X,X,ls,op,lambda,device_cuda
//    torch::Tensor output = FFM<float>(X,X,b,device_cuda,ls,op);
//    torch::Tensor output_ref = torch::zeros_like(output);
//    X = X.to(device_cuda);
//    b = b.to(device_cuda);
//    output_ref = output_ref.to(device_cuda);
//    rbf_call<float>(X, X, b, output_ref,ls,op, false);
//
//    std::cout<<output<<std::endl;
//    printf("--------------------------------------\n");
//    std::cout<<output_ref<<std::endl;
//    std::cout<<((output_ref-output)/output_ref).abs_().mean()<<std::endl;

}
