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

//    torch::Tensor X_train = read_csv<float>("X_train_PCA.csv",11619,3); something wrong with data probably...
//    torch::Tensor b_train = read_csv<float>("Y_train.csv",11619,1); something wrong with data probably...
    torch::Tensor X_train = torch::rand({20000,nd});
    torch::Tensor b_train = torch::randn({20000,1});
//
    float ls = 1.0;
    float lambda = 1e-2;
    int T = 10;
    int max_its = 50;
    float tol = 1e-6;
    rbf_pointer<float> op,op_grad;
    cudaMemcpyFromSymbol(&op, rbf_pointer_func<float>, sizeof(rbf_pointer<float>)); //rbf_pointer_func,rbf_pointer_grad
    cudaMemcpyFromSymbol(&op_grad, rbf_pointer_grad<float>, sizeof(rbf_pointer<float>)); //rbf_pointer_func,rbf_pointer_grad
    torch::Tensor res,res_ref;
    //
    FMM_obj<float> ffm_obj = FMM_obj<float>(X_train,X_train,ls,op,lambda,device_cuda);
//    FMM_obj<float> ffm_obj_grad = FMM_obj<float>(X,X,ls,op_grad,lambda,device_cuda);
    exact_MV<float> ffm_obj_exact = exact_MV<float>(X_train,X_train,ls,op,lambda,device_cuda);
//    exact_MV<float> ffm_obj_grad_exact = exact_MV<float>(X,X,ls,op_grad,lambda,device_cuda);

    auto start = std::chrono::high_resolution_clock::now();
    res = ffm_obj*b_train;
    auto end = std::chrono::high_resolution_clock::now();
    res_ref = ffm_obj_exact*b_train;
    auto end_2 = std::chrono::high_resolution_clock::now();
    auto duration_1 = std::chrono::duration_cast<std::chrono::seconds>(end-start);
    auto duration_2 = std::chrono::duration_cast<std::chrono::seconds>(end_2-end);
    std::cout<<duration_1.count()<<std::endl;
    std::cout<<duration_2.count()<<std::endl;


////    torch::Tensor output = ffm_obj_test*b;
////    exact_MV<float> exact_obj_test = exact_MV<float>(X,X,ls,op,lambda,device_cuda);
////    torch::Tensor output_ref = exact_obj_test*b;
//    torch::Tensor loss,grad,b_inv;
////    std::tie(b_inv,tridiag_matrix) = CG(ffm_obj_test,b,(float) 1e-6,(int) 100,true);
////    std::tie(log_det,trace) = trace_and_log_det_calc(ffm_obj,ffm_obj_grad,(int)10,(int)50,(float)1e-6);
////    std::cout<<log_det<<std::endl;
////    std::cout<<trace<<std::endl;
//    std::tie(loss,grad,b_inv)=calculate_loss_and_grad<float>(ffm_obj,ffm_obj_grad,b,T,max_its,tol);
//
////    log_det = calculate_one_lanczos_triag(tridiag_matrix);
//    std::cout<<loss<<std::endl;
//    std::cout<<grad<<std::endl;
//    std::cout<<b_inv<<std::endl;


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
