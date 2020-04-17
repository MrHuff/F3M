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

    torch::Tensor X = read_csv<float>("X.csv",1000,3);
    torch::Tensor b = read_csv<float>("V.csv",1000,2);
    std::cout<<X<<std::endl;
    std::cout<<b<<std::endl;

    torch::Tensor output = FFM(X,X,b,device_cuda);
    torch::Tensor output_ref = torch::zeros_like(output);
    X = X.to(device_cuda);
    b = b.to(device_cuda);
    output_ref = output_ref.to(device_cuda);
    rbf_call<float>(X, X, b, output_ref,false);
    output_ref = output_ref.to("cpu");
    std::cout<<output<<std::endl;
    printf("--------------------------------------\n");
    std::cout<<output_ref<<std::endl;
    std::cout<<((output_ref-output)/output_ref).abs_().mean()<<std::endl;

}
