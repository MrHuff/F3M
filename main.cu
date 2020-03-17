/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *
 */
//#include <stdio.h>
#include "matrix.cu"

/*
*********************************************************************
function name: main
description: test and compare
parameters:
            none
return: none
*********************************************************************
*/

int main(int argc, char const *argv[])
{
    int N=64;
    int M =2;
    auto mat = generate_row_major_random_matrix(N, M);
//    print_mat(std::cout,mat,N,M);
    auto cuda_mat = allocate_cuda_mat<float>(N,M);
    auto cuda_kernel_mat = allocate_cuda_mat<float>(N,N);
    int grid_X = max((int)ceil((float)N/(float)BLOCK_SIZE),1);

    std::cout<< grid_X<<std::endl;
    dim3 grid(grid_X,grid_X);
    dim3 block_dim(BLOCK_SIZE,BLOCK_SIZE);
    host_to_cuda(cuda_mat,mat,N,M);
    print_mat_cuda<float><<<N,BLOCK_SIZE>>>(cuda_mat,M,N);
    cudaDeviceSynchronize();


//    int n = 10;
//    int m = 10;
//    cumatrix<float> a(n, m), b(n, m),f(1,1); //this calls the constructors
//    cumatrix<float> c = a * b; //Do matmul, constructor, we initialize a new cumatrix<float> which is c in the function.
//
////    f = c; //this calls the copy ass
////    cumatrix<float> bub(a); //this calls the copy constructor
////    cumatrix<float> foo = test_move_constructor(cumatrix<float>(1,1)); //this calls the move constructor
////    c =  test_move_constructor(cumatrix<float>(1,1)); //this calls the move assignment op
    return 0;
}