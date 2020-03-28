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
#include "n_roon.cuh"
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


//    print_row_mat(mat,nx,nd); //figure out how to move pointer of pointer to device!
//    print_mat(std::cout,mat,N,M);
//    auto cuda_mat_res = allocate_cuda_mat<float>(nx,ny);

    auto mat = generate_row_random_matrix(nx, nd); //matrix but contigous in the first element!
//    print_row_mat(mat,nx,nd);
    auto cuda_mat_x = allocate_cuda_mat<float>(nx,nd);
    host_to_cuda(cuda_mat_x,mat[0],nx,nd); //moves row major matrix to cuda memory

    auto b = generate_row_random_matrix(nx,1);
    auto cuda_b = allocate_cuda_mat<float>(nx,1);
    host_to_cuda(cuda_b,b[0],nx,1); //moves row major matrix to cuda memory
    auto cuda_b_res = allocate_cuda_mat<float>(nx,1);
    auto cuda_kernel_mat = allocate_cuda_mat<float>(nx,ny);
    auto cuda_kernel_mat_2 = allocate_cuda_mat<float>(nx,ny);

    auto cuda_b_res_2 = allocate_cuda_mat<float>(nx,1);


//    print_row_mat(b,nx,1); //figure out how to move pointer of pointer to device!

//    print_mat_cuda<float><<<grid_X,BLOCK_SIZE>>>(cuda_mat_x);
//    cudaDeviceSynchronize();
//    rbf_1d_kernel<<<grid_X,BLOCK_SIZE>>>(cuda_mat_x,cuda_mat_x,cuda_mat_res);
//    int grid_X = max((int)ceil((float)nx/(float)BLOCK_SIZE),1);
    dim3 blockSize;
    int denonminator = std::max(1,(int) (nd*sizeof(float)));
    blockSize.x = min(BLOCK_SIZE,min(MAXTHREADSPERBLOCK,(int) ( (float)SHAREDMEMPERBLOCK / float(denonminator))));
    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);
    printf("%i \n",gridSize.x );

    rbf_1d_reduce_shared<<<gridSize, blockSize,blockSize.x * (nd) * sizeof(float)>>>(cuda_mat_x, cuda_mat_x, cuda_b, cuda_b_res);
    cudaDeviceSynchronize();
    auto cpu_res = cuda_to_host_and_pointer(cuda_b_res,nx,1);
    print_mat(std::cout,cpu_res,nx,1); //ok does something but very incorrectly
    printf("--------------------------------------------------------------------------------------------\n");
    rbf_1d_reduce_simple<<<gridSize, blockSize,blockSize.x * (nd) * sizeof(float)>>>(cuda_mat_x, cuda_mat_x, cuda_b, cuda_b_res_2);
    cudaDeviceSynchronize();
    auto cpu_res_2 = cuda_to_host_and_pointer(cuda_b_res_2,nx,1);
    print_mat(std::cout,cpu_res_2,nx,1); //ok does something but very incorrectly


//    rbf_1d_kernel_shared<<<gridSize, blockSize,blockSize.x * (nd) * sizeof(float)>>>(cuda_mat_x, cuda_mat_x, cuda_kernel_mat);
//    cudaDeviceSynchronize();
//    auto cpu_res = cuda_to_host_and_pointer(cuda_kernel_mat,nx,ny);
//    print_mat(std::cout,cpu_res,nx,nx); //ok does something but very incorrectly
//    printf("\n");
//    rbf_1d_kernel<<<gridSize, blockSize,blockSize.x * (nd) * sizeof(float)>>>(cuda_mat_x, cuda_mat_x, cuda_kernel_mat_2);
//    cudaDeviceSynchronize();
//    auto cpu_res_2 = cuda_to_host_and_pointer(cuda_kernel_mat_2,nx,ny);
//    print_mat(std::cout,cpu_res_2,nx,nx); //ok does something but very incorrectly

//


//    auto cuda_kernel_mat = allocate_cuda_mat<float>(N,N);
//
//    std::cout<< grid_Y<<std::endl;
//    std::cout<< grid_X<<std::endl;
//
//    dim3 grid(grid_Y,grid_X);
//    dim3 block_dim(BLOCK_SIZE,BLOCK_SIZE);
//    host_to_cuda(cuda_mat,mat,N,M);
//    print_mat_cuda<float><<<grid,block_dim>>>(cuda_mat,M,N);
//    cudaDeviceSynchronize();


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