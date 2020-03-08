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
    int n = 10;
    int m = 10;
    cumatrix<float> a(n, m), b(n, m);
    std::cout << a;
//    std::cout << a*b;
//
    cumatrix<float> c = a * b;
    std::cout << c;


//    int m, n, k;
//    /* Fixed seed for illustration */
//    srand(3333);
////    printf("please type in m n and k\n");
////    scanf("%d %d %d", &m, &n, &k);
//
//    m=1000;
//    n=1000;
//    k=1000;
//
//    // allocate memory in host RAM, h_cc is used to store CPU result
//    int *h_a, *h_b, *h_c, *h_cc;
//    cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
//    cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
//    cudaMallocHost((void **) &h_c, sizeof(int)*m*k);
//    cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);
//
//    // random initialize matrix A
//    for (int i = 0; i < m; ++i) {
//        for (int j = 0; j < n; ++j) {
//            h_a[i * n + j] = rand() % 1024;
//        }
//    }
//
//    // random initialize matrix B
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < k; ++j) {
//            h_b[i * k + j] = rand() % 1024;
//        }
//    }
//
//    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
//
//    // some events to count the execution time
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//
//    // start to count execution time of GPU version
//    cudaEventRecord(start, 0);
//    // Allocate memory space on the device
//    int *d_a, *d_b, *d_c;
//    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
//    cudaMalloc((void **) &d_b, sizeof(int)*n*k);
//    cudaMalloc((void **) &d_c, sizeof(int)*m*k);
//
//    // copy matrix A and B from host to device memory
//    cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);
//
//    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
//    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
//    dim3 dimGrid(grid_cols, grid_rows);
//    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//
//    // Launch kernel
//    if(m == n && n == k)
//    {
//        gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
//    }
//    else
//    {
//        gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
//    }
//    // Transefr results from device to host
//    cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
//    cudaDeviceSynchronize();
//    // time counting terminate
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//
//    // compute time elapse on GPU computing
//    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
//    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);
//
//    // start the CPU version
//    cudaEventRecord(start, 0);
//
//    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);
//
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
//    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);
//
//    // validate results computed by GPU
//    int all_ok = 1;
//    for (int i = 0; i < m; ++i)
//    {
//        for (int j = 0; j < k; ++j)
//        {
//            //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
//            if(h_cc[i*k + j] != h_c[i*k + j])
//            {
//                all_ok = 0;
//            }
//        }
//        //printf("\n");
//    }
//
//    // roughly compute speedup
//    if(all_ok)
//    {
//        printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
//    }
//    else
//    {
//        printf("incorrect results\n");
//    }
//
//    // free memory
//    cudaFree(d_a);
//    cudaFree(d_b);
//    cudaFree(d_c);
//    cudaFreeHost(h_a);
//    cudaFreeHost(h_b);
//    cudaFreeHost(h_c);
//    cudaFreeHost(h_cc);
    return 0;
}