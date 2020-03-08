/*
*********************************************************************
function name: gpu_matrix_mult
description: dot product of two matrix (not only square)
parameters:
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C)
            to store the result
Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/

#pragma once
#include <iostream>
#include <ostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>
#include <stdexcept>
#define BLOCK_SIZE 32


__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m)
    {
        for(int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

/*
*********************************************************************
function name: gpu_square_matrix_mult
description: dot product of two matrix (not only square) in GPU
parameters:
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C)
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*********************************************************************
*/
//__global__ void gpu_square_matrix_mult(int *d_a, int *d_b, int *d_result, int n)
//{
//    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
//    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];
//
//    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
//    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
//    int tmp = 0;
//    int idx;
//
//    for (int sub = 0; sub < gridDim.x; ++sub)
//    {
//        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
//        if(idx >= n*n)
//        {
//            // n may not divisible by BLOCK_SIZE
//            tile_a[threadIdx.y][threadIdx.x] = 0;
//        }
//        else
//        {
//            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
//        }
//
//        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
//        if(idx >= n*n)
//        {
//            tile_b[threadIdx.y][threadIdx.x] = 0;
//        }
//        else
//        {
//            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
//        }
//        __syncthreads();
//
//        for (int k = 0; k < BLOCK_SIZE; ++k)
//        {
//            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
//        }
//        __syncthreads();
//    }
//    if(row < n && col < n)
//    {
//        d_result[row * n + col] = tmp;
//    }
//}

/*
*********************************************************************
function name: gpu_matrix_transpose
description: matrix transpose
parameters:
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*********************************************************************
*/
__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}
/*
*********************************************************************
function name: cpu_matrix_mult
description: dot product of two matrix (not only square) in CPU,
             for validating GPU results
parameters:
            &a CPU host pointer to a m X n matrix (A)
            &b CPU host pointer to a n X k matrix (B)
            &c CPU host output purpose pointer to a m X k matrix (C)
            to store the result
return: none
*********************************************************************
*/
void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) throw new std::runtime_error("");
    }
}
template<typename T>
struct cumatrix {
    // types:
//    typedef float T;
//    typedef T &reference;
//    typedef const T &const_reference;
//    typedef int int;
//    typedef T *pointer;



    T *elemns;
    T *d_elemns;
    bool in_device = false;
    int N, M;

    cumatrix(int n, int m) {
        N = n; M = m;
        elemns = new T[N * M];
        fill_rand();
    }
    cumatrix(const cumatrix<T>& copy_matrix)
    : N(copy_matrix.N),M(copy_matrix.M),in_device(copy_matrix.in_device)
    {
        std::cout<<"copy"<<std::endl;
        std::cout<<N<<std::endl;
        std::cout<<M<<std::endl;
        std::cout<<in_device<<std::endl;
        elemns = new T[N * M];
        gpuErrchk(cudaMemcpy(elemns,copy_matrix.elemns,N*M* sizeof(T),cudaMemcpyHostToHost)) //Use cuda functions for copying stuff!!
        if (in_device){
            std::cout<<"cuda copy"<<std::endl;
            gpuErrchk(cudaMalloc((void **) &d_elemns, N * M * sizeof(T)));
            gpuErrchk(cudaMemcpy(d_elemns,copy_matrix.d_elemns,N*M* sizeof(T),cudaMemcpyDeviceToDevice))
        }

    }
    cumatrix& operator=(cumatrix<T> &&rhs){
        std::cout<<"move ass"<<std::endl;

    }
    cumatrix& operator=(const cumatrix<T> &rhs){
        std::cout<<"copy ass"<<std::endl;

    }
    ~cumatrix() {
        if (in_device){
            release_device_data();
        }
        delete [] elemns;
    }
    constexpr int size() const noexcept { return (N * M); }
    constexpr int rows() const noexcept { return N; }
    constexpr int cols() const noexcept { return M; }
    T& operator[](int n) { return elemns[n]; }
    const T& operator[](int n) const { return elemns[n]; }
    T& operator()(int x, int y) { return elemns[x * M + y]; }
    const T& operator()(int x, int y) const {  return elemns[x * M + y]; }
    T *data() noexcept { return elemns; }
    T* get_device_pointer(bool copy = true)
    {
        if (!in_device) {
            gpuErrchk(cudaMalloc((void **) &d_elemns, N * M * sizeof(T)));
            if (copy) gpuErrchk(cudaMemcpy(d_elemns, elemns, N * M * sizeof(T), cudaMemcpyHostToDevice));
            in_device = true;
        }
        return d_elemns;
    };
    void refresh_from_device()
    {
        if (in_device) gpuErrchk(cudaMemcpy(elemns, d_elemns, N * M * sizeof(T), cudaMemcpyDeviceToHost));
    }
    ;
    void refresh_to_device()
    {
        if (in_device) gpuErrchk(cudaMemcpy(d_elemns, elemns, N * M * sizeof(T), cudaMemcpyHostToDevice));
    };
    void release_device_data()
    {
        if (in_device) {
            gpuErrchk(cudaFree(d_elemns));
            in_device = false;
        }
    };
    void fill_rand(){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dis(0, 1);
        for (int i = 0; i < size(); i++)
            (*this)[i] = dis(gen);
    };
};
template<typename T>
cumatrix<T> operator*(cumatrix<T>& a, cumatrix<T>& b) {
    cumatrix<T> output(a.rows(), b.cols());
    const T *d_a = a.get_device_pointer();
    const T *d_b = b.get_device_pointer();
    T *d_c = output.get_device_pointer(false);
    int N = a.rows(), M = b.cols(), K = a.cols();
    int lda = N, ldb = K, ldc = N;
    const float alpha = 1;
    const float beta = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc);

    cublasDestroy(handle);
    output.refresh_from_device();


    return output;
}
template<typename T>
std::ostream& operator << (std::ostream& out, const cumatrix<T>& mat){
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            out << " " << mat(i, j);
        }
        out << std::endl;
    }
    return out;
}