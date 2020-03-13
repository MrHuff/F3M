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


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) throw new std::runtime_error("");
    }
}
template<typename T>
struct cumatrix {

    T *elemns;
    T *d_elemns;
    bool in_device = false;
    int N, M;

    cumatrix(int n, int m) {
        N = n; M = m;
        elemns = new T[N * M];
        fill_rand();
        std::cout<<"constructor"<<std::endl;

    }
    cumatrix(const cumatrix<T>& copy_matrix) //copy constructor
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

    cumatrix(cumatrix<T> &&obj) noexcept {
        std::cout<<"move constructor"<<std::endl;
        this->elemns = obj.elemns;
        this->d_elemns = obj.d_elemns;
        in_device = obj.in_device;
        N=obj.N;
        M=obj.M;
        obj.elemns = nullptr;
        obj.d_elemns = nullptr;
    }

    cumatrix& operator=(const cumatrix<T> &rhs){ //actually copies the bad boy
        std::cout<<"copy ass"<<std::endl;
        return *this = cumatrix<T>(rhs);
    }

    cumatrix& operator=(cumatrix<T>&& other) noexcept{ //rvalues only!
        std::cout<<"move ass"<<std::endl;
        if (this != &other)
        {
            // Free the existing resource.
            delete[] elemns; //well you have to remove what you have to replace it
            this->release_device_data(); //well you have to remove what you have to replace it

            // Copy the data pointer and its length from the
            // source object.
            elemns = other.elemns;
            if (other.in_device){
                d_elemns = other.get_device_pointer();
                in_device = other.in_device;
//                other.release_device_data(); //dont destroy other's data! we just wanna move it around!
            }
            N = other.N;
            M = other.M;
            // Release the data pointer from the source object so that
            // the destructor does not free the memory multiple times.
            other.elemns = nullptr;
            other.d_elemns = nullptr;
        }
        return *this;
    }

    ~cumatrix() {
        std::cout<<"destroy"<<std::endl;
        if (d_elemns != nullptr){
            release_device_data();
        }
        if (elemns != nullptr){
            delete [] elemns;
        }
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
template<typename T>
cumatrix<T> test_move_constructor(cumatrix<T> a)
{
    return a;
}