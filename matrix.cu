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
#include <cublas_v2.h>
#include <random>
#include <stdexcept>
#include <stdio.h>
#define BLOCK_SIZE 192
#define MAXTHREADSPERBLOCK 1024
#define SHAREDMEMPERBLOCK 49152
#define nx 1000
#define ny 1000
#define nd 3
#define square_int 2
#include <assert.h>

template<typename T>
void host_to_cuda(T *cuda_pointer, T* const host_pointer ,int N,int M){
    cudaMemcpy(cuda_pointer, host_pointer, sizeof(T)*N*M, cudaMemcpyHostToDevice);
};
template<typename T>
void cuda_to_host(T* const cuda_pointer, T *host_pointer ,int N,int M){
    cudaMemcpy(host_pointer,cuda_pointer, sizeof(T)*N*M, cudaMemcpyDeviceToHost);
};
template<typename T>
T* cuda_to_host_and_pointer(T* const cuda_pointer ,int N,int M){
    T *host_pointer = new T[N*M];
    cudaMemcpy(host_pointer,cuda_pointer, sizeof(T)*N*M, cudaMemcpyDeviceToHost);
    return host_pointer;
};

template<typename T>
T* allocate_cuda_mat(int N,int M){
    T *d;
    cudaMalloc((void **) &d, sizeof(T)*N*M);
    return d;
};

void print_row_mat(float ** mat, int N, int M){
    for (int i = 0; i < N; i++) {
        printf("row %i :",i);
//        float * tmp = mat[i];
        for (int j = 0; j < M; j++) {
            printf(" %f ",mat[i][j]);
            }
        printf("\n");
        }
}

void print_mat(std::ostream& out,float* const mat, int N, int M){
    for (int i = 0; i < N; i++) {
        printf("row %i :",i);
        for (int j = 0; j < M; j++) {
            printf(" %f ",mat[i * M + j]);
            if (j==(M-1)){
                printf("\n");
            }

//            printf("i: %i, j: %i, val: %f \n",i,j,mat[i * M + j]);
//            out << " " << mat[i * M + j]; //Most cancer thing about C++ and cuda. Row major and column major matrices. Imagine tensors rip.
//            out << " " << i * M + j;
//            printf("i: %i, j: %i ",i,j);
        }
    }
};

//Stare at keops code, think idea is to do 1d block calculations and let each block "iterate over the data" in y direction
//
template <typename T>
__global__ void print_mat_cuda(T* const mat){
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i>nx*nd-1){return;}
    printf("%i: %f \n",i,mat[i]);
}
float** generate_row_random_matrix(int const N, int const M){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0, 1);
    auto ** A = new float*[N];
    A[0] = new float[N * M];
    for (int i = 1; i < N; ++i) A[i] = A[i-1] + M;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
//            A[i][j] = i*M+j;
            A[i][j] = dis(gen);

        }
    }
    return A;
}


float* generate_row_major_random_matrix(int const N, int const M){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0, 2);
    auto *mat = new float[N*M];
    for (int i = 0; i < N; i++) { //for rows
        for (int j = 0; j < M; j++) { //for columns, M = width
            mat[i * M + j] = dis(gen);
//            printf("i: %i, j: %i, val: %f \n",i,j,mat[i * M + j]);
        }
    }
    return mat;
}
__device__ float square(float x){
    return powf(x,square_int);
};

__device__ float rbf_simple(float x[],float y[]){
    float tmp=0;
    for (int k=0;k<nd;k++){
//        printf("%f",square(x[k]-y[k]));
//        printf("%f\n",x[k]-y[k]);

        tmp -= square(x[k]-y[k]);
    };
//    printf("%f \n",tmp);

    return expf(tmp);
};


//blockDim,gridDim = give dim of block, grid
//blockIdx,threadIdx = is specific index of thread and block. Grid have no idx obv
//The execution configuration (of a global function call) is specified by inserting an expression of the form <<<Dg,Db,Ns,S>>>, where:
//
//Dg (dim3) specifies the dimension and size of the grid.
//Db (dim3) specifies the dimension and size of each block
//Ns (size_t) specifies the number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory.
//S (cudaStream_t) specifies the associated stream, is an optional parameter which defaults to 0.
// figure out way to provide constant dimensions
// Compute on device : grid and block are both 1d
//GpuConv1D_ranges.cu block sparse version
//GpuConv1D.cu regular
//Maybe start with naive global memory loop
//Do simulated 2-d, flatten is the way to go.

__device__ static void load(int c, float *xi, const float *px) {
    assert(xi != nullptr);
    assert(px != nullptr);
    /*
     * px is an "array" of pointers to data arrays of appropriate sizes.
     * That is, px[0] = *px     is a pointer to a TYPE array of size Ni * FIRST
     * Then,    px[1] = *(px+1) is a pointer to a TYPE array of size Ni * NEXT::FIRST; etc.
     *
     * (where Ni is the max value of "i" you should expect)
     * Obviously, we do not make any sanity check... so beware of illicit memory accesses !
     */
#pragma unroll
    for (int k = 0; k < nd; k++) {
        //assert(&((*px)[i * FIRST + k]) != nullptr);
        xi[nd*threadIdx.x+k] = px[c*nd + k]; // First, load the i-th line of px[0]  -> xi[ 0 : FIRST ].
    }
}


__global__ void rbf_1d_kernel_shared(const float *input_x,const float *input_y, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    float x_i[nd];
    extern __shared__ float yj[];
    for (int k = 0; k < nd; k++) {
        x_i[k] = input_x[i * nd + k];
    }
    for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {
        // get the current column
        int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass.
        printf("%i\n",j);
        if (j < ny) { // we load yj from device global memory only if j<ny
            load(j,yj,input_y);
        }
        __syncthreads();

//                printf("%f %f %f \n",yj[0],yj[1],yj[2]);
//        printf("%f %f %f \n",yj[3],yj[4],yj[5]);
        if (i < nx) { // we compute x1i only if needed
            float *yjrel = yj; // Loop on the columns of the current block.
            for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += nd) {
                output[i*ny+jrel+tile*blockDim.x] = rbf_simple(x_i,yjrel);
//                printf(" %f \n",yjrel[0]);
            }
            __syncthreads();
        }
    };
}


__global__ void rbf_1d_reduce_shared(const float *input_x, const float *input_y, const float *b, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    float x_i[nd];
    float acc = 0.0;
    extern __shared__ float yj[];
//    printf("thread% i \n",i);
    for (int k = 0; k < nd; k++) {
        x_i[k] = input_x[i * nd + k];
    }
    for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {
        // get the current column
//        printf("%i \n",jstart); //Sums incorrectly!
        int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
//        printf("%i \n",j); //a subset of y_data is not loaded for two threads!

        if (j < ny) { // we load yj from device global memory only if j<ny
            load(j,yj,input_y);
        }
        __syncthreads();


        if (i < nx) { // we compute x1i only if needed
            float *yjrel = yj; // Loop on the columns of the current block.
            for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += nd) {
                acc+= rbf_simple(x_i,yjrel)*b[jrel+jstart]; //sums incorrectly cause pointer is fucked not sure if allocating properly
            if (i==960){
//                printf("%f %f %f  \n",yjrel[0],yjrel[1],yjrel[2]); //For some reason gives incorrect
////                printf("%f %f %f  \n",x_i[0],x_i[1],x_i[2]);
//                printf("k=%f  b=%f  \n",rbf_simple(x_i,yjrel),b[jrel+jstart]);
                printf("%i: %f y: %f  %f  %f  \n",jrel+jstart,acc,yjrel[0],yjrel[1],yjrel[2]); //pointer yjrel acting funny, not pointing to right place in data! Remember 64 + 192 = 256!

            }

            }
            __syncthreads();
        }
        if (i < nx) {
            output[i] = acc;
        }
    };
}

__global__ void rbf_1d_reduce_simple(const float *input_x, const float *input_y, const float *b, float *output){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>nx-1){return;}
    float x_i[nd];
    float y_j[nd];
    float acc=0.0;
//    printf("thread% i \n",i);
    for (int k=0;k<nd;k++){
        x_i[k] = input_x[i*nd+k];
    }
    for (int p=0;p<ny;p++){
        for (int k=0;k<nd;k++){
            y_j[k] = input_y[p*nd+k];
        };
        acc+= rbf_simple(x_i,y_j)*b[p];
        if (i==960){
//            printf("%f %f %f  \n",y_j[0],y_j[1],y_j[2]);
////            printf("%f %f %f  \n",x_i[0],x_i[1],x_i[2]);
//            printf("k=%f  b=%f  \n",rbf_simple(x_i,y_j),b[p]); //k is being weird...
            printf("%i: %f y:  %f  %f  %f  \n",p,acc,y_j[0],y_j[1],y_j[2]);

        }
    };
    output[i] = acc;
};
__global__ void rbf_1d_kernel(const float *input_x,const float *input_y, float *output){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>nx-1){return;}
    float x_i[nd];
    float y_j[nd];
//    printf("thread% i \n",i);

    for (int k=0;k<nd;k++){
        x_i[k] = input_x[i*nd+k];
    }

    for (int p=0;p<ny;p++){
        for (int k=0;k<nd;k++){
            y_j[k] = input_y[p*nd+k];
        };
        output[i*ny+p] = rbf_simple(x_i,y_j);
    };
};
//

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