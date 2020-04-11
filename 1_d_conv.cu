

#pragma once
#include <iostream>
#include <ostream>
#include <cublas_v2.h>
#include <random>
#include <stdexcept>
#include <ATen/ATen.h> //hardcore lib
#include <torch/torch.h> //for n00bs like me, direct translation to python rofl
#include <stdio.h>
#include <assert.h>
#define BLOCK_SIZE 192
#define MAXTHREADSPERBLOCK 1024
#define SHAREDMEMPERBLOCK 49152
#define nx 1000
#define ny 1000
#define nd 3
#define square_int 2
#define MAX_STREAMS 32

template<typename T>
std::tuple<dim3,dim3,int> get_kernel_launch_params(int cols,int height){
    dim3 blockSize;
    dim3 gridSize;
    int denonminator = std::max(1,(int) (cols*sizeof(T)));
    blockSize.x = min(BLOCK_SIZE,min(MAXTHREADSPERBLOCK,(int) ( (float)SHAREDMEMPERBLOCK / float(denonminator))));
    gridSize.x = height / blockSize.x + (height % blockSize.x == 0 ? 0 : 1);
    return std::make_tuple(blockSize,gridSize,blockSize.x * (cols+1) * sizeof(T));
};

__device__ float square(float x){
    return powf(x,square_int);
};

__device__ float rbf_simple(float x[],float y[]){
    float tmp=0;
    for (int k=0;k<nd;k++){
        tmp -= square(x[k]-y[k]);
    };
    return expf(tmp);
};


template <typename scalar_t>
__global__ void print_torch_cuda_1D(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > nx) { return; }
    for (int j = 0; j<nd; j++) {
        printf("%i: %f \n", i, X_data[i][j]);
    }
}
template <typename scalar_t>
__device__ static void torch_load_y(int index, scalar_t *shared_mem, const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> y){
#pragma unroll
    for (int k = 0; k < nd; k++) {
        //assert(&((*px)[i * FIRST + k]) != nullptr);
        shared_mem[nd * threadIdx.x + k] = y[index][k]; // First, load the i-th line of px[0]  -> shared_mem[ 0 : FIRST ].
        // Don't use thread id -> nvidia-chips doesn't work like that! It only got a third of the way
        // Some weird memory allocation issue
    }
}
template <typename scalar_t>
__device__ static void torch_load_b(int col_index ,int index, scalar_t *shared_mem, const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b){
    shared_mem[threadIdx.x] = b[index][col_index];
}

template <typename scalar_t>
__global__ void conv_1d_torch_rbf(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                                  const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                                  const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b,
                                  torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    float x_i[nd];
    float acc = 0.0;
    extern __shared__ float buffer[];
    float *yj = &buffer[0];
    float *bj = &buffer[blockDim.x*nd];
    if (i<nx) {
        for (int k = 0; k < nd; k++) {
            x_i[k] = X_data[i][k];
        }
    }
    for (int b_ind=0; b_ind<output.size(1); b_ind++) {
        for (int jstart = 0; jstart < ny; jstart += blockDim.x) {
            int j = jstart + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
            if (j < ny) { // we load yj from device global memory only if j<ny
                torch_load_y<scalar_t>(j, yj, Y_data);
                torch_load_b<scalar_t>(b_ind ,j, bj, b);
            }
            __syncthreads();
            //ok maybe its not top prio to fix this, maybe just use even threads for good reference...
            if (i < nx) { // we compute x1i only if needed
                float *yjrel = yj; // Loop on the columns of the current block.
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += nd) {
                    acc += rbf_simple(x_i, yjrel) *  bj[jrel]; //sums incorrectly cause pointer is fucked not sure if allocating properly
                }
            }
            __syncthreads(); //Lesson learned! Thread synching really important for cuda programming and memory loading when indices are dependent on threadIdx.x!
        };
        if (i < nx) {
            output[i][b_ind] = acc;
        }
        __syncthreads();
    };
}
template <typename scalar_t>
__global__ void rbf_1d_reduce_simple_torch(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                                           const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                                           const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b,
                                           torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>nx-1){return;}
    float x_i[nd];
    float y_j[nd];
    float acc=0.0;
    for (int k=0;k<nd;k++){
        x_i[k] = X_data[i][k];
    }
    for (int b_size=0; b_size<b.size(1);b_size++){
        for (int p=0;p<ny;p++){
            for (int k=0;k<nd;k++){
                y_j[k] = Y_data[p][k];
            };
            acc+= rbf_simple(x_i,y_j)*b[p][b_size];
        };
        output[i][b_size] = acc;
    }
};

template <typename scalar_t>
__global__ void near_field_rbf_shared(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                               const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                               const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
                               const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_indices,
                               torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    unsigned int y_n = Y_data.size(0);
    unsigned int x_n = X_indices.size(0);
    int x_ind = X_indices[i];
    float x_i[nd];
    float acc = 0.0;
    extern __shared__ float buffer[];
    float *yj = &buffer[0];
    float *bj = &buffer[blockDim.x*nd];
    if (i<x_n) {
        for (int k = 0; k < nd; k++) {
            x_i[k] = X_data[x_ind][k];
        }
    }
    for (int b_ind=0; b_ind<output.size(1); b_ind++) {
        for (int jstart = 0, tile = 0; jstart < y_n; jstart += blockDim.x, tile++) {
            int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
            if (j < y_n) { // we load yj from device global memory only if j<ny
                torch_load_y<scalar_t>(j, yj, Y_data);
                torch_load_b<scalar_t>(b_ind ,j, bj, b_data);
            }
            __syncthreads();
            //ok maybe its not top prio to fix this, maybe just use even threads for good reference...
            if (i < x_n) { // we compute x1i only if needed
                float *yjrel = yj; // Loop on the columns of the current block.
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < y_n - jstart); jrel++, yjrel += nd) {
                    acc += rbf_simple(x_i, yjrel) * bj[jrel]; //sums incorrectly cause pointer is fucked not sure if allocating properly
                }
            }
            __syncthreads(); //Lesson learned! Thread synching really important for cuda programming and memory loading when indices are dependent on threadIdx.x!
        };
        if (i < x_n) {
            atomicAdd(output[x_ind][b_ind],acc);
        }
        __syncthreads();
    };
}

template <typename scalar_t>
__global__ void near_field_rbf(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                                      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                                      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
                                      const int * X_indices,
                                      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    unsigned int y_n = Y_data.size(0);
    int x_ind = X_indices[i];
    float x_i[nd];
    float y_j[nd];
    float acc=0.0;
    for (int k=0;k<nd;k++){
        x_i[k] = X_data[x_ind][k];
    }
    for (int b_size=0; b_size<b_data.size(1);b_size++){
        for (int p=0;p<y_n;p++){
            for (int k=0;k<nd;k++){
                y_j[k] = Y_data[p][k];
            };
            acc+= rbf_simple(x_i,y_j)*b_data[p][b_size];
        };
        atomicAdd(output[x_ind][b_size],acc);
    }
};

