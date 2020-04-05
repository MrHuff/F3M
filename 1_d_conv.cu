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



dim3 get_blocksize(){
    dim3 blockSize;
    int denonminator = std::max(1,(int) (nd*sizeof(float)));
    blockSize.x = min(BLOCK_SIZE,min(MAXTHREADSPERBLOCK,(int) ( (float)SHAREDMEMPERBLOCK / float(denonminator))));
    return blockSize;
};
dim3 get_gridsize(dim3 blockSize){
    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);
    return gridSize;
}

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
__device__ static void torch_load(int c, scalar_t *xi, const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> y){
#pragma unroll
    for (int k = 0; k < nd; k++) {
        //assert(&((*px)[i * FIRST + k]) != nullptr);
        xi[nd*threadIdx.x+k] = y[c][k]; // First, load the i-th line of px[0]  -> xi[ 0 : FIRST ].
        // Don't use thread id -> nvidia-chips doesn't work like that! It only got a third of the way
        // Some weird memory allocation issue
    }
}

template <typename scalar_t>
__global__ void conv_1d_torch_rbf(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                                  const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                                  const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b,
                                  torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    float x_i[nd];
    float acc = 0.0;
    extern __shared__ float yj[];
    if (i<nx) {
        for (int k = 0; k < nd; k++) {
            x_i[k] = X_data[i][k];
        }
    }
    for (int b_ind=0; b_ind<output.size(1); b_ind++) {
        for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {
            int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
            if (j < ny) { // we load yj from device global memory only if j<ny
                torch_load<scalar_t>(j, yj, Y_data);
            }
            __syncthreads();
            //ok maybe its not top prio to fix this, maybe just use even threads for good reference...
            if (i < nx) { // we compute x1i only if needed
                float *yjrel = yj; // Loop on the columns of the current block.
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += nd) {
                    acc += rbf_simple(x_i, yjrel) * b[jrel + jstart][b_ind]; //sums incorrectly cause pointer is fucked not sure if allocating properly
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


