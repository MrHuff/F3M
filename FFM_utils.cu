//
// Created by rhu on 2020-07-05.
//

#pragma once
#include <torch/torch.h> //for n00bs like me, direct translation to python rofl
#include <npp.h>

template <typename scalar_t, int nd>
__global__ void box_division_cum(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> alpha,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> multiply,
        const scalar_t * int_mult,
        const scalar_t * edge,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> global_vector_counter_cum,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> perm
){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>X_data.size(0)-1){return;}
    int idx=0;
    for (int p = 0;p<nd;p++) {
        idx += multiply[p]*(int)floor( *int_mult * (X_data[i][p] - alpha[p]) / *edge);
    }
    atomicAdd(&global_vector_counter_cum[perm[idx]+1],1);

}

template <typename scalar_t, int nd>
__global__ void center_perm(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers_natural,
                            const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> alpha,
                            const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> multiply,
                            const scalar_t * int_mult,
                            const scalar_t * edge,
                                 torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> perm){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>centers_natural.size(0)-1){return;}
    int idx=0;
    for (int p = 0;p<nd;p++) {
        idx += multiply[p]*(int)floor( *int_mult * (centers_natural[i][p] - alpha[p]) / *edge);
    }
    perm[idx] = i;

}


template <typename scalar_t, int nd>
__global__ void box_division_assign(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> alpha,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> multiply,
        const scalar_t * int_mult,
        const scalar_t * edge,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> global_vector_counter_cum,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> perm,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> global_unique,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> sorted_index
){

    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>X_data.size(0)-1){return;}
    int idx=0;
    for (int p = 0;p<nd;p++) {
        idx += multiply[p]*(int)floor( *int_mult * (X_data[i][p] - alpha[p]) / *edge);
    }
    sorted_index[atomicAdd(&global_unique[perm[idx]],1)+global_vector_counter_cum[perm[idx]]] = i;
}

//

//box_idx , data_point_idx.
//0 : [- 1- 1- 1- 1-]
//1 : [- 1- -1- -1- -1- 1]
//global_vector_counter: [32 50 ... 64] cumsum [0 32 82 ... n]
//n [0 0 0 0 0 0 0 0 ] indices in the sorted box order
// new global vector counter... + cumsum[box_indx] + value of col.

__global__ void parse_x_boxes(
        const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> box_cumsum,
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> results
){
    int tid  = threadIdx.x + blockDim.x*blockIdx.x;
    int n = box_cumsum.size(0);
    if (tid<n){
        int box_nr = box_cumsum[tid][0];
        if (tid==0){
            results[box_nr][0] = 0;
            results[box_nr][1] = box_cumsum[tid][1];
        }else{
            results[box_nr][0] = box_cumsum[tid-1][1];
            results[box_nr][1] = box_cumsum[tid][1];
        }
    }
    __syncthreads();
}

template <typename scalar_t, int nd>
__global__ void get_cheb_idx_data(
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> cheb_nodes,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> cheb_data,
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> cheb_idx,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indices
){
    int n = cheb_data.size(0);
    int i = threadIdx.x+blockIdx.x*blockDim.x; // Thread nr
    if (i>n-1){return;}
    int sampled_i = indices[i];
    int lap_nodes = cheb_nodes.size(0);
    int idx;
    int tmp;
    extern __shared__ scalar_t buffer[];
    if (threadIdx.x<lap_nodes){
        buffer[threadIdx.x] = cheb_nodes[threadIdx.x];
    }
    __syncthreads();
    for (int j=0;j<nd;j++){
        tmp = sampled_i % (int)round(pow(lap_nodes,j+1));
        idx = (int) floor((float)tmp/(float)pow(lap_nodes,j));
        cheb_idx[i][j] = idx;
        cheb_data[i][j] = buffer[idx];
    }
}

template <typename scalar_t, int nd>
__global__ void get_smolyak_indices(
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> cheb_nodes,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> cheb_data,
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> cheb_idx,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> size_per_dim,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> cum_prod
){
    int n = cheb_data.size(0);
    int i = threadIdx.x+blockIdx.x*blockDim.x; // Thread nr
    if (i>n-1){return;}
    extern __shared__ scalar_t buffer[];
    int lap_nodes = cheb_nodes.size(0);
    int idx;
    int tmp;
    int acc = 0;
    if (threadIdx.x<lap_nodes){
        buffer[threadIdx.x] = cheb_nodes[threadIdx.x];
    }
    __syncthreads();
    for (int j=0;j<nd;j++){
        if (j==0){
            tmp = i % cum_prod[j];
            idx = (int) floor((float)tmp/1.0);
        }
        else{
            tmp = i % cum_prod[j];
            idx = (int) floor((float)tmp/(float)cum_prod[j-1]);
            acc += size_per_dim[j-1];
        }
        cheb_idx[i][j] = idx+acc;
        cheb_data[i][j] = buffer[cheb_idx[i][j]];
    }
}


template <int nd>
__global__ void get_centers(
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> centers
){
    int n = centers.size(0);
    int i = threadIdx.x+blockIdx.x*blockDim.x; // Thread nr
    if (i>n-1){return;}
    int idx;
    for (int j=0;j<nd;j++){
        idx = (int) floor((float)(i%(int)pow(2,j+1))/(float)pow(2,j));
        centers[i][j] = (idx==0) ? -1 : 1;
    }
}


template <typename scalar_t, int nd>
__global__ void boolean_separate_interactions(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers_X,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers_Y,
        const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> interactions,
        const scalar_t * edge,
        torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> is_far_field

){
    int i = threadIdx.x+blockIdx.x*blockDim.x; // Thread nr
    int n = interactions.size(0);
    if (i>n-1){return;}
    int by = interactions[i][1];
    int bx = interactions[i][0];
    scalar_t distance[nd];
    for (int k=0;k<nd;k++){
        distance[k]=centers_Y[by][k] - centers_X[bx][k];
    }
    if (get_2_norm<scalar_t,nd>(distance)>=(*edge*2+1e-6)){
        is_far_field[i]=true;
    }
}

template<typename scalar_t>
__inline__ __device__ scalar_t warpReduceMax(scalar_t val)
{
#pragma unroll
    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    {
        val = max(__shfl_xor_sync(0xFFFFFFFF, val, mask), val);
    }

    return val;
}
template<typename scalar_t>
__inline__ __device__ scalar_t warpReduceMin(scalar_t val)
{
#pragma unroll
    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    {
        val = min(__shfl_xor_sync(0xFFFFFFFF, val, mask), val);
    }

    return val;
}

__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
          __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}
__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
          __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
    return old;
}

template<typename scalar_t,int cols>
__global__ void reduceMaxMinOptimizedWarpMatrix(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
        torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> maxOut,
        torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> minOut
        )
{
    __shared__ scalar_t sharedMax;
    __shared__ scalar_t sharedMin;
    int size = input.size(0);
    int increment = gridDim.x*blockDim.x;
    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    for(int j=0;j<cols;j++){
        sharedMax = -NPP_MAXABS_32F;
        sharedMin = NPP_MAXABS_32F;
        scalar_t localMax = -NPP_MAXABS_32F;
        scalar_t localMin = NPP_MAXABS_32F;
        if (tid<size){
        for (int i = tid; i < size; i += increment) //iterate through warps...
        {
            if (input[i][j] > localMax)
            {
                localMax = input[i][j];
            }
            if (input[i][j] < localMin )
            {
                localMin = input[i][j];
            }
        }
        }
        __syncthreads();

        scalar_t warpMax = warpReduceMax(localMax);
        scalar_t warpMin = warpReduceMin(localMin);
        int lane = threadIdx.x % warpSize;
        if (lane == 0)
        {
            atomicMaxFloat(&sharedMax, warpMax);
            atomicMinFloat(&sharedMin, warpMin);
        }
        __syncthreads();
        if (0 == threadIdx.x)
        {
            atomicMaxFloat(&maxOut[j],sharedMax);
            atomicMinFloat(&minOut[j],sharedMin);
        }
        __syncthreads();
    }
}

__global__ void get_keep_mask(
        const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> interactions,
        torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> keep_x_box,
        torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> keep_y_box,
        torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> output
        ){
    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    if (tid>interactions.size(0)-1){return;}
    output[tid] = keep_x_box[interactions[tid][0]]*keep_y_box[interactions[tid][1]];
}