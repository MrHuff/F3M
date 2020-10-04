//
// Created by rhu on 2020-07-05.
//

#pragma once
#include <torch/torch.h> //for n00bs like me, direct translation to python rofl
#include <npp.h>
template <typename scalar_t, int nd>
__global__ void box_division(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> b,
                             torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> old_indices,
                             torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> global_vector_counter_cum,
                             const int * divide_num
){

    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>X_data.size(0)-1){return;}
    int old_ind = old_indices[i]; //Plan A. Group threads. Plan B. Bitonic sort
    old_indices[i] = old_ind* *divide_num;
    for (int k=0;k<nd;k++){
        old_indices[i]+= (centers[old_ind][k]<=X_data[i][k])*b[k];
    }
    atomicAdd(&global_vector_counter_cum[old_indices[i]+1],1);
}

__global__ void  group_index(
                             torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> old_indices,
                             torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> global_vector_counter_cum,
                             torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> global_unique,
                             torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> sorted_index
){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>old_indices.size(0)-1){return;}
    int box_ind = old_indices[i];
    int index = atomicAdd(&global_unique[box_ind],1);
    sorted_index[index+global_vector_counter_cum[box_ind]] = i;
}


//

//box_idx , data_point_idx.
//0 : [- 1- 1- 1- 1-]
//1 : [- 1- -1- -1- -1- 1]
//global_vector_counter: [32 50 ... 64] cumsum [0 32 82 ... n]
//n [0 0 0 0 0 0 0 0 ] indices in the sorted box order
// new global vector counter... + cumsum[box_indx] + value of col.


template <
        typename    Key,
        int         BLOCK_THREADS,
        int         ITEMS_PER_THREAD>
__launch_bounds__ (BLOCK_THREADS)
__global__ void BlockSortKernel(
        Key         *d_in,          // Tile of input
        Key         *d_out,
        Key         *d_val
        // Tile of output
        )     // Elapsed cycle count of block scan
{
    enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
    // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    // Specialize BlockRadixSort type for our thread block
    typedef BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD,Key> BlockRadixSortT;
    // Shared memory
    __shared__ union TempStorage
    {
        typename BlockLoadT::TempStorage        load;
        typename BlockLoadT::TempStorage        load_val;
        typename BlockRadixSortT::TempStorage   sort;
    } temp_storage;
    // Per-thread tile items
    Key items[ITEMS_PER_THREAD];
    Key values[ITEMS_PER_THREAD];
    // Our current block's offset
    int block_offset = blockIdx.x * TILE_SIZE;
    // Load items into a blocked arrangement
    BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);
    // Barrier for smem reuse
    __syncthreads();
    BlockLoadT(temp_storage.load_val).Load(d_val + block_offset, values);
    __syncthreads();

    // Sort keys
    BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(items,values);
    // Store output in striped fashion
    StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + block_offset, values);
    // Store elapsed clocks
}




__global__ void parse_x_boxes(
        const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> box_cumsum,
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> results
){
    unsigned int nr_x_boxes = results.size(0);
    int i = threadIdx.x+blockIdx.x*blockDim.x; // Thread nr
    if (i>nr_x_boxes-1){return;}
    unsigned int nr_of_relevant_boxes = box_cumsum.size(0);
    for (int j=0;j<nr_of_relevant_boxes;j++){
        if(i==box_cumsum[j][0]){ //if match
            if (j==0){
                results[i][0]=0;
            }else{
                results[i][0]=box_cumsum[j-1][1];
            }
            results[i][1]=box_cumsum[j][1];

        }
    }
    __syncthreads();
}

template <typename scalar_t, int nd>
__global__ void get_cheb_idx_data(
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> cheb_nodes,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> cheb_data,
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> cheb_idx
){
    int n = cheb_data.size(0);
    int i = threadIdx.x+blockIdx.x*blockDim.x; // Thread nr
    if (i>n-1){return;}
    int lap_nodes = cheb_nodes.size(0);
    extern __shared__ scalar_t buffer[];
    if (threadIdx.x<lap_nodes){
        buffer[threadIdx.x] = cheb_nodes[threadIdx.x];
    }
    int idx;
    __syncthreads();
    for (int j=0;j<nd;j++){
        idx = (int) floor((i%(int)pow(lap_nodes,j+1))/pow(lap_nodes,j));
        cheb_idx[i][j] = idx;
        cheb_data[i][j] = cheb_nodes[idx];
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
        idx = (int) floor((i%(int)pow(2,j+1))/pow(2,j));
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
    const unsigned int FULL_MASK = 0xffffffff;
#pragma unroll
    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    {
        val = max(__shfl_xor_sync(FULL_MASK, val, mask), val);
    }

    return val;
}
template<typename scalar_t>
__inline__ __device__ scalar_t warpReduceMin(scalar_t val)
{
    const unsigned int FULL_MASK = 0xffffffff;
#pragma unroll
    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    {
        val = min(__shfl_xor_sync(FULL_MASK, val, mask), val);
    }

    return val;
}


template <class T >
__inline__ __device__ int warpBroadcast(T val, int predicate)
{
    const unsigned int FULL_MASK = 0xffffffff;

    unsigned int mask = __ballot_sync(FULL_MASK, predicate);

    int lane = 0;
    for (;!(mask & 1); ++lane)
    {
        mask >>= 1;
    }

    return __shfl_sync(FULL_MASK, val, lane);
}

__global__ void reduceMaxIdxOptimizedWarp(const float* __restrict__ input, const int size, float* maxOut, int* maxIdxOut)
{
    float localMax = 0.f;
    int localMaxIdx = 0;

    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        float val = input[i];

        if (localMax < abs(val))
        {
            localMax = abs(val);
            localMaxIdx = i;
        }
    }

    const float warpMax = warpReduceMax(localMax);

    const int warpMaxIdx = warpBroadcast(localMaxIdx, warpMax == localMax);

    const int lane = threadIdx.x % warpSize;

    if (lane == 0)
    {
        int warpIdx = threadIdx.x / warpSize;
        maxOut[warpIdx] = warpMax;
        maxIdxOut[warpIdx] = warpMaxIdx;
    }
}

__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
          __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;

}__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
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
    if(threadIdx.x+blockDim.x*blockIdx.x>size-1){return;}
    for(int j=0;j<cols;j++){
        scalar_t localMax = -NPP_MAXABS_32F;
        scalar_t localMin = NPP_MAXABS_32F;
        for (int i = threadIdx.x+blockDim.x*blockIdx.x; i < size; i += increment) //iterate through warps...
        {
            if (localMax < input[i][j])
            {
                localMax = input[i][j];
            }
            if (localMin > input[i][j])
            {
                localMin = input[i][j];
            }
        }
        __syncthreads();

        const scalar_t warpMax = warpReduceMax(localMax);
        const scalar_t warpMin = warpReduceMin(localMin);
        const int lane = threadIdx.x % warpSize;
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

    }
}
//template < typename TYPE >
//struct PLUS_INFINITY;
//
//
//template <>
//struct PLUS_INFINITY< float > {
//    static constexpr float value = INFINITY_FLOAT;
//};
//
//template <>
//struct PLUS_INFINITY< double > {
//    static constexpr double value = INFINITY_DOUBLE;
//};
