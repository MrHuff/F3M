

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
#define cube_int 3

#define MAX_STREAMS 32
#define laplace_nodes 5



template<typename T>
using rbf_pointer = T (*) (T[], T[],const T *);

template<typename T>
std::tuple<dim3,dim3,int> get_kernel_launch_params(int cols,int height){
    dim3 blockSize;
    dim3 gridSize;
    int denonminator = std::max(1,(int) (cols*sizeof(T)));
    blockSize.x = min(BLOCK_SIZE,min(MAXTHREADSPERBLOCK,(int) ( (float)SHAREDMEMPERBLOCK / float(denonminator))));
    gridSize.x = height / blockSize.x + (height % blockSize.x == 0 ? 0 : 1);
    return std::make_tuple(blockSize,gridSize,blockSize.x * (cols+1) * sizeof(T));
};

template<typename T>
std::tuple<dim3,dim3,int,torch::Tensor> skip_kernel_launch(int cols,
        int & blksize,
        torch::Tensor & box_sizes,
        torch::Tensor & box_idx){
    dim3 blockSize;
    dim3 gridSize;
    blockSize.x = blksize;
    std::vector<int> block_box_idx={};
    auto box_size_accessor = box_sizes.accessor<int,1>();
    auto box_idx_accessor = box_idx.accessor<int,1>();

    int n = box_sizes.size(0);
    int size,boxes_needed;
    for (int i=0;i<n-1;i++){
        size = box_size_accessor[i+1];
        boxes_needed = (int)ceil((float)size/(float)blksize);
//        std::cout<<size<<std::endl;
//        std::cout<<boxes_needed<<std::endl;
        for (int j=0;j<boxes_needed;j++){
            block_box_idx.push_back(box_idx_accessor[i]);
        }
    }
    std::cout<<block_box_idx<<std::endl;
    int total_blocks_needed =  block_box_idx.size();
    gridSize.x =total_blocks_needed;
    torch::Tensor output_block = torch::zeros({total_blocks_needed}).toType(torch::kInt32);
    auto block_box_idx_accessor = output_block.accessor<int,1>();
    for (int i = 0;i<total_blocks_needed;i++){
        block_box_idx_accessor[i] =  block_box_idx[i];
    }

    //from_blob not fucking working...

    return std::make_tuple(blockSize,gridSize,blockSize.x * (cols+1) * sizeof(T),output_block);
};

template<typename T>
__device__ T square(T x){
    return powf(x,square_int);
};

template<typename T>
__device__ T cube(T x){
    return powf(x,cube_int);
};

template<typename T>
__device__ T rbf_simple(T x[],T y[]){
    T tmp=0;
    for (int k=0;k<nd;k++){
        tmp += square<T>(x[k]-y[k]);
    };
    return expf(-tmp);
};

template<typename T>
__device__ T rbf(T x[],T y[],const T *ls){
    T tmp=0;
    for (int k=0;k<nd;k++){
        tmp += square<T>(x[k]-y[k]);
    };
    return expf(-tmp/square<T>(*ls));
};
template<typename T>
__device__ T rbf_grad(T x[],T y[],const T *ls){
    T tmp=0;
    for (int k=0;k<nd;k++){
        tmp += square<T>(x[k]-y[k]);
    };
    return expf(-tmp/square<T>(*ls))*tmp/cube<T>(*ls);
};
template<typename T>
__device__ rbf_pointer<T> rbf_pointer_func = rbf<T>;
template<typename T>
__device__ rbf_pointer<T> rbf_pointer_grad = rbf_grad<T>;


template <typename scalar_t>
__global__ void print_test(const scalar_t * data) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > nx) { return; }
    printf("%i: %f \n", i, data[i]);
}

template <typename scalar_t>
__global__ void edit_test(const scalar_t * data) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > nx) { return; }
    atomicAdd(&data[i],1);
}


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
__device__ static void torch_load_b(
        int col_index,
        int index,
        scalar_t *shared_mem,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b){
    shared_mem[threadIdx.x] = b[index][col_index];
}


//Consider caching the kernel value if b is in Nxd.
template <typename scalar_t>
__global__ void rbf_1d_reduce_shared_torch(
                                const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                               const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                               const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
                               torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                                scalar_t * ls,
                                rbf_pointer<scalar_t> op){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    unsigned int x_n = X_data.size(0);
    scalar_t x_i[nd];
    if (i<x_n) {
        for (int k = 0; k < nd; k++) {
            x_i[k] = X_data[i][k];
        }
    }
    unsigned int y_n = Y_data.size(0);
    extern __shared__ scalar_t buffer[];
    scalar_t *yj = &buffer[0];
    scalar_t *bj = &buffer[blockDim.x*nd];
    scalar_t acc;
    for (int b_ind=0; b_ind<output.size(1); b_ind++) {
        acc=0.0;
        for (int jstart = 0, tile = 0; jstart < y_n; jstart += blockDim.x, tile++) {
            int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
            if (j < y_n) { // we load yj from device global memory only if j<ny
                torch_load_y<scalar_t>(j, yj, Y_data);
                torch_load_b<scalar_t>(b_ind ,j, bj, b_data);
            }
            __syncthreads();
            if (i < x_n) { // we compute x1i only if needed
                scalar_t *yjrel = yj; // Loop on the columns of the current block.
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < y_n - jstart); jrel++, yjrel += nd) {
                    acc += (*op)(x_i, yjrel,ls) * bj[jrel]; //sums incorrectly cause pointer is fucked not sure if allocating properly
                }
            }
            __syncthreads(); //Lesson learned! Thread synching really important for cuda programming and memory loading when indices are dependent on threadIdx.x!
        };
        if (i < x_n) {
            output[i][b_ind] = acc;
        }
        __syncthreads();
    };
}

template <typename scalar_t>
__global__ void rbf_1d_reduce_simple_torch(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                                      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                                      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
                                      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                                      scalar_t * ls,
                                      rbf_pointer<scalar_t> op){

    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    unsigned int x_n = X_data.size(0);
    if (i>x_n-1){return;}
    unsigned int y_n = Y_data.size(0);
    scalar_t x_i[nd];
    scalar_t y_j[nd];
    scalar_t acc;
    for (int k=0;k<nd;k++){
        x_i[k] = X_data[i][k];
    }
    for (int b_ind=0; b_ind < b_data.size(1); b_ind++){
        acc=0.0;
        for (int p=0;p<y_n;p++){
            for (int k=0;k<nd;k++){
                y_j[k] = Y_data[p][k];
            };
            acc+= (*op)(x_i,y_j,ls)*b_data[p][b_ind];
        };
        output[i][b_ind]=acc;
    }
    __syncthreads();

};

template <typename scalar_t>
__device__ scalar_t calculate_laplace(
        scalar_t l_p[],
        scalar_t & x_ij,
        int & feature_num
        ){
    scalar_t res=1.0;
    for (int i=0; i<laplace_nodes;i++){
        if (i!=feature_num){ //Calculate the Laplace feature if i!=m...
            res = res * (x_ij-l_p[i])/(l_p[i]-l_p[feature_num]);
        }
    }
    return res;
}

template <typename scalar_t>
__device__ scalar_t calculate_laplace_product(
        scalar_t l_p[],
        scalar_t x_i[],
        int combs[],
        scalar_t b){
    for (int i=0; i<nd;i++){
        b = b*calculate_laplace(l_p, x_i[i], combs[i]);
    }
    return b;
}

template <typename scalar_t>
__global__ void laplace_interpolation(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                                      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
                                      const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> lap_nodes,
                                      const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> combinations,
                                      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output
){

    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    unsigned int x_n = X_data.size(0);
    if (i>x_n-1){return;}
    scalar_t x_i[nd];
    scalar_t l_p[laplace_nodes];
    int comb_j[nd];
    for (int k=0;k<nd;k++){
        x_i[k] = X_data[i][k];
//        printf("x: %f\n",x_i[k]);
    }
    for (int k=0;k<laplace_nodes;k++){
        l_p[k] = lap_nodes[k];
//        printf("l_p: %f\n",l_p[k]);
    }
    unsigned int y_n = combinations.size(0);

    for (int b_ind=0; b_ind < b_data.size(1); b_ind++){
        for (int p=0;p<y_n;p++){
            for (int k=0;k<nd;k++){
                comb_j[k] = combinations[p][k];
//                printf("comb_j: %i\n",comb_j[k]);
            };
            atomicAdd(&output[b_ind][p],calculate_laplace_product(l_p, x_i, comb_j, b_data[i][b_ind]));
        };
    }
    __syncthreads();
};

template <typename scalar_t>
__global__ void laplace_interpolation_transpose(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                                      const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
                                      const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> lap_nodes,
                                      const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> combinations,
                                      torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output
){

    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    unsigned int x_n = X_data.size(0);
    if (i>x_n-1){return;}
    scalar_t acc;
    scalar_t x_i[nd];
    scalar_t l_p[laplace_nodes];
    int comb_j[nd];
    for (int k=0;k<nd;k++){
        x_i[k] = X_data[i][k];
//        printf("x: %f\n",x_i[k]);
    }
    for (int k=0;k<laplace_nodes;k++){
        l_p[k] = lap_nodes[k];
//        printf("l_p: %f\n",l_p[k]);
    }
    unsigned int y_n = combinations.size(0);

    for (int b_ind=0; b_ind < b_data.size(1); b_ind++){
        acc=0.0;
        for (int p=0;p<y_n;p++){
            for (int k=0;k<nd;k++){
                comb_j[k] = combinations[p][k];
//                printf("comb_j: %i\n",comb_j[k]);
            };
            acc+=calculate_laplace_product(l_p, x_i, comb_j,b_data[p][b_ind]);
        };
        output[i][b_ind] = acc;
    }
    __syncthreads();
};


__device__ int calculate_box_ind(int &current_thread_idx,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> counts,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_box_idx){
    int nr_of_counts = counts.size(0); //remember 0 included!
    for (int i=0;i<nr_of_counts-1;i++){
        if ( current_thread_idx>=counts[i] && current_thread_idx<counts[i+1]){
            return x_box_idx[i];
        }
    }

}
//so this is correct...
template <typename scalar_t>
__global__ void skip_conv_1d(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data,
                             torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                             scalar_t * ls,
                             rbf_pointer<scalar_t> op,
                             const torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> boolean_interaction_mask,//Actually make a boolean mask
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_boxes_count,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> y_boxes_count,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_box_idx
                             ){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    unsigned int x_n = X_data.size(0);
    unsigned int M = boolean_interaction_mask.size(1);
    if (i>x_n-1){return;}
    scalar_t x_i[nd];
    scalar_t y_j[nd];
    scalar_t acc;
    int box_ind,start,end;
    for (int k=0;k<nd;k++){
        x_i[k] = X_data[i][k];
    }
    box_ind = calculate_box_ind(i,x_boxes_count,x_box_idx);
//    printf("thread %i: %i\n",i,box_ind);
    for (int b_ind=0; b_ind < b_data.size(1); b_ind++) { //for all dims of b
        acc=0.0;
        for (int j = 0; j < M; j++) { // iterate through every existing ybox
            if (boolean_interaction_mask[box_ind][j]) { //if there is an interaction
                start = y_boxes_count[j]; // 0 to something
                end = y_boxes_count[j + 1]; // seomthing
                for (int j_2 = start; j_2 < end; j_2++) {
                    for (int k = 0; k < nd; k++) {
                        y_j[k] = Y_data[j_2][k];
                    };
                    acc += (*op)(x_i, y_j, ls) * b_data[j_2][b_ind];

                }
            }
        }
        output[i][b_ind] = acc;
    }
    __syncthreads();
}

//Crux is getting the launch right presumably or parallelizaiton/stream. using a minblock or 32*n.
template <typename scalar_t>
__global__ void skip_conv_1d_shared(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data,
                             torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                             scalar_t * ls,
                             rbf_pointer<scalar_t> op,
                             const torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> boolean_interaction_mask,//Actually make a boolean mask
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_boxes_count,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> y_boxes_count,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_box_idx,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indicator
){
    int box_ind,start,end,a,b;
    box_ind = blockIdx.x;
    a = x_boxes_count[box_ind];
    b = x_boxes_count[box_ind+1];
    int i = a+ threadIdx.x; // current thread

    unsigned int M = boolean_interaction_mask.size(1);
    scalar_t x_i[nd];
    scalar_t y_j[nd];
    scalar_t acc;
    extern __shared__ scalar_t buffer[];
    scalar_t *yj = &buffer[0];
    scalar_t *bj = &buffer[blockDim.x*nd];


    //Load these points only... the rest gets no points... threadIdx.x +a to b. ...
    if (i<b) {
        for (int k = 0; k < nd; k++) {
            x_i[k] = X_data[i][k];
        }
    }
//    box_ind = calculate_box_ind(i,x_boxes_count,x_box_idx);
//    printf("thread %i: %i\n",i,box_ind);
    for (int b_ind=0; b_ind < b_data.size(1); b_ind++) { //for all dims of b
        acc=0.0;
        for (int m = 0; m < M; m++) { // iterate through every existing ybox
            if (boolean_interaction_mask[box_ind][m]) { //if there is an interaction
                start = y_boxes_count[m]; // 0 to something
                end = y_boxes_count[m + 1]; // seomthing

                for (int jstart = start, tile = 0; jstart < end; jstart += blockDim.x, tile++) {
                    int j = start+tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
                    if (j < end) { // we load yj from device global memory only if j<ny
                        torch_load_y<scalar_t>(j, yj, Y_data);
                        torch_load_b<scalar_t>(b_ind ,j, bj, b_data);
                    }
                    __syncthreads();
                    if (i < b) { // we compute x1i only if needed
                        scalar_t *yjrel = yj; // Loop on the columns of the current block.
                        for (int jrel = 0; (jrel < blockDim.x) && (jrel < end - jstart); jrel++, yjrel += nd) {
                            acc += (*op)(x_i, yjrel,ls) * bj[jrel]; //sums incorrectly cause pointer is fucked not sure if allocating properly
                        }
                    }
//                    __syncthreads(); //Lesson learned! Thread synching really important for cuda programming and memory loading when indices are dependent on threadIdx.x!
                };
                if (i < b) {
                    output[i][b_ind] = acc;
                }
            }
        }
    }
    __syncthreads();
}
