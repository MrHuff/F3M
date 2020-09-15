

#pragma once
#include <iostream>
#include <ostream>
#include <cublas_v2.h>
#include <random>
#include <stdexcept>
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
#define laplace_nodes 4 //Might wanna be able to fix this... (4,n=10000,156ms,1e-5),(4,n=10000,303ms,1e-6),(3,n=10000,60ms,1e-1)



//template<typename T>
//using rbf_pointer = T (*) (T[], T[],const T *);

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
std::tuple<dim3,dim3,int,torch::Tensor,torch::Tensor> skip_kernel_launch(int cols,
        int & blksize,
        torch::Tensor & box_sizes,
        torch::Tensor & box_idx){
    dim3 blockSize;
    dim3 gridSize;
    blockSize.x = blksize;

    torch::Tensor boxes_needed = torch::ceil(box_sizes.toType(torch::kFloat32)/(float)blksize).toType(torch::kLong); //blkSize is wrong and fix stuff
    torch::Tensor output_block = box_idx.repeat_interleave({boxes_needed});// Adjust for special case
    std::vector<torch::Tensor> cont = {};
    boxes_needed = boxes_needed.to("cpu").toType(torch::kInt32);
    auto accessor = boxes_needed.accessor<int,1>();
    for (int i=0;i<boxes_needed.size(0);i++){
        int b = accessor[i];
        cont.push_back(torch::arange(b));
    }
    torch::Tensor block_idx_within_box = torch::cat(cont).toType(torch::kInt32).to(output_block.device());
    //from_blob not fucking working...
    gridSize.x = output_block.size(0);
    return std::make_tuple(blockSize,gridSize,(blockSize.x * (cols+1)+cols) * sizeof(T),output_block,block_idx_within_box);
};

template<typename T>
__device__ T square(T x){
    return x*x;
};

template<typename T>
__device__ T cube(T x){
    return x*x*x;
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
__device__ inline static T rbf(T x[],T y[],const T *ls){
    T tmp=0;
    for (int k=0;k<nd;k++){
        tmp += square<T>(x[k]-y[k]);
    };
    return expf(-tmp/square<T>(*ls));
};
template<typename T>
__device__ inline static T rbf_grad(T x[],T y[],const T *ls){
    T tmp=0;
    for (int k=0;k<nd;k++){
        tmp += square<T>(x[k]-y[k]);
    };
    return expf(-tmp/square<T>(*ls))*tmp/cube<T>(*ls);
};
//template<typename T>
//__device__ rbf_pointer<T> rbf_pointer_func = rbf<T>;
//template<typename T>
//__device__ rbf_pointer<T> rbf_pointer_grad = rbf_grad<T>;
//

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
__device__ inline static void torch_load_y(int index, scalar_t *shared_mem, torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> y){
#pragma unroll
    for (int k = 0; k < nd; k++) {
        //assert(&((*px)[i * FIRST + k]) != nullptr);
        shared_mem[nd * threadIdx.x + k] = y[index][k]; // First, load the i-th line of px[0]  -> shared_mem[ 0 : FIRST ].
        // Don't use thread id -> nvidia-chips doesn't work like that! It only got a third of the way
        // Some weird memory allocation issue
    }
}
template <typename scalar_t>
__device__ inline static void torch_load_b(
        int col_index,
        int index,
        scalar_t *shared_mem,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b){
    shared_mem[threadIdx.x] = b[index][col_index];
}
template <typename scalar_t>
__device__ inline static void torch_load_y_v2(int index, scalar_t *shared_mem,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> y,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> y_idx_reordering
        ){
#pragma unroll
    for (int k = 0; k < nd; k++) {
        //assert(&((*px)[i * FIRST + k]) != nullptr);
        shared_mem[nd * threadIdx.x + k] = y[y_idx_reordering[index]][k]; // First, load the i-th line of px[0]  -> shared_mem[ 0 : FIRST ].
        // Don't use thread id -> nvidia-chips doesn't work like that! It only got a third of the way
        // Some weird memory allocation issue
    }
}
template <typename scalar_t>
__device__ inline static void torch_load_b_v2(
        int col_index,
        int index,
        scalar_t *shared_mem,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> y_idx_reordering){
    shared_mem[threadIdx.x] = b[y_idx_reordering[index]][col_index];
}


//Consider caching the kernel value if b is in Nxd.
template <typename scalar_t>
__global__ void rbf_1d_reduce_shared_torch(
                                const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                               const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                               const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
                               torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                                scalar_t * ls){
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
                    acc += rbf(x_i, yjrel,ls) * bj[jrel]; //sums incorrectly cause pointer is fucked not sure if allocating properly
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
                                      scalar_t * ls){

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
            acc+= rbf(x_i,y_j,ls)*b_data[p][b_ind];
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

template <typename scalar_t>
__device__ void xy_l1_dist(
        scalar_t * c_X,
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> c_Y,
        scalar_t * dist){
#pragma unroll
    for (int k = 0; k < nd; k++) {
        dist[k] = c_Y[k]-c_X[k];
    }
}

template <typename scalar_t>
__device__ scalar_t get_2_norm(scalar_t * dist){
    scalar_t acc=0;
#pragma unroll
    for (int k = 0; k < nd; k++) {
        acc+= square<scalar_t>(dist[k]);
    }
    return sqrt(acc);
}

template <typename scalar_t>
__device__ bool far_field_bool(scalar_t & l2_dist,scalar_t * edge){
    return l2_dist>=(*edge*2+1e-6);
}

template <typename scalar_t>
__device__ bool far_field_comp(scalar_t * c_X,scalar_t * c_Y,scalar_t * edge){
    scalar_t dist[nd];
    xy_l1_dist<scalar_t>(c_X,c_Y,dist);
    scalar_t l2 = get_2_norm(dist);
    return far_field_bool(l2,edge);
}





template <typename scalar_t>
__global__ void skip_conv_1d(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data,
                             torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                             scalar_t * ls,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers_X,//Actually make a boolean mask
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers_Y,//Actually make a boolean mask
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_boxes_count,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> y_boxes_count,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_box_idx,
                             const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> interactions_x_parsed,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> interactions_y
){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    unsigned int x_n = X_data.size(0);
    unsigned int M = centers_X.size(0);
    if (i>x_n-1){return;}
    scalar_t x_i[nd];
    scalar_t y_j[nd];
    scalar_t acc;
    int box_ind,start,end;
    box_ind = calculate_box_ind(i,x_boxes_count,x_box_idx);
    for (int k=0;k<nd;k++){
        x_i[k] = X_data[i][k];
    }
//    printf("thread %i: %i\n",i,box_ind);
    for (int b_ind=0; b_ind < b_data.size(1); b_ind++) { //for all dims of b
        acc=0.0;
        for (int j = interactions_x_parsed[box_ind][0]; j < interactions_x_parsed[box_ind][1]; j++) { // iterate through every existing ybox
            start = y_boxes_count[interactions_y[j]]; // 0 to something
            end = y_boxes_count[interactions_y[j] + 1]; // seomthing
            for (int j_2 = start; j_2 < end; j_2++) {
                for (int k = 0; k < nd; k++) {
                    y_j[k] = Y_data[j_2][k];
                };
                acc += rbf(x_i, y_j, ls) * b_data[j_2][b_ind];

            }
        }
        output[i][b_ind] = acc;
    }
}

//Crux is getting the launch right presumably or parallelizaiton/stream. using a minblock or 32*n.
//Refactor, this is calculated last with all near_fields available. Use same logic...

template <typename scalar_t>
__global__ void skip_conv_1d_shared(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data,
                             torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                             scalar_t * ls,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers_X,//Actually make a boolean mask
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers_Y,//Actually make a boolean mask
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_boxes_count,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> y_boxes_count,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indicator,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> box_block_indicator,
                            const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_idx_reordering,
                            const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> y_idx_reordering,
                            const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> interactions_x_parsed,
                            const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> interactions_y

){
    int box_ind,start,end,a,b;
    box_ind = indicator[blockIdx.x];
    a = x_boxes_count[box_ind];
    b = x_boxes_count[box_ind+1];
    int i = a + threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x; // Use within box, block index i.e. same size as indicator...

    unsigned int M = centers_X.size(0);
    scalar_t x_i[nd];
    scalar_t acc;
    extern __shared__ scalar_t buffer[];
    scalar_t *yj = &buffer[0];
    scalar_t *bj = &buffer[blockDim.x*nd];
    scalar_t *cX_i = &buffer[(blockDim.x)*(nd+1)];
    if (threadIdx.x<nd){
        cX_i[threadIdx.x] = centers_X[box_ind][threadIdx.x];
    }
    //Load these points only... the rest gets no points... threadIdx.x +a to b. ...
    if (i<b) {
        for (int k = 0; k < nd; k++) {
            x_i[k] = X_data[x_idx_reordering[i]][k];
        }
    }
//    box_ind = calculate_box_ind(i,x_boxes_count,x_box_idx);
//    printf("thread %i: %i\n",i,box_ind);
    for (int b_ind=0; b_ind < b_data.size(1); b_ind++) { //for all dims of b
        acc=0.0;

        for (int m = interactions_x_parsed[box_ind][0]; m < interactions_x_parsed[box_ind][1]; m++) {
            //Pass near field interactions...
            start = y_boxes_count[interactions_y[m]]; // 0 to something
            end = y_boxes_count[interactions_y[m] + 1]; // seomthing
            for (int jstart = start, tile = 0; jstart < end; jstart += blockDim.x, tile++) {
                int j = start+tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
                if (j < end) { // we load yj from device global memory only if j<ny
                    torch_load_y_v2<scalar_t>(j, yj, Y_data,y_idx_reordering);
                    torch_load_b_v2<scalar_t>(b_ind ,j, bj, b_data,y_idx_reordering);
                }
                __syncthreads();
                if (i < b) { // we compute x1i only if needed
                    scalar_t *yjrel = yj; // Loop on the columns of the current block.
                    for (int jrel = 0; (jrel < blockDim.x) && (jrel < end - jstart); jrel++, yjrel += nd) {
                        acc += rbf(x_i, yjrel,ls) * bj[jrel]; //sums incorrectly cause pointer is fucked not sure if allocating properly
                    }
                }
                __syncthreads(); //Lesson learned! Thread synching really important for cuda programming and memory loading when indices are dependent on threadIdx.x!
            };

        }
        if (i < b) {
            output[x_idx_reordering[i]][b_ind] += acc;
        }
    }
    __syncthreads();

}

//Implement shuffle reduce.
template <typename scalar_t>
__global__ void laplace_shared(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> lap_nodes,
        const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> combinations,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indicator,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> box_block_indicator,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_boxes_count,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers,
        const scalar_t * edge,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> idx_reordering
){
    int box_ind = indicator[blockIdx.x];
    int a,b,cheb_data_size;
    cheb_data_size = combinations.size(0);
    scalar_t factor = 2./(*edge);
    a = x_boxes_count[box_ind];
    b = x_boxes_count[box_ind+1];
    int i = a+ threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x; // current thread
    scalar_t x_i[nd];
    scalar_t b_i;
    if (i<b) {
        for (int k = 0; k < nd; k++) {
            x_i[k] = factor*(X_data[idx_reordering[i]][k]-centers[box_ind][k]);
        }
    }
    extern __shared__ int int_buffer[];
    extern __shared__ scalar_t buffer[];
    int *yj = &int_buffer[0];
    scalar_t *l_p = &buffer[blockDim.x*nd];
    int b_size = output.size(1);
    if (threadIdx.x<laplace_nodes){
        l_p[threadIdx.x] = lap_nodes[threadIdx.x];
    }
    for (int b_ind=0; b_ind<b_size; b_ind++) {
        if (i<b) {
            b_i = b_data[idx_reordering[i]][b_ind];
        }
        for (int jstart = 0, tile = 0; jstart < cheb_data_size; jstart += blockDim.x, tile++) {
            int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
            if (j < cheb_data_size) { // we load yj from device global memory only if j<ny
                torch_load_y<int>(j, yj, combinations);
            }

            __syncthreads();
            if (i < b) { // we compute x1i only if needed
                int *yjrel = yj; // Loop on the columns of the current block.
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < cheb_data_size - jstart); jrel++, yjrel += nd) {
                    //Do shuffle if possible...
                    atomicAdd(&output[jstart+jrel+box_ind*cheb_data_size][b_ind],calculate_laplace_product(l_p, x_i, yjrel,b_i)); //for each p, sum accross x's...
                }
            }
            __syncthreads();

        };
    };
    __syncthreads();

}

template <typename scalar_t>
__global__ void laplace_shared_transpose(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> lap_nodes,
        const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> combinations,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indicator,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> box_block_indicator,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_boxes_count,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers,
        const scalar_t * edge,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> idx_reordering
        ) {

    int box_ind, a, b,cheb_data_size;
    cheb_data_size = combinations.size(0);
    scalar_t factor = 2./(*edge);
    box_ind = indicator[blockIdx.x];
    a = x_boxes_count[box_ind];
    b = x_boxes_count[box_ind + 1];
    int i = a + threadIdx.x + box_block_indicator[blockIdx.x] * blockDim.x; // current thread
    scalar_t x_i[nd];
    if (i<b) {
        for (int k = 0; k < nd; k++) {
            x_i[k] = factor*(X_data[idx_reordering[i]][k]-centers[box_ind][k]);
        }
    }
    scalar_t acc;
    extern __shared__ int int_buffer[];
    extern __shared__ scalar_t buffer[];
    int *yj = &int_buffer[0];
    scalar_t *bj = &buffer[blockDim.x * nd];
    scalar_t *l_p = &buffer[blockDim.x * (nd + 1)];
    if (threadIdx.x < laplace_nodes) {
        l_p[threadIdx.x] = lap_nodes[threadIdx.x];
    }
    for (int b_ind = 0; b_ind < b_data.size(1); b_ind++) { //for all dims of b
        acc = 0.0;
        for (int jstart = 0, tile = 0; jstart < cheb_data_size; jstart += blockDim.x, tile++) {
            int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
            if (j < cheb_data_size) { // we load yj from device global memory only if j<ny
                torch_load_y<int>(j, yj, combinations);
                torch_load_b<scalar_t>(b_ind, j + box_ind * cheb_data_size, bj, b_data); //b's are incorrectly loaded
            }
            __syncthreads();
            if (i < b) { // we compute x1i only if needed
                int *yjrel = yj; // Loop on the columns of the current block.
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < cheb_data_size - jstart); jrel++, yjrel += nd) {
                    acc += calculate_laplace_product(l_p, x_i, yjrel, bj[jrel]);
                }
            }
            __syncthreads();

        }
        if (i < b) {
            output[idx_reordering[i]][b_ind] += acc;
        }

    }
    __syncthreads();
}

//Both these needs updating, i.e. pass additional appendage vector and pass interaction vector.

//[[0,1],[0,2],[0,3],[0,4],[0,5]...] ~ O(n_b^2x2)

//Thrust
template <typename scalar_t>
__global__ void skip_conv_far_boxes_opt(//the slow poke, optimize this to be faster...
                                    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> cheb_data,
                                    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data,
                                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                                    scalar_t * ls,
                                    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers_X,
                                    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers_Y,
                                    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indicator,
                                    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> box_block_indicator,
                                    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> interactions_x_parsed,
                                    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> interactions_y

){
    int box_ind,a,b,cheb_data_size;
    cheb_data_size = cheb_data.size(0);
    box_ind = indicator[blockIdx.x];
    a = box_ind* cheb_data_size;
    b = (box_ind+1)* cheb_data_size;
    int i = a + threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x; // Use within box, block index i.e. same size as indicator...
    scalar_t x_i[nd];
//    scalar_t y_j[nd];
    scalar_t acc;
    extern __shared__ scalar_t buffer[];
    scalar_t *yj = &buffer[0];
    scalar_t *bj = &buffer[cheb_data_size*nd];
//    scalar_t *distance = &buffer[cheb_data_size*(nd+1)];
    scalar_t *cX_i = &buffer[cheb_data_size*(nd+1)]; //Get another shared memory pointer.

    if (threadIdx.x<nd){
        cX_i[threadIdx.x] = centers_X[box_ind][threadIdx.x];
    }
    //Load these points only... the rest gets no points... threadIdx.x +a to b. ...
    if (i<b) {
        for (int k = 0; k < nd; k++) {
            x_i[k] = cheb_data[threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x][k];
        }
    }

//    box_ind = calculate_box_ind(i,x_boxes_count,x_box_idx);
//    printf("thread %i: %i\n",i,box_ind);
    if (interactions_x_parsed[box_ind][0]>-1) {
        for (int b_ind = 0;b_ind < b_data.size(1); b_ind++) { //for all dims of b A*b, b \in \mathbb{R}^{n\times d}, d>=1.
            acc = 0.0;
            for (int m = interactions_x_parsed[box_ind][0]; m < interactions_x_parsed[box_ind][1]; m++) {
                for (int jstart = 0, tile = 0; jstart < cheb_data_size; jstart += blockDim.x, tile++) {
                    int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
                    if (j < cheb_data_size) {
                        for (int k = 0; k < nd; k++) {
                            yj[j*nd+k] = cheb_data[j][k]+centers_Y[interactions_y[m]][k] - cX_i[k];
                        }
                        bj[j] = b_data[j + interactions_y[m] * cheb_data_size][b_ind];
                    }
                    __syncthreads(); //Need to be smart with this, don't slow down the others!
                    if (i < b) { // we compute x1i only if needed
                        scalar_t *yjrel = yj; // Loop on the columns of the current block.
                        for (int j = 0; (j < blockDim.x) && (j < cheb_data_size - jstart); j++, yjrel += nd) {
                            acc += rbf(x_i, yjrel, ls)* bj[j]; //sums incorrectly cause pointer is fucked not sure if allocating properly
                        }
                    }
                }
                __syncthreads(); //Lesson learned! Thread synching really important for cuda programming and memory loading when indices are dependent on threadIdx.x!

            }
            if (i < b) { // we compute x1i only if needed
                output[i][b_ind] += acc;
            }
            __syncthreads();

        }
//        __syncthreads();
    }
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

template <typename scalar_t>
__global__ void get_cheb_idx_data(
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> cheb_nodes,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> cheb_data,
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> cheb_idx
){
    int d = cheb_data.size(1);
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
    for (int j=0;j<d;j++){
        idx = (int) floor((i%(int)pow(lap_nodes,j+1))/pow(lap_nodes,j));
        cheb_idx[i][j] = idx;
        cheb_data[i][j] = cheb_nodes[idx];
    }
}

__global__ void get_centers(
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> centers
){
    int d = centers.size(1);
    int n = centers.size(0);
    int i = threadIdx.x+blockIdx.x*blockDim.x; // Thread nr
    if (i>n-1){return;}
    int idx;
    __syncthreads();
    for (int j=0;j<d;j++){
        idx = (int) floor((i%(int)pow(2,j+1))/pow(2,j));
        centers[i][j] = (idx==0) ? -1 : 1;
    }
}



template <typename scalar_t>
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
    if (get_2_norm(distance)>=(*edge*2+1e-6)){
        is_far_field[i]=true;
    }


}
