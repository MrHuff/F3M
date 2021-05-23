
#pragma once
#include "linearprobing.cu"
#include <iostream>
#include <ostream>
#include <cublas_v2.h>
#include <random>
#include <stdexcept>
#include <npp.h>
#include <torch/torch.h> //for n00bs like me, direct translation to python rofl
#define BLOCK_SIZE 192
#define MAXTHREADSPERBLOCK 1024
#define SHAREDMEMPERBLOCK 49152
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif
//template<typename T, int nd>

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
                                                                         torch::Tensor& box_idx){
    dim3 blockSize;
    dim3 gridSize;
    blockSize.x = blksize;

    torch::Tensor boxes_needed = torch::ceil(box_sizes.toType(torch::kFloat32)/(float)blksize).toType(torch::kLong); //blkSize is wrong and fix stuff
    torch::Tensor output_block = box_idx.repeat_interleave(boxes_needed).toType(torch::kInt32);// Adjust for special case
    std::vector<torch::Tensor> cont = {};
    boxes_needed = boxes_needed.to("cpu").toType(torch::kInt32);
    auto accessor = boxes_needed.accessor<int,1>();
    for (int i=0;i<boxes_needed.size(0);i++){
        int b = accessor[i];
        cont.push_back(torch::arange(b));
    }
    torch::Tensor block_idx_within_box = torch::cat(cont).toType(torch::kInt32).to(output_block.device());//fix this...
    gridSize.x = output_block.size(0);
    return std::make_tuple(blockSize,gridSize,(blockSize.x * (cols+1)+cols) * sizeof(T),output_block,block_idx_within_box);
};

template<typename T>
__device__ inline T square(T x){
    return x*x;
};



template<typename T, int nd>
__device__ inline static T square_dist(T x[],T y[]){
    T dist=(T)0;
    for (int k=0;k<nd;k++){
        dist += square<T>(x[k]-y[k]);
    };
    return dist;
};

template<typename T, int nd>
__device__ inline static T rbf(T x[],T y[],const T *ls){
    T dist=square_dist<T,nd>(x,y);
    return expf(-dist*(*ls));
};


//template<typename T, int nd>
//__device__ rbf_pointer<T> rbf_pointer_func = rbf<T>;
//template<typename T, int nd>
//__device__ rbf_pointer<T> rbf_pointer_grad = rbf_grad<T>;
//

template <typename scalar_t,int nd>
__device__ inline static void torch_load_y(int index, scalar_t *shared_mem, torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> y){
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
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> b){
    shared_mem[threadIdx.x] = b[index][col_index];
}
template <typename scalar_t,int nd>
__device__ inline static void torch_load_y_v2(int reorder_index, scalar_t *shared_mem,
                                              const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> y
){
#pragma unroll
    for (int k = 0; k < nd; k++) {
        //assert(&((*px)[i * FIRST + k]) != nullptr);
        shared_mem[nd * threadIdx.x + k] = y[reorder_index][k]; // First, load the i-th line of px[0]  -> shared_mem[ 0 : FIRST ].
        // Don't use thread id -> nvidia-chips doesn't work like that! It only got a third of the way
        // Some weird memory allocation issue
    }
}
template <typename scalar_t,int nd>
__device__ inline static void torch_load_b_v2(
        int col_index,
        int reorder_index,
        scalar_t *shared_mem,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> b){
    shared_mem[threadIdx.x] = b[reorder_index][col_index];
}


//Consider caching the kernel value if b is in Nxd.
template <typename scalar_t,int nd>
__global__ void rbf_1d_reduce_shared_torch(
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> Y_data,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
        torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> output,
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
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
    scalar_t *buffer = reinterpret_cast<scalar_t *>(my_smem);
    scalar_t *yj = &buffer[0];
    scalar_t *bj = &buffer[blockDim.x*nd];
    scalar_t acc;
    for (int b_ind=0; b_ind<output.size(1); b_ind++) {
        acc=0;
        for (int jstart = 0, tile = 0; jstart < y_n; jstart += blockDim.x, tile++) {
            int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
            if (j < y_n) { // we load yj from device global memory only if j<ny
                torch_load_y<scalar_t,nd>(j, yj, Y_data);
                torch_load_b<scalar_t>(b_ind ,j, bj, b_data);
            }
            __syncthreads();
            if (i < x_n) { // we compute x1i only if needed
                scalar_t *yjrel = yj; // Loop on the columns of the current block.
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < y_n - jstart); jrel++, yjrel += nd) {
                    acc += rbf<scalar_t,nd>(x_i, yjrel,ls) * bj[jrel]; //sums incorrectly cause pointer is fucked not sure if allocating properly
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

template <typename scalar_t,int nd>
__global__ void rbf_1d_reduce_simple_torch(const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
                                           const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                                           const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
                                           torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> output,
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
        acc=0;
        for (int p=0;p<y_n;p++){
            for (int k=0;k<nd;k++){
                y_j[k] = Y_data[p][k];
            };
            acc+= rbf<scalar_t,nd>(x_i,y_j,ls)*b_data[p][b_ind];
        };
        output[i][b_ind]=acc;
    }
    __syncthreads();

};

template <typename scalar_t,int nd>
__device__ __forceinline__ scalar_t calculate_barycentric_lagrange(//not sure this is such a great idea, when nan how avoid...
        scalar_t *l_p,
        scalar_t *W,
        scalar_t *x_i,
        scalar_t *l_x,
        const bool * pBoolean,
        const int *combs,
        scalar_t b
){
    for (int i = 0; i < nd; i++) {
        if (pBoolean[i]){
            if (x_i[i]!=l_p[combs[i]]){
                b=(scalar_t) 0;
                break;
            }
        }else{
            b*=l_x[i]*W[combs[i]]/(x_i[i]-l_p[combs[i]]);
        }
    }
    return b;
}



template <typename scalar_t,int nd>
__device__ __forceinline__ scalar_t get_2_norm(scalar_t * dist){
    scalar_t acc=0;
#pragma unroll
    for (int k = 0; k < nd; k++) {
        acc+= square<scalar_t>(dist[k]);
    }
    return sqrt(acc);
}

//Crux is getting the launch right presumably or parallelizaiton/stream. using a minblock or 32*n.
//Refactor, this is calculated last with all near_fields available. Use same logic...

template <typename scalar_t,int nd>
__global__ void skip_conv_1d_shared(const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
                                    const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                                    const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> b_data,
                                    torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> output,
                                    scalar_t * ls,
                                    const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> x_boxes_count,
                                    const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> y_boxes_count,
                                    const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> block_box_indicator,
                                    const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> box_block_indicator,
                                    const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> x_idx_reordering,
                                    const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> y_idx_reordering,
                                    const torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> interactions_x_parsed,
                                    const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> interactions_y

){
    int i,box_ind,start,end,a,b,int_m,x_idx_reorder,b_size,interactions_a,interactions_b;
    box_ind = block_box_indicator[blockIdx.x];
    a = x_boxes_count[box_ind];
    b = x_boxes_count[box_ind+1];
    i = a + threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x; // Use within box, block index i.e. same size as indicator...
    scalar_t x_i[nd];
    scalar_t acc;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
    scalar_t *buffer = reinterpret_cast<scalar_t *>(my_smem);
    scalar_t *yj = &buffer[0];
    scalar_t *bj = &buffer[blockDim.x*nd];
    //Load these points only... the rest gets no points... threadIdx.x +a to b. ...
    b_size = b_data.size(1);
    if (i<b) {
        x_idx_reorder = x_idx_reordering[i];
        for (int k = 0; k < nd; k++) {
            x_i[k] = X_data[x_idx_reorder][k];
        }
    }

    interactions_a = interactions_x_parsed[box_ind][0];
    interactions_b= interactions_x_parsed[box_ind][1];
    if (interactions_a>-1) {
        for (int b_ind = 0; b_ind < b_size; b_ind++) { //for all dims of b
            acc = 0;

            for (int m = interactions_a; m < interactions_b; m++) {
                //Pass near field interactions...
                int_m = interactions_y[m];
                start = y_boxes_count[int_m]; // 0 to something
                end = y_boxes_count[int_m + 1]; // seomthing
                for (int jstart = start, tile = 0; jstart < end; jstart += blockDim.x, tile++) {
                    int j = start + tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
                    if (j < end) { // we load yj from device global memory only if j<ny
                        torch_load_y_v2<scalar_t, nd>(y_idx_reordering[j], yj, Y_data);
                        torch_load_b_v2<scalar_t, nd>(b_ind, y_idx_reordering[j], bj, b_data);
                    }
                    __syncthreads();
                    if (i < b) { // we compute x1i only if needed
                        scalar_t *yjrel = yj; // Loop on the columns of the current block.
                        for (int jrel = 0; (jrel < blockDim.x) && (jrel < end - jstart); jrel++, yjrel += nd) {
                            acc += rbf<scalar_t, nd>(x_i, yjrel, ls) *
                                   bj[jrel]; //sums incorrectly cause pointer is fucked not sure if allocating properly
                        }
                    }
                    __syncthreads(); //Lesson learned! Thread synching really important for cuda programming and memory loading when indices are dependent on threadIdx.x!
                };

            }
            if (i < b) {
                output[x_idx_reorder][b_ind] += acc;
            }
        }
        __syncthreads();
    }
}
template <typename scalar_t,int nd>
__device__ __forceinline__ void lagrange_data_load(scalar_t w[],
                                   scalar_t x_i[],
                                   scalar_t l_x[],
                                   int node_cum_shared[],
                                   scalar_t l_p[],
                                   bool pBoolean[],
                                   scalar_t & factor,
                                   int & idx_reorder,
                                   int & box_ind,
                                   const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers,
                                   const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data
){
#pragma unroll
    for (int k = 0; k < nd; k++) {
        x_i[k] = factor*(X_data[idx_reorder][k]-centers[box_ind][k]);
        scalar_t tmp = 0.;
        pBoolean[k]=false;
        for (int l=node_cum_shared[k];l<node_cum_shared[k+1];l++){ //node_cum_index is wrong or l_p is loaded completely incorrectly!
            if (x_i[k]==l_p[l]){
                pBoolean[k]=true;
                tmp=1;
                break;
            }else{
                tmp += w[l]/(x_i[k]-l_p[l]);
            }
        }
        l_x[k] = 1/tmp;
    }

}

//Implement shuffle reduce.
template <typename scalar_t,int nd>
__global__ void lagrange_shared(
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> lap_nodes,
        const torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> combinations,
        torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> output,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> indicator,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> box_block_indicator,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> x_boxes_count, //error is here! I think it does an access here when its not supposed to...
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers,
        const scalar_t * edge,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> idx_reordering,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> node_list_cum,
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> W

){
    int i,a,b,cheb_data_size,idx_reorder,laplace_n,b_size,box_ind;
    box_ind = indicator[blockIdx.x];
    b_size = output.size(1);
    cheb_data_size = combinations.size(0);
    scalar_t factor = 2/(*edge); //rescaling data to langrange node interval (-1,1)
    laplace_n = lap_nodes.size(0);
    a = x_boxes_count[box_ind]; //Error occurs here
    b = x_boxes_count[box_ind+1];
    i = a+ threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x; // current thread
    scalar_t x_i[nd];
    scalar_t l_x[nd];
    bool pBoolean[nd];
    scalar_t b_i;
    extern __shared__ int int_buffer[];
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
    scalar_t *buffer = reinterpret_cast<scalar_t *>(my_smem);
    int *node_cum_shared = &int_buffer[0];
    int *yj = &int_buffer[1+nd];
    scalar_t *l_p = &buffer[blockDim.x*nd+1+nd];
    scalar_t *w_shared = &buffer[blockDim.x*nd+1+nd+laplace_n];

    if (threadIdx.x<nd+1){
        if (threadIdx.x==0){
            node_cum_shared[threadIdx.x]=0;
        }else{
            node_cum_shared[threadIdx.x] = node_list_cum[threadIdx.x-1];
        }
    }
    __syncthreads();

    for (int jstart = 0, tile = 0; jstart < laplace_n; jstart += blockDim.x, tile++) {
        int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
        if (j < laplace_n) { // we load yj from device global memory only if j<ny
            l_p[j] = lap_nodes[j];
            w_shared[j] = W[j];
        }
        __syncthreads();
    }

    __syncthreads();
    if (i<b) {
        idx_reorder = idx_reordering[i];
        lagrange_data_load<scalar_t,nd>(
                w_shared,
                x_i,
                l_x,
                node_cum_shared,
                l_p,
                pBoolean,
                factor,
                idx_reorder,
                box_ind,
                centers,
                X_data
        );
    }
    __syncthreads();
    for (int b_ind=0; b_ind<b_size; b_ind++) {
        if (i<b) {
            b_i = b_data[idx_reorder][b_ind];
        }
        for (int jstart = 0, tile = 0; jstart < cheb_data_size; jstart += blockDim.x, tile++) {
            int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
            if (j < cheb_data_size) { // we load yj from device global memory only if j<ny
                torch_load_y<int,nd>(j, yj, combinations);
            }
            __syncthreads();
            if (i < b) { // we compute x1i only if needed
                int *yjrel = yj; // Loop on the columns of the current block.
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < cheb_data_size - jstart); jrel++, yjrel += nd) {
//                    atomicAdd(&output[jstart+jrel+box_ind*cheb_data_size][b_ind],calculate_lagrange_product<scalar_t, nd>(l_p, x_i, yjrel, b_i, node_cum_shared)); //for each p, sum accross x's...
                    atomicAdd(&output[jstart+jrel+box_ind*cheb_data_size][b_ind],calculate_barycentric_lagrange<scalar_t,nd>(l_p,w_shared,x_i,l_x,pBoolean,yjrel,b_i)); //for each p, sum accross x's...
                }
            }
            __syncthreads();

        };
    };
    __syncthreads();

}

template <typename scalar_t,int nd>
__global__ void laplace_shared_transpose(
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> lap_nodes,
        const torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> combinations,
        torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> output,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> indicator,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> box_block_indicator,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> x_boxes_count,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers,
        const scalar_t * edge,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> idx_reordering,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> node_list_cum,
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> W
) {

    int laplace_n,i, box_ind, a, b,cheb_data_size,idx_reorder,b_size;
    cheb_data_size = combinations.size(0);
    scalar_t factor = 2/(*edge);
    box_ind = indicator[blockIdx.x];
    a = x_boxes_count[box_ind];
    b = x_boxes_count[box_ind + 1];
    i = a + threadIdx.x + box_block_indicator[blockIdx.x] * blockDim.x; // current thread
    laplace_n = lap_nodes.size(0);
    b_size =  b_data.size(1);
    scalar_t x_i[nd];
    scalar_t l_x[nd];
    bool pBoolean[nd];
    scalar_t acc;
    extern __shared__ int int_buffer[];
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
    scalar_t *buffer = reinterpret_cast<scalar_t *>(my_smem);
    int *node_cum_shared = &int_buffer[0];
    int *yj = &int_buffer[1+nd];
    scalar_t *bj = &buffer[(blockDim.x+1) * nd+1];
    scalar_t *l_p = &buffer[(blockDim.x+1) * nd+blockDim.x+1];
    scalar_t *w_shared = &buffer[(blockDim.x+1) * nd+blockDim.x+1+laplace_n];
    if (threadIdx.x<nd+1){
        if (threadIdx.x==0){
            node_cum_shared[0] = (int) 0;

        }else{
            node_cum_shared[threadIdx.x] = node_list_cum[threadIdx.x-1];
        }
    }
    __syncthreads();

    for (int jstart = 0, tile = 0; jstart < laplace_n; jstart += blockDim.x, tile++) {
        int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
        if (j < laplace_n) { // we load yj from device global memory only if j<ny
            l_p[j] = lap_nodes[j];
            w_shared[j] = W[j];
        }
        __syncthreads();
    }
    if (i<b) {
        idx_reorder = idx_reordering[i];
        lagrange_data_load<scalar_t,nd>(
                w_shared,
                x_i,
                l_x,
                node_cum_shared,
                l_p,
                pBoolean,
                factor,
                idx_reorder,
                box_ind,
                centers,
                X_data
        );
    }
    __syncthreads();

    for (int b_ind = 0; b_ind <b_size; b_ind++) { //for all dims of b
        acc = 0;
        for (int jstart = 0, tile = 0; jstart < cheb_data_size; jstart += blockDim.x, tile++) {
            int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
            if (j < cheb_data_size) { // we load yj from device global memory only if j<ny
                torch_load_y<int,nd>(j, yj, combinations);
                torch_load_b<scalar_t>(b_ind, j + box_ind * cheb_data_size, bj, b_data); //b's are incorrectly loaded
            }
            __syncthreads();
            if (i < b) { // we compute x1i only if needed
                int *yjrel = yj; // Loop on the columns of the current block.
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < cheb_data_size - jstart); jrel++, yjrel += nd) {
//                    acc += calculate_lagrange_product<scalar_t, nd>(l_p, x_i, yjrel, bj[jrel], node_cum_shared);
                    acc += calculate_barycentric_lagrange<scalar_t,nd>(l_p,w_shared,x_i,l_x,pBoolean,yjrel,bj[jrel]);
                }
            }
            __syncthreads();

        }
        if (i < b) {
            output[idx_reorder][b_ind] += acc;
        }

    }
    __syncthreads();
}

//[[0,1],[0,2],[0,3],[0,4],[0,5]...] ~ O(n_b^2x2)
//Move to shared mem experiment!
//Thrust
template <typename scalar_t,int nd>
__global__ void skip_conv_far_boxes_opt(//needs rethinking
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> cheb_data_X,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> cheb_data_Y,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> b_data,
        torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> output,
        scalar_t * ls,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers_X,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers_Y,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> indicator,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> box_block_indicator,
        const torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> interactions_x_parsed,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> interactions_y

){
    int box_ind,a,cheb_data_size,int_m,interactions_a,interactions_b,b_size;
    cheb_data_size = cheb_data_X.size(0);
    box_ind = indicator[blockIdx.x];
    a = box_ind* cheb_data_size;
    int i = a + threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x; // Use within box, block index i.e. same size as indicator...
    int i_calc = threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x;
    scalar_t x_i[nd];
//    scalar_t y_j[nd];
    scalar_t acc;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
    scalar_t *buffer = reinterpret_cast<scalar_t *>(my_smem);
    scalar_t *yj = &buffer[0];
    scalar_t *bj = &buffer[blockDim.x*nd];
//    scalar_t *distance = &buffer[cheb_data_size*(nd+1)];
    scalar_t *cX_i = &buffer[blockDim.x*(nd+1)]; //Get another shared memory pointer.

    if (threadIdx.x<nd){
        cX_i[threadIdx.x] = centers_X[box_ind][threadIdx.x];
    }
    //Load these points only... the rest gets no points... threadIdx.x +a to b. ...
    if (i_calc<cheb_data_size) {
        for (int k = 0; k < nd; k++) {
            x_i[k] = cheb_data_X[i_calc][k];
        }
    }

    interactions_a = interactions_x_parsed[box_ind][0];
    interactions_b= interactions_x_parsed[box_ind][1];
    b_size = b_data.size(1);
    if (interactions_a>-1) {
        for (int b_ind = 0;b_ind <b_size ; b_ind++) { //for all dims of b A*b, b \in \mathbb{R}^{n\times d}, d>=1.
            acc = 0;
            for (int m = interactions_a; m < interactions_b; m++) {
                int_m = interactions_y[m];
                for (int jstart = 0, tile = 0; jstart < cheb_data_size; jstart += blockDim.x, tile++) {
                    int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
                    if (j < cheb_data_size) {
                        for (int k = 0; k < nd; k++) {
                            yj[nd * threadIdx.x+k] = cheb_data_Y[j][k]+centers_Y[int_m][k] - cX_i[k]; //Error also occurs here!
                        }
                        bj[threadIdx.x] = b_data[j + int_m * cheb_data_size][b_ind];
                    }
                    __syncthreads(); //Need to be smart with this, don't slow down the others!
                    if (i_calc < cheb_data_size) { // we compute x1i only if needed
                        scalar_t *yjrel = yj; // Loop on the columns of the current block.
                        for (int p = 0; (p < blockDim.x) && (p < cheb_data_size - jstart); p++, yjrel += nd) {
                            acc += rbf<scalar_t,nd>(x_i, yjrel, ls)* bj[p]; //sums incorrectly cause pointer is fucked not sure if allocating properly
                        }
                    }
                    __syncthreads(); //Lesson learned! Thread synching really important for cuda programming and memory loading when indices are dependent on threadIdx.x!
                }
                __syncthreads(); //Lesson learned! Thread synching really important for cuda programming and memory loading when indices are dependent on threadIdx.x!

            }
            if (i_calc < cheb_data_size) { // we compute x1i only if needed
                output[i][b_ind] += acc;
            }
            __syncthreads();

        }
//        __syncthreads();
    }
}
/*****
 *
 * UTILS
 *
 *
 *
 */


//
// Created by rhu on 2020-07-05.
//

__global__ void parse_x_boxes(
        const torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> box_cumsum,
        torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> results
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
__global__ void get_smolyak_indices(
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> cheb_nodes,
        torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> cheb_data,
        torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> cheb_idx,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> size_per_dim,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> cum_prod
){
    int n = cheb_data.size(0);
    int i = threadIdx.x+blockIdx.x*blockDim.x; // Thread nr
    if (i>n-1){return;}
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
    scalar_t *buffer = reinterpret_cast<scalar_t *>(my_smem);
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
            idx = (int) floor((float)tmp/1);
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
        torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> centers
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
__global__ void boolean_separate_interactions_small(
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers_X,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers_Y,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> unique_og_X,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> unique_og_Y,
        const torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> interactions,
        const scalar_t * edge,
        torch::PackedTensorAccessor64<bool,1,torch::RestrictPtrTraits> is_far_field,
        torch::PackedTensorAccessor64<bool,1,torch::RestrictPtrTraits> is_small_field,
        torch::PackedTensorAccessor64<bool,1,torch::RestrictPtrTraits> keep_mask,
        const int * nr_interpolation_points,
        const int * small_field_limit,
        const bool * do_init_check
){
    int i = threadIdx.x+blockIdx.x*blockDim.x; // Thread nr
    int n = interactions.size(0);
    if (i>n-1){return;}
    int bx = interactions[i][0];
    int by = interactions[i][1];
    int interaction_size;
    scalar_t distance[nd];
    scalar_t cx[nd];
    scalar_t cy[nd];
    interaction_size = unique_og_X[bx]+unique_og_Y[by];
    if (*do_init_check){
        if (interaction_size<(2* *nr_interpolation_points)){
            is_small_field[i]=true;
            keep_mask[i]=false;
            return;
        }
    }
#pragma unroll
    for (int k=0;k<nd;k++){
        cx[k] = centers_X[bx][k];
        cy[k] = centers_Y[by][k];
        distance[k]=cy[k] - cx[k];
    }
    if (get_2_norm<scalar_t,nd>(distance)>=(*edge*2)){
        if(*do_init_check){
            is_far_field[i]=true;
        }
        keep_mask[i]=false;
        return;
    }
    if (interaction_size<(2* *small_field_limit)){
        is_small_field[i]=true;
        keep_mask[i]=false;
        return;
    }
}

template <typename scalar_t, int nd>
__global__ void boolean_separate_interactions_small_var_comp(
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers_X,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers_Y,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> unique_og_X,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> unique_og_Y,
        const torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> interactions,
        const scalar_t * edge,
        torch::PackedTensorAccessor64<bool,1,torch::RestrictPtrTraits> is_far_field,
        torch::PackedTensorAccessor64<bool,1,torch::RestrictPtrTraits> is_small_field,
        torch::PackedTensorAccessor64<bool,1,torch::RestrictPtrTraits> keep_mask,
        const int * nr_interpolation_points,
        const int * small_field_limit,
        const bool * do_init_check,
        const bool * enable_pair,
        const bool * enable_all,
        const scalar_t * crit_distance,
        const scalar_t * ls
){
    int i = threadIdx.x+blockIdx.x*blockDim.x; // Thread nr
    int n = interactions.size(0);
    if (i>n-1){return;}
    int bx = interactions[i][0];
    int by = interactions[i][1];
    int interaction_size;
    scalar_t distance[nd];
    scalar_t cx[nd];
    scalar_t cy[nd];
    interaction_size = unique_og_X[bx]+unique_og_Y[by];
    if (*do_init_check){
        if (interaction_size<(2* *nr_interpolation_points)){
            is_small_field[i]=true;
            keep_mask[i]=false;
            return;
        }
    }

#pragma unroll
    for (int k=0;k<nd;k++){
        cx[k] = centers_X[bx][k];
        cy[k] = centers_Y[by][k];
        distance[k]=cy[k] - cx[k];
    }
    scalar_t norm_dist = get_2_norm<scalar_t,nd>(distance);
    if (norm_dist>=(*edge*2)){
        if(*do_init_check){
            is_far_field[i]=true;
        }
        keep_mask[i]=false;
        return;
    }
    if (*enable_pair) {
        if ((norm_dist*norm_dist*(*ls)) <= 1e-2) {
            is_far_field[i]=true;
            keep_mask[i]=false;
            return;
        }
    }
    if (*enable_all) {
        if (norm_dist <= *crit_distance) {
            is_far_field[i]=true;
            keep_mask[i]=false;
            return;
        }
    }


    if (interaction_size<(2* *small_field_limit)){
        is_small_field[i]=true;
        keep_mask[i]=false;
        return;
    }


}


__global__ void get_keep_mask(
        const torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> interactions,
        torch::PackedTensorAccessor64<bool,1,torch::RestrictPtrTraits> keep_x_box,
        torch::PackedTensorAccessor64<bool,1,torch::RestrictPtrTraits> keep_y_box,
        torch::PackedTensorAccessor64<bool,1,torch::RestrictPtrTraits> output
){
    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    if (tid>interactions.size(0)-1){return;}
    output[tid] = keep_x_box[interactions[tid][0]]*keep_y_box[interactions[tid][1]];
}
__global__ void transpose_to_existing_only(
        torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> interactions,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> old_new_map_X,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> old_new_map_Y
){
    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    if (tid>interactions.size(0)-1){return;}
    int interaction_x = interactions[tid][0];
    int interaction_y = interactions[tid][1];
    interactions[tid][0] = old_new_map_X[interaction_x];
    interactions[tid][1] = old_new_map_Y[interaction_y];


}
__global__ void transpose_to_existing_only_tree(
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> all_boxes,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> removed_indices_x
){
    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    if (tid>all_boxes.size(0)-1){return;}
    unsigned int i_max = removed_indices_x.size(0);
    if(i_max==0) return;
    unsigned int i_min = 0;
    int box_idx = all_boxes[tid];
    if(box_idx <= removed_indices_x[i_min]) return;
    if(box_idx >= removed_indices_x[i_max-1]) {all_boxes[tid] -= i_max; return;}
    unsigned int i_try;
    while(true){
        i_try = (i_min+i_max)/2;
        if(box_idx < removed_indices_x[i_try]) {i_max = i_try;}
        else if(box_idx > removed_indices_x[i_try]) {i_min = i_try;}
        else return;
        if(i_min==i_max-1) break;
    }
    all_boxes[tid] -= i_max;
}


template<typename scalar_t,int cols>
__global__ void box_variance(
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> x_dat_reordering,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> x_box_cum,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> big_enough_boxes,
        torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> output_1
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    unsigned int x_n = big_enough_boxes.size(0);
    scalar_t x_i[cols];
    scalar_t x_i_square[cols];
    int box_ind;
    int limit=1000;
    int start,end,end_num;
    if (i>x_n-1) {
        return;
    }

    box_ind = big_enough_boxes[i];
    for (int c=0;c<cols;c++){
        x_i[c]=0.0;
        x_i_square[c]=0.0;
    }
    start = x_box_cum[box_ind];
    end = x_box_cum[box_ind+1];
    scalar_t acc = end-start;
    if (acc<limit){
        end_num=end;
    }else{
        end_num = start+limit;
        acc = (scalar_t)limit;
    }
    for (int j = start; j<end_num;j++) {
        for (int d = 0;d<cols;d++) {
            x_i[d]+=X_data[x_dat_reordering[j]][d];
            x_i_square[d]+=square(X_data[x_dat_reordering[j]][d]);
        }
    }
    for (int d = 0;d<cols;d++) {
        if (acc>1){
            output_1[i][d] = (x_i_square[d]/acc-square(x_i[d]/acc))*acc/(acc-1);
        }
    }
}


__global__ void repeat_within(
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> short_cumsum,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> repeat_interleaved,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> unique,
        torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> new_right,
        const int * p
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i<unique.size(0)){
        int start;
        int end;
        if (i==0){
            start=0;
            end = short_cumsum[0];
        }else{
            start = short_cumsum[i-1];
            end = short_cumsum[i];
        }
        int distance = end-start;
        int mover = start*(*p);
        for (int r=0;r<*p;r++){
            for(int j=start;j<end;j++){
                new_right[j-start+mover+r*distance][1]=repeat_interleaved[j];
            }
        }

    }

}


__global__ void repeat_add(
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> arr,
        torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> new_interactions,
        const int * p
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i<new_interactions.size(0)){
        int j = i%(*p); //modulo index
        int right = new_interactions[i][1];
        new_interactions[i][1] = (*p)*right+arr[j];
    }

}

template <typename scalar_t, int nd>
__global__ void box_division_cum_hash(
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> alpha,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> multiply,
        const scalar_t * int_mult,
        const scalar_t * edge,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> global_vector_counter_cum,
        KeyValue* perm,
        int * hash_size

){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>X_data.size(0)-1){return;}
    int idx=0;
    for (int p = 0;p<nd;p++) {
        idx += multiply[p]*(int)floor( *int_mult * (X_data[i][p] - alpha[p]) / *edge);
    }
    int perm_val = device_lookup(perm,idx,hash_size);
    atomicAdd(&global_vector_counter_cum[perm_val+1],1);

}

template <typename scalar_t, int nd>
__global__ void center_perm_hash(const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers_natural,
                                 const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> alpha,
                                 const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> multiply,
                                 const scalar_t * int_mult,
                                 const scalar_t * edge,
                                 KeyValue* perm,
                                 int * hash_size
){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>centers_natural.size(0)-1){return;}
    int idx=0;
    for (int p = 0;p<nd;p++) {
        idx += multiply[p]*(int)floor( *int_mult * (centers_natural[i][p] - alpha[p]) / *edge);
    }
    device_insert(perm,idx,i,hash_size);
}


template <typename scalar_t, int nd>
__global__ void box_division_assign_hash(
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> alpha,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> multiply,
        const scalar_t * int_mult,
        const scalar_t * edge,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> global_vector_counter_cum,
        KeyValue* perm,
        int * hash_size,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> global_unique,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> sorted_index
){

    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>X_data.size(0)-1){return;}
    int idx=0;
    for (int p = 0;p<nd;p++) {
        idx += multiply[p]*(int)floor( *int_mult * (X_data[i][p] - alpha[p]) / *edge);
    }
    int perm_val = device_lookup(perm,idx,hash_size);
    sorted_index[atomicAdd(&global_unique[perm_val],1)+global_vector_counter_cum[perm_val]] = i;
}