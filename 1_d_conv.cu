
#pragma once
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

template<typename T>
__device__ inline  T cube(T x){
    return x*x*x;
};

template<typename T, int nd>
__device__ T rbf_simple(T x[],T y[]){
    T tmp=0;
    for (int k=0;k<nd;k++){
        tmp += square<T>(x[k]-y[k]);
    };
    return expf(-tmp);
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
    return expf(-dist/(2* *ls));
};
template<typename T, int nd>
__device__ inline static T rbf_grad_ls(T x[],T y[],const T *ls){
    T dist=square_dist<T,nd>(x,y);
    return expf(-dist/(2* *ls))*dist/square<T>(*ls);
};
template<typename T, int nd>
__device__ inline static T rbf_grad_dist_abs(T x[],T y[],const T *ls){
    T dist=square_dist<T,nd>(x,y);
    return expf(-dist/(2* *ls))*sqrt(dist)/(*ls);
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

template <typename scalar_t>
__device__ scalar_t calculate_lagrange( //Try using double precision!
        scalar_t *l_p,
        scalar_t & x_ij,
        int & feature_num,
        int & a,
        int & b
        ){
    scalar_t res=1;
    for (int i=a; i<b;i++){
        if (i!=feature_num){ //Calculate the Laplace feature if i!=m...
            res *= (x_ij-l_p[i])/(l_p[i]-l_p[feature_num]);
        }
    }
    return res;
}

template <typename scalar_t,int nd>
__device__ scalar_t calculate_lagrange_product( //Tends to become really inaccurate for high dims and "too many lagrange nodes"
        scalar_t *l_p,
        scalar_t *x_i,
            int *combs,
        scalar_t b,
        int *node_cum_shared){
    scalar_t tmp;
    for (int i=0; i<nd;i++){
        tmp = calculate_lagrange(l_p, x_i[i], combs[i], node_cum_shared[i], node_cum_shared[i + 1]);
        if (tmp==0){
            b=(scalar_t) 0.;
            break;
        } else{
            b *=tmp;  //Bug might be kronecker delta effect,i.e. sometimes x_i[]
        }
    }
    return b;
}

template <typename scalar_t,int nd>
__device__ scalar_t calculate_barycentric_lagrange(//not sure this is such a great idea, when nan how avoid...
        scalar_t *l_p,
        scalar_t *w_shared,
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
            b*= l_x[i] * w_shared[combs[i]] / (x_i[i] - l_p[combs[i]]);
        }
    }
    return b;
}


template <typename scalar_t,int nd>
__device__ scalar_t calculate_barycentric_lagrange_one_pass(//not sure this is such a great idea, when nan how avoid...
        scalar_t *y_in,
        scalar_t * center,
        const int *combs,
        scalar_t * factor,
        scalar_t *l_p,
        scalar_t *w,
        scalar_t b,
        const int * node_cum_shared

){
    bool pBoolean;
    scalar_t x_i;
    scalar_t l_x;
    for (int i = 0; i < nd; i++) {
        x_i = *factor*(y_in[i]-center[i]); //correct y_in, correct center
        scalar_t tmp = 0.;
        pBoolean=false;
        for (int l=node_cum_shared[i];l<node_cum_shared[i+1];l++){ //node_cum_index is wrong or l_p is loaded completely incorrectly!
            if (x_i==l_p[l]){
                pBoolean=true;
                tmp=1;
                break;
            }else{
                tmp += w[l]/(x_i-l_p[l]);
            }
        }
        l_x = 1/tmp;
        if (pBoolean){
            if (x_i!=l_p[combs[i]]){
                b=(scalar_t) 0;
                break;
            }
        }else{
            b*= l_x * w[combs[i]] / (x_i - l_p[combs[i]]);
        }
    }
    return b;
}


__device__ int calculate_box_ind(int &current_thread_idx,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> counts,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> x_box_idx){
    int nr_of_counts = counts.size(0); //remember 0 included!
    for (int i=0;i<nr_of_counts-1;i++){
        if ( current_thread_idx>=counts[i] && current_thread_idx<counts[i+1]){
            return x_box_idx[i];
        }
    }
}


template <typename scalar_t,int nd>
__device__ scalar_t get_2_norm(scalar_t * dist){
    scalar_t acc=0;
#pragma unroll
    for (int k = 0; k < nd; k++) {
        acc+= square<scalar_t>(dist[k]);
    }
    return sqrt(acc);
}


template <typename scalar_t,int nd>
__global__ void skip_conv_1d(const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
                             const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                             const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> b_data,
                             torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> output,
                             scalar_t * ls,
                             const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> x_boxes_count,
                             const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> y_boxes_count,
                             const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> x_box_idx,
                             const torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> interactions_x_parsed,
                             const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> interactions_y
){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    unsigned int x_n = X_data.size(0);
    if (i>x_n-1){return;}
    scalar_t x_i[nd];
    scalar_t y_j[nd];
    scalar_t acc;
    int box_ind,start,end,int_j;
    box_ind = calculate_box_ind(i,x_boxes_count,x_box_idx);
    for (int k=0;k<nd;k++){
        x_i[k] = X_data[i][k];
    }
//    printf("thread %i: %i\n",i,box_ind);
    for (int b_ind=0; b_ind < b_data.size(1); b_ind++) { //for all dims of b
        acc=0;
        for (int j = interactions_x_parsed[box_ind][0]; j < interactions_x_parsed[box_ind][1]; j++) { // iterate through every existing ybox
            int_j = interactions_y[j];
            start = y_boxes_count[int_j]; // 0 to something
            end = y_boxes_count[int_j + 1]; // seomthing
            for (int j_2 = start; j_2 < end; j_2++) {
                for (int k = 0; k < nd; k++) {
                    y_j[k] = Y_data[j_2][k];
                };
                acc += rbf<scalar_t,nd>(x_i, y_j, ls) * b_data[j_2][b_ind];

            }
        }
        output[i][b_ind] = acc;
    }
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
                        torch_load_b_v2<scalar_t, nd>(b_ind,y_idx_reordering[j], bj, b_data);
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
__global__ void skip_conv_1d_shared_transpose(const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
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
                                    const torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> interactions_y_parsed,
                                    const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> interactions_x

){
    int i,box_ind,start,end,a,b,int_m,y_idx_reorder,b_size,interactions_a,interactions_b;
    box_ind = block_box_indicator[blockIdx.x];
    a = y_boxes_count[box_ind];
    b = y_boxes_count[box_ind+1];
    i = a + threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x; // Use within box, block index i.e. same size as indicator...
    scalar_t y_i[nd];
    scalar_t b_i;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
    scalar_t *buffer = reinterpret_cast<scalar_t *>(my_smem);
    scalar_t *xj = &buffer[0];
    if (i<b) {
        y_idx_reorder = y_idx_reordering[i];
        for (int k = 0; k < nd; k++) {
            y_i[k] = Y_data[y_idx_reorder][k];
        }
    }

    __syncthreads();
    interactions_a = interactions_y_parsed[box_ind][0];
    interactions_b= interactions_y_parsed[box_ind][1];
    b_size = b_data.size(1);
    for (int b_ind=0; b_ind<b_size; b_ind++) {
        if (i<b) {
            b_i = b_data[y_idx_reorder][b_ind];
        }
        if (interactions_a>-1) {
            for (int m = interactions_a; m < interactions_b; m++) {
                int_m = interactions_x[m];
                start = x_boxes_count[int_m]; // 0 to something
                end = x_boxes_count[int_m + 1]; // seomthing
                for (int jstart = start, tile = 0; jstart < end; jstart += blockDim.x, tile++) {
                    int j = start + tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
                    if (j < end) { // we load yj from device global memory only if j<ny
                        torch_load_y_v2<scalar_t, nd>(x_idx_reordering[j], xj, X_data);
                    }
                    __syncthreads();
                    if (i < b) { // we compute x1i only if needed
                        scalar_t *xjrel = xj; // Loop on the columns of the current block.
                        for (int jrel = 0; (jrel < blockDim.x) && (jrel < end - jstart); jrel++, xjrel += nd) {
                            atomicAdd(&output[x_idx_reordering[jstart+jrel]][b_ind],rbf<scalar_t, nd>(xjrel, y_i, ls) * b_i); //for each p, sum accross x's...
                        }
                    }
                    __syncthreads(); //Lesson learned! Thread synching really important for cuda programming and memory loading when indices are dependent on threadIdx.x!
                };
            }
        __syncthreads();

        }
    };
    __syncthreads();

}


template <typename scalar_t,int nd>
__device__ void lagrange_data_load(scalar_t w[],
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
    int i,a,b,cheb_data_size,idx_reorder,lagrange_n,b_size,box_ind;
    box_ind = indicator[blockIdx.x];
    b_size = output.size(1);
    cheb_data_size = combinations.size(0);
    scalar_t factor = 2/(*edge); //rescaling data to langrange node interval (-1,1)
    lagrange_n = lap_nodes.size(0);
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
    scalar_t *w_shared = &buffer[blockDim.x*nd+1+nd+lagrange_n];

    if (threadIdx.x<nd+1){
        if (threadIdx.x==0){
            node_cum_shared[threadIdx.x]=0;
        }else{
            node_cum_shared[threadIdx.x] = node_list_cum[threadIdx.x-1];
        }
    }
    __syncthreads();

    for (int jstart = 0, tile = 0; jstart < lagrange_n; jstart += blockDim.x, tile++) {
        int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
        if (j < lagrange_n) { // we load yj from device global memory only if j<ny
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
__global__ void lagrange_shared_transpose(
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

    int lagrange_n,i, box_ind, a, b,cheb_data_size,idx_reorder,b_size;
    cheb_data_size = combinations.size(0);
    scalar_t factor = 2/(*edge);
    box_ind = indicator[blockIdx.x];
    a = x_boxes_count[box_ind];
    b = x_boxes_count[box_ind + 1];
    i = a + threadIdx.x + box_block_indicator[blockIdx.x] * blockDim.x; // current thread
    lagrange_n = lap_nodes.size(0);
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
    scalar_t *w_shared = &buffer[(blockDim.x+1) * nd+blockDim.x+1+lagrange_n];
    if (threadIdx.x<nd+1){
        if (threadIdx.x==0){
            node_cum_shared[0] = (int) 0;

        }else{
            node_cum_shared[threadIdx.x] = node_list_cum[threadIdx.x-1];
        }
    }
    __syncthreads();

    for (int jstart = 0, tile = 0; jstart < lagrange_n; jstart += blockDim.x, tile++) {
        int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
        if (j < lagrange_n) { // we load yj from device global memory only if j<ny
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

template <typename scalar_t,int nd>
__global__ void lagrange_shared_v2(
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> Y_data,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> lap_nodes,
        const torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> combinations,
        torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> output,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> indicator,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> box_block_indicator,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> y_cum, //error is here! I think it does an access here when its not supposed to...
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers,
        const scalar_t * edge,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> idx_reordering,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> node_list_cum,
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> W

){
    int i,a,cheb_data_size,lagrange_n,b_size,box_ind,i_calc;
    box_ind = indicator[blockIdx.x];
    b_size = output.size(1);
    cheb_data_size = combinations.size(0);
    scalar_t factor = 2/(*edge); //rescaling data to langrange node interval (-1,1)
    lagrange_n = lap_nodes.size(0);
    a = cheb_data_size*box_ind;
    i = a+ threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x; // current thread
    i_calc = threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x;
    int x_i[nd];
    if (i_calc<cheb_data_size) {
        for (int k = 0; k < nd; k++) {
            x_i[k] = combinations[i_calc][k];
        }
    }
    extern __shared__ int int_buffer[];
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
    scalar_t *buffer = reinterpret_cast<scalar_t *>(my_smem);
    int *node_cum_shared = &int_buffer[0];
    scalar_t *yj = &buffer[1+nd];
    scalar_t *bj = &buffer[(blockDim.x+1) * nd+1];
    scalar_t *center = &buffer[(blockDim.x+1) * nd+blockDim.x+1];
    scalar_t *l_p = &buffer[(blockDim.x+1) * nd+blockDim.x+1+nd];
    scalar_t *w_shared = &buffer[(blockDim.x+1) * nd+blockDim.x+1+lagrange_n+nd];
    if (threadIdx.x<nd+1){
        if (threadIdx.x<nd){
            center[threadIdx.x]=centers[box_ind][threadIdx.x];
        }
        if (threadIdx.x==0){
            node_cum_shared[threadIdx.x]=0;
        }else{
            node_cum_shared[threadIdx.x] = node_list_cum[threadIdx.x-1];
        }
    }
    __syncthreads();

    for (int jstart = 0, tile = 0; jstart < lagrange_n; jstart += blockDim.x, tile++) {
        int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
        if (j < lagrange_n) { // we load yj from device global memory only if j<ny
            l_p[j] = lap_nodes[j];
            w_shared[j] = W[j];
        }
        __syncthreads();
    }

    __syncthreads();
    scalar_t acc;
    int Y_start = y_cum[box_ind];
    int Y_end = y_cum[box_ind+1];
    for (int b_ind=0; b_ind<b_size; b_ind++) {
        acc=0;
        for (int jstart = Y_start, tile = 0; jstart < Y_end; jstart += blockDim.x, tile++) {
            int j =jstart+ tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
            if (j < Y_end) { // we load yj from device global memory only if j<ny
                torch_load_y_v2<scalar_t, nd>(idx_reordering[j], yj, Y_data);
                torch_load_b_v2<scalar_t, nd>(b_ind,idx_reordering[j],bj, b_data);
            }
            __syncthreads();
            if (i_calc < cheb_data_size) { // we compute x1i only if needed
                scalar_t *yjrel = yj; // Loop on the columns of the current block.
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < Y_end - jstart); jrel++, yjrel += nd) {
                    acc += calculate_barycentric_lagrange_one_pass<scalar_t,nd>(
                            yjrel,
                            center,
                            x_i,
                            &factor,
                             l_p,
                            w_shared,
                            bj[jrel],
                    node_cum_shared
                            );
                }
            }
            __syncthreads();

        };
        if (i_calc < cheb_data_size) { // we compute x1i only if needed
            output[i][b_ind] = acc;
        }
    };
    __syncthreads();

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
template <typename scalar_t, int nd>
__device__ int get_global_index(
                                int i,
                                const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
                                const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> alpha,
                                const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> multiply,
                                const scalar_t * edge,
                                torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> old_perm,
                                torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> depth_perm_idx_adder,
                                const int * current_depth,
                                const int * dim_fac
                                ){
    int global_idx=0;
    scalar_t cur_edge;
    scalar_t cur_alpha[nd];
    int bin;
    int idx;
    for (int p = 0;p<nd;p++) {
        cur_alpha[p]=alpha[p];
    }
    for (int d=0;d<*current_depth;d++){
        cur_edge = *edge/(float)pow(2.0,d);
        idx=0;
        for (int p = 0;p<nd;p++) {
            bin = (int)floor( 2 * (X_data[i][p] - cur_alpha[p])/cur_edge);
            cur_alpha[p] = cur_alpha[p]+bin*cur_edge/2;
            idx += multiply[p]*bin;
        }
//        if (i<100){
//            printf("inside index %i : depth perm adder %i  idx %i  \n",d,depth_perm_idx_adder[d],idx);
//            printf("global_index in depth %i = %i \n",d,old_perm[depth_perm_idx_adder[d]+idx+global_idx*(*dim_fac)]);
//        }


        global_idx = old_perm[depth_perm_idx_adder[d]+idx+global_idx*(*dim_fac)];
    }
    cur_edge = cur_edge/2;
    idx=0;
    for (int p = 0;p<nd;p++) {
        bin = (int)floor( 2 * (X_data[i][p] - cur_alpha[p])/cur_edge);
        idx += multiply[p]*bin;
    }
//    if (i<100) {
//        printf("global_index = %i \n", global_idx * (*dim_fac) + idx);
//        printf("idx bottom line = %i \n", idx);
//    }
    return global_idx*(*dim_fac)+idx;
}


template <typename scalar_t, int nd>
__global__ void box_division_cum(
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> alpha,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> multiply,
        const scalar_t * edge,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> global_vector_counter_cum,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> perm,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> old_perm,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> depth_perm_idx_adder,
        const int * current_depth,
        const int * dim_fac

){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>X_data.size(0)-1){return;}
    if (*current_depth==0){
        int idx=0;
        for (int p = 0;p<nd;p++) {
            idx += multiply[p]*(int)floor( 2 * (X_data[i][p] - alpha[p]) / *edge);
        }
        atomicAdd(&global_vector_counter_cum[perm[idx]+1],1);
        return;
    }else{
        int global_index;
        global_index = get_global_index<scalar_t,nd>(
                                                    i,
                                                    X_data,
                                                    alpha,
                                                    multiply,
                                                    edge,
                                                    old_perm,
                                                    depth_perm_idx_adder,
                                                    current_depth,
                                                    dim_fac
                );
        atomicAdd(&global_vector_counter_cum[perm[global_index]+1],1);
        return;
    }

}

template <typename scalar_t, int nd>
__global__ void center_perm(const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers_natural,
                            const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> alpha,
                            const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> multiply,
                            const scalar_t * edge,
                            torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> perm,
                            torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> old_perm,
                            torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> depth_perm_idx_adder,
                            const int * current_depth,
                            const int * dim_fac
                            ){

    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>centers_natural.size(0)-1){return;}

    if (*current_depth==0){
        int idx=0;
        for (int p = 0;p<nd;p++) {
            idx += multiply[p]*(int)floor( 2 * (centers_natural[i][p] - alpha[p]) / *edge);
        }
        perm[idx] = i;
        return;
    }else{
        int global_index;
        global_index = get_global_index<scalar_t,nd>(
                i,
                centers_natural,
                alpha,
                multiply,
                edge,
                old_perm,
                depth_perm_idx_adder,
                current_depth,
                dim_fac
        );

        perm[global_index]=i;
        return;
    }

}


template <typename scalar_t, int nd>
__global__ void box_division_assign(
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> alpha,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> multiply,
        const scalar_t * edge,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> global_vector_counter_cum,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> global_unique,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> sorted_index,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> perm,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> old_perm,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> depth_perm_idx_adder,
        const int * current_depth,
        const int * dim_fac
){

    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>X_data.size(0)-1){return;}
    int idx;
    if (*current_depth==0){
        idx=0;
        for (int p = 0;p<nd;p++) {
            idx += multiply[p]*(int)floor( 2 * (X_data[i][p] - alpha[p]) / *edge);
        }
        sorted_index[atomicAdd(&global_unique[perm[idx]],1)+global_vector_counter_cum[perm[idx]]] = i;
        return;
    }else{
        int global_index;
        global_index = get_global_index<scalar_t,nd>(
                i,
                X_data,
                alpha,
                multiply,
                edge,
                old_perm,
                depth_perm_idx_adder,
                current_depth,
                dim_fac
        );

        sorted_index[atomicAdd(&global_unique[perm[global_index]],1)+global_vector_counter_cum[perm[global_index]]] = i;
        return;
    }
}




//

//box_idx , data_point_idx.
//0 : [- 1- 1- 1- 1-]
//1 : [- 1- -1- -1- -1- 1]
//global_vector_counter: [32 50 ... 64] cumsum [0 32 82 ... n]
//n [0 0 0 0 0 0 0 0 ] indices in the sorted box order
// new global vector counter... + cumsum[box_indx] + value of col.

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
__global__ void get_cheb_idx_data(
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> cheb_nodes,
        torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> cheb_data,
        torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> cheb_idx,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> indices
){
    int n = cheb_data.size(0);
    int i = threadIdx.x+blockIdx.x*blockDim.x; // Thread nr
    if (i>n-1){return;}
    int sampled_i = indices[i];
    int lap_nodes = cheb_nodes.size(0);
    int idx;
    int tmp;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
    scalar_t *buffer = reinterpret_cast<scalar_t *>(my_smem);
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
        const int * nr_interpolation_points,
        const int * small_field_limit
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
    for (int k=0;k<nd;k++){
        cx[k] = centers_X[bx][k];
        cy[k] = centers_Y[by][k];
        distance[k]=cy[k] - cx[k];
        interaction_size = unique_og_X[bx]+unique_og_Y[by];
    }

    if (get_2_norm<scalar_t,nd>(distance)>=(*edge*2)){
        if (interaction_size<(2* *nr_interpolation_points)){
            is_small_field[i]=true;
        }else{
            is_far_field[i]=true;
        }
    }else{
        if (interaction_size<(2* *small_field_limit)){
            is_small_field[i]=true;
        }
    }
}

template <typename scalar_t, int nd>
__global__ void boolean_separate_interactions_small_var_comp(
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers_X,
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> centers_Y,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> unique_og_X,
        const torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> unique_og_Y,
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> eff_var_X,
        const torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> eff_var_Y,
        const torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> interactions,
        const scalar_t * edge,
        torch::PackedTensorAccessor64<bool,1,torch::RestrictPtrTraits> is_far_field,
        torch::PackedTensorAccessor64<bool,1,torch::RestrictPtrTraits> is_small_field,
        const int * nr_interpolation_points,
        const int * small_field_limit,
        const scalar_t * eff_var_limit
){
    int i = threadIdx.x+blockIdx.x*blockDim.x; // Thread nr
    int n = interactions.size(0);
    if (i>n-1){return;}
    int bx = interactions[i][0];
    int by = interactions[i][1];
    int interaction_size;
    scalar_t tot_var;
    scalar_t distance[nd];
    scalar_t cx[nd];
    scalar_t cy[nd];
    for (int k=0;k<nd;k++){
        cx[k] = centers_X[bx][k];
        cy[k] = centers_Y[by][k];
        distance[k]=cy[k] - cx[k];
        interaction_size = unique_og_X[bx]+unique_og_Y[by];
        tot_var = eff_var_X[bx] + eff_var_Y[by];
    }

    if (tot_var<= *eff_var_limit){
        if (interaction_size<(2* *small_field_limit)){
            is_small_field[i]=true;
        }else{
            is_far_field[i]=true;
        }
    }else{
        if (get_2_norm<scalar_t,nd>(distance)>=(*edge*2)){
            if (interaction_size<(2* *nr_interpolation_points)){
                is_small_field[i]=true;
            }else{
                is_far_field[i]=true;
            }
        }else{
            if (interaction_size<(2* *small_field_limit)){
                is_small_field[i]=true;
            }
        }
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
        const torch::PackedTensorAccessor64<scalar_t,2,torch::RestrictPtrTraits> input,
        torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> maxOut,
        torch::PackedTensorAccessor64<scalar_t,1,torch::RestrictPtrTraits> minOut
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


__global__ void transpose_to_existing_only_tree_perm(
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> perm,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> removed_indices_x
){//Bottle neck perhaps, switch to binary search
    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    if (tid > perm.size(0) - 1){return;}
    unsigned int nx = removed_indices_x.size(0);
    int box_idx = perm[tid];
    for (int i=0;i<nx;i++){
        if(box_idx == removed_indices_x[i]){
            perm[tid] = -1;
        }
        if(box_idx > removed_indices_x[i]){
            perm[tid] -=1;
        }

    }
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


__global__ void transpose_to_existing_only_X(
        torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> interactions,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> old_new_map_X
){
    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    if (tid>interactions.size(0)-1){return;}
    int interaction_x = interactions[tid][0];
    interactions[tid][0] = old_new_map_X[interaction_x];
}

__global__ void transpose_to_existing_only_Y(
        torch::PackedTensorAccessor64<int,2,torch::RestrictPtrTraits> interactions,
        torch::PackedTensorAccessor64<int,1,torch::RestrictPtrTraits> old_new_map_Y
){
    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    if (tid>interactions.size(0)-1){return;}
    int interaction_y = interactions[tid][1];
    interactions[tid][1] = old_new_map_Y[interaction_y];
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
    int limit=10000;
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





