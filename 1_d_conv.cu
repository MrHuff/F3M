

#pragma once
#include <iostream>
#include <ostream>
#include <cublas_v2.h>
#include <random>
#include <stdexcept>
#include <torch/torch.h> //for n00bs like me, direct translation to python rofl
#define BLOCK_SIZE 192
#define MAXTHREADSPERBLOCK 1024
#define SHAREDMEMPERBLOCK 49152
//template<typename T, int nd>
//using rbf_pointer = T (*) (T[], T[],const T *);


template<typename scalar_t>
__inline__ __device__ scalar_t warpReduceSum(scalar_t val) {
#pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF,val, offset);
    return val;
}
template<typename scalar_t>
__inline__ __device__ scalar_t blockReduceSum(scalar_t val,scalar_t shared[]) {

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum<scalar_t>(val);     // Each warp performs partial reduction

    if (lane==0){
        shared[wid]=val;
    } // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    if (threadIdx.x < blockDim.x / warpSize){
        val = shared[lane];
    }else{
        val=0;
    }
    __syncthreads();              // Wait for all partial reductions

    if (wid==0){
        val = warpReduceSum<scalar_t>(val);
    }  //Final reduce within first warp

    return val;
}


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
__device__ inline static T rbf(T x[],T y[],const T *ls){
    T tmp=0;
    for (int k=0;k<nd;k++){
        tmp += square<T>(x[k]-y[k]);
    };
    return expf(-tmp/square<T>(*ls));
};
template<typename T, int nd>
__device__ inline static T rbf_grad(T x[],T y[],const T *ls){
    T tmp=0;
    for (int k=0;k<nd;k++){
        tmp += square<T>(x[k]-y[k]);
    };
    return expf(-tmp/square<T>(*ls))*tmp/cube<T>(*ls);
};
//template<typename T, int nd>
//__device__ rbf_pointer<T> rbf_pointer_func = rbf<T>;
//template<typename T, int nd>
//__device__ rbf_pointer<T> rbf_pointer_grad = rbf_grad<T>;
//

template <typename scalar_t,int nd>
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
template <typename scalar_t,int nd>
__device__ inline static void torch_load_y_v2(int reorder_index, scalar_t *shared_mem,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> y
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
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b){
    shared_mem[threadIdx.x] = b[reorder_index][col_index];
}


//Consider caching the kernel value if b is in Nxd.
template <typename scalar_t,int nd>
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
            acc+= rbf<scalar_t,nd>(x_i,y_j,ls)*b_data[p][b_ind];
        };
        output[i][b_ind]=acc;
    }
    __syncthreads();

};

template <typename scalar_t>
__device__ scalar_t calculate_laplace( //Try using double precision!
        scalar_t l_p[],
        scalar_t & x_ij,
        int & feature_num,
        int & a,
        int & b
        ){
    scalar_t res=1.0;
    for (int i=a; i<b;i++){
        if (i!=feature_num){ //Calculate the Laplace feature if i!=m...
            res *= (x_ij-l_p[i])/(l_p[i]-l_p[feature_num]);
//            res += log(x_ij-l_p[i])-log(l_p[i]-l_p[feature_num]);
        }
    }
    return res;
}

template <typename scalar_t,int nd>
__device__ scalar_t calculate_laplace_product( //Tends to become really inaccurate for high dims and "too many lagrange nodes"
        scalar_t l_p[],
        scalar_t x_i[],
        int combs[],
        scalar_t b,
        int node_cum_shared[]){
    for (int i=0; i<nd;i++){
        b *= calculate_laplace(l_p, x_i[i], combs[i],node_cum_shared[i],node_cum_shared[i+1]); //Interpolating b by constructing langrange interpolation at x's
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

template <typename scalar_t,int nd>
__device__ void xy_l1_dist(
        scalar_t * c_X,
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> c_Y,
        scalar_t * dist){
#pragma unroll
    for (int k = 0; k < nd; k++) {
        dist[k] = c_Y[k]-c_X[k];
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

template <typename scalar_t>
__device__ bool far_field_bool(scalar_t & l2_dist,scalar_t * edge){
    return l2_dist>=(*edge*2+1e-6);
}

template <typename scalar_t,int nd>
__device__ bool far_field_comp(scalar_t * c_X,scalar_t * c_Y,scalar_t * edge){
    scalar_t dist[nd];
    xy_l1_dist<scalar_t>(c_X,c_Y,dist);
    scalar_t l2 = get_2_norm(dist);
    return far_field_bool(l2,edge);
}





template <typename scalar_t,int nd>
__global__ void skip_conv_1d(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data,
                             torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                             scalar_t * ls,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_boxes_count,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> y_boxes_count,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_box_idx,
                             const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> interactions_x_parsed,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> interactions_y
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
        acc=0.0;
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
__global__ void skip_conv_1d_shared(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Y_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data,
                             torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
                             scalar_t * ls,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_boxes_count,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> y_boxes_count,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> block_box_indicator,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> box_block_indicator,
                            const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_idx_reordering,
                            const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> y_idx_reordering,
                            const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> interactions_x_parsed,
                            const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> interactions_y

){
    int box_ind,start,end,a,b,int_m,x_idx_reorder,b_size,interactions_a,interactions_b;
    box_ind = block_box_indicator[blockIdx.x];
    a = x_boxes_count[box_ind];
    b = x_boxes_count[box_ind+1];
    int i = a + threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x; // Use within box, block index i.e. same size as indicator...

    scalar_t x_i[nd];
    scalar_t acc;
    extern __shared__ scalar_t buffer[];
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
//    box_ind = calculate_box_ind(i,x_boxes_count,x_box_idx);
//    printf("thread %i: %i\n",i,box_ind);
    interactions_a = interactions_x_parsed[box_ind][0];
    interactions_b= interactions_x_parsed[box_ind][1];
    if (interactions_a>-1) {
        for (int b_ind = 0; b_ind < b_size; b_ind++) { //for all dims of b
            acc = 0.0;

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

//Implement shuffle reduce.
template <typename scalar_t,int nd>
__global__ void laplace_shared(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b_data, //also put b's in shared mem for maximum perform.
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> lap_nodes,
        const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> combinations,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> indicator,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> box_block_indicator,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> x_boxes_count, //error is here! I think it does an access here when its not supposed to...
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers,
        const scalar_t * edge,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> idx_reordering,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> node_list_cum
){
    int box_ind = indicator[blockIdx.x];
    int a,b,cheb_data_size,idx_reorder;
    cheb_data_size = combinations.size(0);
    scalar_t factor = 2./(*edge); //rescaling data to langrange node interval (-1,1)
    a = x_boxes_count[box_ind];
    b = x_boxes_count[box_ind+1];
    int i = a+ threadIdx.x+box_block_indicator[blockIdx.x]*blockDim.x; // current thread
    scalar_t x_i[nd];
    scalar_t b_i;
    if (i<b) {
        idx_reorder = idx_reordering[i];
        for (int k = 0; k < nd; k++) {
            x_i[k] = factor*(X_data[idx_reorder][k]-centers[box_ind][k]);
        }
    }
    extern __shared__ int int_buffer[];
    extern __shared__ scalar_t buffer[];
    int *laplace_n = &int_buffer[0];
    int *node_cum_shared = &int_buffer[1];
    int *yj = &int_buffer[2+nd];
    scalar_t *l_p = &buffer[blockDim.x*nd+2+nd];
    int b_size = output.size(1);
    if (threadIdx.x==0){
        laplace_n[0] = lap_nodes.size(0);
        node_cum_shared[0] = (int) 0;
    }
    __syncthreads();
    if (threadIdx.x<nd){
        node_cum_shared[threadIdx.x+1] = node_list_cum[threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x<laplace_n[0]){
        l_p[threadIdx.x] = lap_nodes[threadIdx.x];
    }
    __syncthreads();
    scalar_t *shared_sum = &buffer[blockDim.x*nd+2+nd+laplace_n[0]];
    scalar_t val=0;
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
            int *yjrel = yj; // Loop on the columns of the current block.
            for (int jrel = 0; (jrel < blockDim.x) && (jrel < cheb_data_size - jstart); jrel++, yjrel += nd) {
                if (i < b) { // we compute x1i only if needed
                    //Do shuffle if possible...
                    val = calculate_laplace_product<scalar_t,nd>(l_p, x_i, yjrel, b_i, node_cum_shared);
                }
                __syncthreads();
                val = blockReduceSum<scalar_t>(val, shared_sum); //Make sure entire block actually does reduction! Else funny business ensues
                if (threadIdx.x == 0) {
                    atomicAdd(&output[jstart + jrel + box_ind * cheb_data_size][b_ind], val);
                }
                __syncthreads();
//                    atomicAdd(&output[jstart+jrel+box_ind*cheb_data_size][b_ind],calculate_laplace_product(l_p, x_i, yjrel,b_i,laplace_n)); //for each p, sum accross x's...

            }
            __syncthreads();

        };
    };
    __syncthreads();

}

template <typename scalar_t,int nd>
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
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> idx_reordering,
        const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> node_list_cum
) {

    int box_ind, a, b,cheb_data_size,idx_reorder,b_size;
    cheb_data_size = combinations.size(0);
    scalar_t factor = 2./(*edge);
    box_ind = indicator[blockIdx.x];
    a = x_boxes_count[box_ind];
    b = x_boxes_count[box_ind + 1];
    int i = a + threadIdx.x + box_block_indicator[blockIdx.x] * blockDim.x; // current thread
    scalar_t x_i[nd];
    if (i<b) {
        idx_reorder = idx_reordering[i];
        for (int k = 0; k < nd; k++) {
            x_i[k] = factor*(X_data[idx_reorder][k]-centers[box_ind][k]);
        }
    }
    scalar_t acc;
    extern __shared__ int int_buffer[];
    extern __shared__ scalar_t buffer[];
    int *laplace_n = &int_buffer[0];
    int *node_cum_shared = &int_buffer[1];
    int *yj = &int_buffer[2+nd];
    scalar_t *bj = &buffer[(blockDim.x+1) * nd+2];
    scalar_t *l_p = &buffer[(blockDim.x+1) * nd+blockDim.x+2];
    if (threadIdx.x==0){
        laplace_n[0] = lap_nodes.size(0);
        node_cum_shared[0] =(int) 0;
    }
    __syncthreads();

    if (threadIdx.x < laplace_n[0]) {
        l_p[threadIdx.x] = lap_nodes[threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x<nd){
        node_cum_shared[threadIdx.x+1] = node_list_cum[threadIdx.x];
    }
    __syncthreads();

    b_size =  b_data.size(1);
    for (int b_ind = 0; b_ind <b_size; b_ind++) { //for all dims of b
        acc = 0.0;
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
                    acc += calculate_laplace_product<scalar_t,nd>(l_p, x_i, yjrel, bj[jrel],node_cum_shared);
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

//Both these needs updating, i.e. pass additional appendage vector and pass interaction vector.

//[[0,1],[0,2],[0,3],[0,4],[0,5]...] ~ O(n_b^2x2)
//Move to shared mem experiment!
//Thrust
template <typename scalar_t,int nd>
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
    int box_ind,a,b,cheb_data_size,int_m,interactions_a,interactions_b,b_size;
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
    scalar_t *bj = &buffer[blockDim.x*nd];
//    scalar_t *distance = &buffer[cheb_data_size*(nd+1)];
    scalar_t *cX_i = &buffer[blockDim.x*(nd+1)]; //Get another shared memory pointer.

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
    interactions_a = interactions_x_parsed[box_ind][0];
    interactions_b= interactions_x_parsed[box_ind][1];
    b_size = b_data.size(1);
    if (interactions_a>-1) {
        for (int b_ind = 0;b_ind <b_size ; b_ind++) { //for all dims of b A*b, b \in \mathbb{R}^{n\times d}, d>=1.
            acc = 0.0;
            for (int m = interactions_a; m < interactions_b; m++) {
                int_m = interactions_y[m];
                for (int jstart = 0, tile = 0; jstart < cheb_data_size; jstart += blockDim.x, tile++) {
                    int j = tile * blockDim.x + threadIdx.x; //periodic threadIdx.x you dumbass. 0-3 + 0-2*4
                    if (j < cheb_data_size) {
                        for (int k = 0; k < nd; k++) {
                            yj[nd * threadIdx.x+k] = cheb_data[j][k]+centers_Y[int_m][k] - cX_i[k];
                        }
                        bj[threadIdx.x] = b_data[j + int_m * cheb_data_size][b_ind];
                    }
                    __syncthreads(); //Need to be smart with this, don't slow down the others!
                    if (i < b) { // we compute x1i only if needed
                        scalar_t *yjrel = yj; // Loop on the columns of the current block.
                        for (int p = 0; (p < blockDim.x) && (p < cheb_data_size - jstart); p++, yjrel += nd) {
                            acc += rbf<scalar_t,nd>(x_i, yjrel, ls)* bj[p]; //sums incorrectly cause pointer is fucked not sure if allocating properly
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

