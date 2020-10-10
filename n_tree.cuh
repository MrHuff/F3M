//
// Created by rhu on 2020-03-28.
//

#pragma once
#include "1_d_conv.cu"
#include "utils.h"
#include <vector>
#include "algorithm"
#include "FFM_utils.cu"
//template<typename T>


void print_tensor(torch::Tensor & tensor){
    std::cout<<tensor<<std::endl;
}

template<typename T>
at::ScalarType dtype() { return at::typeMetaToScalarType(caffe2::TypeMeta::Make<T>()); }


template<typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> calculate_edge(const torch::Tensor &X,const torch::Tensor &Y,const std::string & gpu_device){
//    torch::Tensor Xmin = std::get<0>(X.min(0));
//    torch::Tensor Xmax = std::get<0>(X.max(0));
//    torch::Tensor Ymin = std::get<0>(Y.min(0));
//    torch::Tensor Ymax = std::get<0>(Y.max(0));
    torch::Tensor Xmin = torch::zeros(X.size(1)).toType(dtype<scalar_t>()).to(gpu_device);
    torch::Tensor Xmax = torch::zeros(X.size(1)).toType(dtype<scalar_t>()).to(gpu_device);
    torch::Tensor Ymin = torch::zeros(X.size(1)).toType(dtype<scalar_t>()).to(gpu_device);
    torch::Tensor Ymax = torch::zeros(X.size(1)).toType(dtype<scalar_t>()).to(gpu_device);
    dim3 blockSize,gridSize;
    blockSize.x = 1024;
    gridSize.x = 20;
    reduceMaxMinOptimizedWarpMatrix<scalar_t,nd><<<gridSize,blockSize,8>>>(X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                        Xmax.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                                                                        Xmin.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();
    reduceMaxMinOptimizedWarpMatrix<scalar_t,nd><<<gridSize,blockSize,8>>>(Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                        Ymax.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                                                                        Ymin.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();
    torch::Tensor edge = torch::cat({Xmax - Xmin, Ymax - Ymin}).max();
    return std::make_tuple(edge*1.01,Xmin,Ymin);
};
template<typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor> calculate_edge_X(const torch::Tensor &X,const std::string & gpu_device){
    torch::Tensor Xmin = torch::zeros(X.size(1)).toType(dtype<scalar_t>()).to(gpu_device);
    torch::Tensor Xmax = torch::zeros(X.size(1)).toType(dtype<scalar_t>()).to(gpu_device);
    dim3 blockSize,gridSize;
    blockSize.x = 1024;
    gridSize.x = 10;
    reduceMaxMinOptimizedWarpMatrix<scalar_t,nd><<<gridSize,blockSize,8>>>(X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                                    Xmax.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                                                                                            Xmin.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
                            );
    cudaDeviceSynchronize();
    torch::Tensor edge = (Xmax - Xmin).max();
    return std::make_tuple(edge*1.01,Xmin);
};


template <typename scalar_t,int dim>
struct n_tree_cuda{
    torch::Tensor &data;
    torch::Tensor edge,
    xmin,
    sorted_index,
    unique_counts,
    multiply_cpu,
    multiply_cpu_base,
    multiply_gpu_base,
    multiply_gpu,
    box_indices_sorted,
    centers_unique,
    centers,
    unique_counts_cum,
    pos_neg,
    tmp,
    coord_tensor,
    centers_sorted,
    perm;
    std::string device;
    int dim_fac,depth;
    int *dim_fac_pointer;
    float avg_nr_points;
    n_tree_cuda(torch::Tensor &e, torch::Tensor &d, torch::Tensor &xm, const std::string &cuda_str ):data(d){
        device = cuda_str;
        xmin = xm;
        edge = e;
        dim_fac = pow(2,dim);
        cudaMalloc((void **)&dim_fac_pointer, sizeof(int));
        cudaMemcpy(dim_fac_pointer, &dim_fac, sizeof(int), cudaMemcpyHostToDevice);  //Do delete somewhere
        sorted_index  = torch::zeros({data.size(0)}).toType(torch::kInt32).contiguous().to(device);
        avg_nr_points =  (float)d.size(0);
        multiply_cpu_base = torch::pow(2, torch::arange(dim - 1, -1, -1).toType(torch::kInt32));
        multiply_gpu_base = multiply_cpu_base.toType(torch::kInt32).to(device);
        multiply_cpu= multiply_cpu_base;
        multiply_gpu = multiply_gpu_base;
        coord_tensor = torch::zeros({dim_fac,dim}).toType(torch::kInt32).to(device);
        get_centers<dim><<<8,192>>>(coord_tensor.packed_accessor32<int,2,torch::RestrictPtrTraits>());
        cudaDeviceSynchronize();
        centers_unique = xmin + 0.5 * edge;
        centers_unique = centers_unique.unsqueeze(0);
        centers = xmin + 0.5 * edge;
        centers = centers.unsqueeze(0);
        pos_neg = torch::cat({-torch::ones({1,dim}).toType(dtype<scalar_t>()),torch::ones({1,dim}).toType(dtype<scalar_t>()) },0).to(device);
        depth = 0;
    }

    void expand_center_unique(){
        if (depth==0){
            centers_unique = centers_unique.repeat({2,1})+0.25*edge*pos_neg;
        }else{
            pos_neg = pos_neg.repeat({2,1});
            centers_unique = centers_unique.repeat_interleave(2,0)+0.25*edge*pos_neg;
        }
    }

    void generate_centers_grouped_by(){
        std::vector<torch::Tensor> container = {};
        auto m_accesor = multiply_cpu.accessor<int ,1>();
        for (int j = 0;j<dim;j++){
            tmp = centers_unique.slice(1,j,j+1).repeat_interleave(m_accesor[j]).repeat(m_accesor[dim-1-j]);
            container.push_back(tmp);
        }
        centers_sorted = torch::stack(container,1);

    }
    void natural_center_divide(){
        if (depth==0){
            centers = centers.repeat_interleave(dim_fac,0)+ 0.25 * edge * coord_tensor;
        }else{
            centers = centers.repeat_interleave(dim_fac,0)+ 0.25 * edge * coord_tensor.repeat({centers.size(0),1});
        }
    }
    void divide(){
        expand_center_unique();
        generate_centers_grouped_by();
        natural_center_divide();
        edge = edge*0.5;
        unique_counts_cum = torch::zeros(centers.size(0)+1).toType(torch::kInt32).contiguous().to(device);
        unique_counts= torch::zeros(centers.size(0)).toType(torch::kInt32).contiguous().to(device);
        perm = torch::zeros(centers.size(0)).toType(torch::kInt32).contiguous().to(device);
        dim3 blockSize,gridSize;
        int memory;
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(dim, centers.size(0));
        find_center_perm<scalar_t,dim><<<gridSize,blockSize>>>(centers_sorted.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                centers.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                        perm.packed_accessor32<int,1,torch::RestrictPtrTraits>());
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(dim, data.size(0));
        box_division_cum<scalar_t,dim><<<gridSize,blockSize>>>(
                data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                edge.data_ptr<scalar_t>(),
                centers_sorted.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                multiply_gpu.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                unique_counts_cum.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                perm.packed_accessor32<int,1,torch::RestrictPtrTraits>()
        );
        cudaDeviceSynchronize();
        unique_counts_cum = unique_counts_cum.cumsum(0).toType(torch::kInt32);
        box_division_assign<scalar_t,dim><<<gridSize,blockSize>>>(
                data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                edge.data_ptr<scalar_t>(),
                centers_sorted.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                multiply_gpu.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                unique_counts_cum.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                unique_counts.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                sorted_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                perm.packed_accessor32<int,1,torch::RestrictPtrTraits>()
        );
        cudaDeviceSynchronize();

        box_indices_sorted = unique_counts.nonzero().squeeze();
        centers = centers.index({box_indices_sorted.toType(torch::kLong),torch::indexing::Slice()});
        unique_counts = unique_counts.index({box_indices_sorted.toType(torch::kLong)});
        //replace with inplace sorting and unique...
        avg_nr_points = unique_counts.toType(torch::kFloat32).mean().item<float>();
        multiply_cpu = multiply_cpu*multiply_cpu_base ;
        multiply_gpu = multiply_gpu*multiply_gpu_base;
        depth += 1;

    };

//    void divide_old(){
//        dim3 blockSize,gridSize;
//        int memory;
//        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(dim, data.size(0));
//        unique_counts_cum = torch::zeros(centers.size(0)*dim_fac+1).toType(torch::kInt32).contiguous().to(device);;
//        unique_counts= torch::zeros(centers.size(0)*dim_fac).toType(torch::kInt32).contiguous().to(device);;
//        box_division<scalar_t,dim><<<gridSize,blockSize>>>(
//                data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
//                centers.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
//                multiply.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
//                box_indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
//                unique_counts_cum.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
//                dim_fac_pointer
//
//        );
//        cudaDeviceSynchronize();
//        unique_counts_cum = unique_counts_cum.cumsum(0).toType(torch::kInt32);
//        group_index<<<gridSize,blockSize>>>(
//                box_indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
//                unique_counts_cum.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
//                unique_counts.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
//                sorted_index.packed_accessor32<int,1,torch::RestrictPtrTraits>()
//        );
//        cudaDeviceSynchronize();
//        std::tie(box_indices_sorted,tmp) = torch::_unique(box_indicator,true,false);
//
//        //replace with inplace sorting and unique...
//        avg_nr_points = unique_counts.toType(torch::kFloat32).mean().item<float>();
//        natural_center_divide();
//        centers = centers.index(box_indices_sorted.toType(torch::kLong));
//        edge = edge*0.5;
//        depth += 1;
//    };
    std::tuple<torch::Tensor,torch::Tensor> get_box_sorted_data(){
        return std::make_tuple(unique_counts,sorted_index);
    }
};




template <typename scalar_t, int nd>
void rbf_call(
        torch::Tensor & cuda_X_job,
        torch::Tensor & cuda_Y_job,
        torch::Tensor & cuda_b_job,
        torch::Tensor & output_job,
        scalar_t & ls,
        bool shared = true
        ){

    scalar_t *d_ls;
    cudaMalloc((void **)&d_ls, sizeof(scalar_t));
    cudaMemcpy(d_ls, &ls, sizeof(scalar_t), cudaMemcpyHostToDevice);

    dim3 blockSize,gridSize;
    int memory;
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(nd, cuda_X_job.size(0));

    if(shared){
        rbf_1d_reduce_shared_torch<scalar_t,nd><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            cuda_Y_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            cuda_b_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            output_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            d_ls
                                                                            );
    }else{
        rbf_1d_reduce_simple_torch<scalar_t,nd><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            cuda_Y_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            cuda_b_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            output_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            d_ls
                                                                            );
    }
    cudaDeviceSynchronize();

}


int optimal_blocksize(int &min_box_size){
    std::vector<int> candidates = {0,32,64,96,128,160,192,224,256,352};
    for (int i=1;i<candidates.size();i++){
        if( (min_box_size>=candidates[i-1]) and (min_box_size<=candidates[i])){
            return candidates[i];
        }
    }
    return 352;
}

template <typename scalar_t,int nd>
void call_skip_conv(
        torch::Tensor & cuda_X_job,
        torch::Tensor & cuda_Y_job,
        torch::Tensor & cuda_b_job,
        torch::Tensor & output_job,
        scalar_t & ls,
        torch::Tensor & x_boxes_count,
        torch::Tensor & y_boxes_count,
        torch::Tensor & x_box_idx,
        const std::string & device_gpu,
        torch::Tensor & x_idx_reordering,
        torch::Tensor & y_idx_reordering,
        torch::Tensor & x_boxes_count_cumulative,
        torch::Tensor & y_boxes_count_cumulative,
        torch::Tensor & interactions_x_parsed,
        torch::Tensor & interactions_y,
        bool shared = true
){
    scalar_t *d_ls;
    cudaMalloc((void **)&d_ls, sizeof(scalar_t));
    cudaMemcpy(d_ls, &ls, sizeof(scalar_t), cudaMemcpyHostToDevice);
    dim3 blockSize,gridSize;
    int memory,blkSize;

    if(shared){
        torch::Tensor block_box_indicator,box_block_indicator;
        int min_size=x_boxes_count.min().item<int>();
        blkSize = optimal_blocksize(min_size);
        std::tie(blockSize, gridSize, memory, block_box_indicator, box_block_indicator) = skip_kernel_launch<scalar_t>(nd, blkSize, x_boxes_count, x_box_idx);
        block_box_indicator = block_box_indicator.to(device_gpu);
        box_block_indicator = box_block_indicator.to(device_gpu);

        skip_conv_1d_shared<scalar_t,nd><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                        cuda_Y_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                        cuda_b_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                        output_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                        d_ls,
                                                                        x_boxes_count_cumulative.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                        y_boxes_count_cumulative.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                        block_box_indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                        box_block_indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                        x_idx_reordering.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                        y_idx_reordering.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                        interactions_x_parsed.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                                                        interactions_y.packed_accessor32<int,1,torch::RestrictPtrTraits>()
                                                                     );

    }else{
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(nd, cuda_X_job.size(0));
        x_box_idx = x_box_idx.to(device_gpu);
        torch::Tensor x_boxes_count_cumulative_alt = x_boxes_count.cumsum(0).toType(torch::kInt32).to(device_gpu);
        torch::Tensor y_boxes_count_cumulative_alt = y_boxes_count.cumsum(0).toType(torch::kInt32).to(device_gpu);
        skip_conv_1d<scalar_t,nd><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              cuda_Y_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              cuda_b_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              output_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              d_ls,
                                                                 x_boxes_count_cumulative_alt.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                 y_boxes_count_cumulative_alt.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                              x_box_idx.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                              interactions_x_parsed.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                                              interactions_y.packed_accessor32<int,1,torch::RestrictPtrTraits>()
                                                                      );
    } //fix types...
    cudaDeviceSynchronize();
}

//Could represent indices with [box_id, nr of points], make sure its all concatenated correctly and in order.
//local variable: x_i, box belonging. How to get loading scheme. [nr of points].cumsum().  Iteration schedual...
// calculate box belonging.

template <typename scalar_t,int nd>
void near_field_compute_v2(
                        torch::Tensor & interactions_x_parsed,
                        torch::Tensor & interactions_y,
                        n_tree_cuda<scalar_t,nd> & x_box,
                        n_tree_cuda<scalar_t,nd> & y_box,
                        torch::Tensor & output,
                        torch::Tensor& b,
                        const std::string & device_gpu,
                        scalar_t & ls){

    call_skip_conv<scalar_t,nd>(x_box.data,
                             y_box.data,
                             b,
                             output,
                             ls,
                             x_box.unique_counts,
                                y_box.unique_counts,
                                x_box.box_indices_sorted,
                             device_gpu,
                             x_box.sorted_index,
                                y_box.sorted_index,
                             x_box.unique_counts_cum,
                             y_box.unique_counts_cum,
                             interactions_x_parsed,
                             interactions_y,
                             true);
};
template <typename scalar_t>
torch::Tensor chebyshev_nodes_1D(const int & nodes){
    float PI = atan(1.)*4.;
    torch::Tensor chebyshev_nodes = torch::arange(1, nodes+1).toType(dtype<scalar_t>());
    chebyshev_nodes = cos((chebyshev_nodes*2.-1.)*PI/(2*chebyshev_nodes.size(0)));
    return chebyshev_nodes;
};


template <typename scalar_t,int nd>
void apply_laplace_interpolation_v2(
        n_tree_cuda<scalar_t,nd>& n_tree,
        torch::Tensor &b,
        const std::string & device_gpu,
        torch::Tensor & nodes,
        torch::Tensor & laplace_indices,
        const bool & transpose,
        torch::Tensor & output
        ){
    torch::Tensor boxes_count,idx_reordering,b_data;
    torch::Tensor & data = n_tree.data;
    torch::Tensor & centers = n_tree.centers;
    torch::Tensor & edge = n_tree.edge;

    std::tie(boxes_count, idx_reordering)=n_tree.get_box_sorted_data();
    dim3 blockSize,gridSize;
    int memory,blkSize;
    torch::Tensor indicator,box_block;
    int min_size=boxes_count.min().item<int>();
    blkSize = optimal_blocksize(min_size);
    std::tie(blockSize,gridSize,memory,indicator,box_block) = skip_kernel_launch<scalar_t>(nd,blkSize,boxes_count,n_tree.box_indices_sorted);
    memory = memory+nodes.size(0)*sizeof(scalar_t)+sizeof(int)+32*sizeof(scalar_t);
    torch::Tensor boxes_count_cumulative = n_tree.unique_counts_cum;
    if (transpose){

        laplace_shared<scalar_t,nd><<<gridSize,blockSize,memory>>>(
                data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                b.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                nodes.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                laplace_indices.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                box_block.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                boxes_count_cumulative.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                centers.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                edge.data_ptr<scalar_t>(),
                idx_reordering.packed_accessor32<int,1,torch::RestrictPtrTraits>()
        );
        cudaDeviceSynchronize();

    }else{
        laplace_shared_transpose<scalar_t,nd><<<gridSize,blockSize,memory>>>(
                data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                b.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                nodes.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                laplace_indices.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                box_block.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                boxes_count_cumulative.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                centers.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                edge.data_ptr<scalar_t>(),
                idx_reordering.packed_accessor32<int,1,torch::RestrictPtrTraits>()
        );
        cudaDeviceSynchronize();
    }
}



template <typename scalar_t,int nd>
torch::Tensor setup_skip_conv(torch::Tensor &cheb_data,
                              torch::Tensor &b_data,
                              torch::Tensor & centers_X,
                              torch::Tensor & centers_Y,
                              torch::Tensor & unique_sorted_boxes_idx,
                              scalar_t & ls,
                              const std::string & device_gpu,
                              torch::Tensor & interactions_x_parsed,
                              torch::Tensor & interactions_y
){
    scalar_t *d_ls;
//    int *d_cheb_data_size;
    cudaMalloc((void **)&d_ls, sizeof(scalar_t));
    cudaMemcpy(d_ls, &ls, sizeof(scalar_t), cudaMemcpyHostToDevice);
    torch::Tensor indicator,box_block,output;
    int cheb_data_size=cheb_data.size(0);
    int blkSize = optimal_blocksize(cheb_data_size);
//    torch::Tensor boxes_count = min_size*torch::ones({unique_sorted_boxes_idx.size(0)+1}).toType(torch::kInt32);
    torch::Tensor boxes_count = cheb_data_size * torch::ones(unique_sorted_boxes_idx.size(0)).toType(torch::kInt32).to(device_gpu);
    dim3 blockSize,gridSize;
    int memory;
    unique_sorted_boxes_idx = unique_sorted_boxes_idx.toType(torch::kInt32);
    std::tie(blockSize,gridSize,memory,indicator,box_block) = skip_kernel_launch<scalar_t>(nd,blkSize,boxes_count,unique_sorted_boxes_idx);
    output = torch::zeros_like(b_data);
    skip_conv_far_boxes_opt<scalar_t,nd><<<gridSize,blockSize,memory>>>(
            cheb_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            b_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            d_ls,
            centers_X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            centers_Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            box_block.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            interactions_x_parsed.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            interactions_y.packed_accessor32<int,1,torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();
    return output;
}

//Always pass interactions..., if there aint any then ok etc...
template <typename scalar_t,int nd>
void far_field_compute_v2(
                       torch::Tensor & interactions_x_parsed,
                       torch::Tensor & interactions_y,
                       n_tree_cuda<scalar_t,nd> & x_box,
                       n_tree_cuda<scalar_t,nd> & y_box,
                       torch::Tensor & output,
                       torch::Tensor &b,
                       const std::string & device_gpu,
                       scalar_t & ls,
                          torch::Tensor & chebnodes_1D,
                          torch::Tensor & laplace_combinations,
                          torch::Tensor & cheb_data_X
){
    torch::Tensor low_rank_y;
    low_rank_y = torch::zeros({cheb_data_X.size(0)*x_box.box_indices_sorted.size(0),b.size(1)}).to(device_gpu);
    apply_laplace_interpolation_v2<scalar_t,nd>(y_box,
                                            b,
                                            device_gpu,
                                            chebnodes_1D,
                                            laplace_combinations,
                                            true,
                                            low_rank_y); //no problems here!

    //Consider limiting number of boxes!!!
    low_rank_y = setup_skip_conv<scalar_t,nd>( //error happens here
            cheb_data_X,
            low_rank_y,
            x_box.centers,
            y_box.centers,
            x_box.box_indices_sorted,
            ls,
            device_gpu,
            interactions_x_parsed,
            interactions_y
            );
    //std::cout<<low_rank_y.slice(0,0,5)<<std::endl;
    apply_laplace_interpolation_v2<scalar_t,nd>(x_box,
                                            low_rank_y,
                                            device_gpu,
                                            chebnodes_1D,
                                            laplace_combinations,
                                            false,
                                            output);
};
torch::Tensor get_new_interactions(torch::Tensor & old_near_interactions, int & p,const std::string & gpu_device){//fix tmrw
    int n = old_near_interactions.size(0);
    torch::Tensor arr = torch::arange(p).toType(torch::kInt32).to(gpu_device);
    torch::Tensor new_interactions_vec = torch::stack({arr.repeat_interleave(p).repeat(n),arr.repeat(p*n)},1)+p*old_near_interactions.repeat_interleave(p*p,0);
    return new_interactions_vec;
}


template <int nd>
torch::Tensor process_interactions(torch::Tensor & interactions,int x_boxes,const std::string & gpu_device){
    torch::Tensor box_indices,tmp,counts,count_cumsum,results;
    std::tie(box_indices,tmp,counts) = torch::unique_consecutive(interactions,false,true);
    count_cumsum = torch::stack({box_indices.toType(torch::kInt32),counts.cumsum(0).toType(torch::kInt32)},1);  //64+1 vec
    results = -torch::ones({x_boxes,2}).toType(torch::kInt32).to(gpu_device);
    dim3 blockSize,gridSize;
    int memory;
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(nd, count_cumsum.size(0));
    parse_x_boxes<<<gridSize,blockSize>>>(
            count_cumsum.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            results.packed_accessor32<int,2,torch::RestrictPtrTraits>()
            );
    cudaDeviceSynchronize();
    return results;

}

template<typename scalar_t,int d>
std::tuple<torch::Tensor,torch::Tensor> parse_cheb_data(
        torch::Tensor & cheb_nodes,
        const std::string & gpu_device
        ){
    int n = (int) pow(cheb_nodes.size(0),d);
    torch::Tensor cheb_idx = torch::zeros({n,d}).toType(torch::kInt32).to(gpu_device);
    torch::Tensor cheb_data = torch::zeros({n,d}).toType(dtype<scalar_t>()).to(gpu_device);
    dim3 block,grid;
    int shared;
    std::tie(block,grid,shared) =  get_kernel_launch_params<scalar_t>(d,n);
    get_cheb_idx_data<scalar_t,d><<<grid,block,shared>>>(
            cheb_nodes.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            cheb_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            cheb_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>()
            );
    cudaDeviceSynchronize();
    return std::make_tuple(cheb_idx,cheb_data);

}

std::tuple<torch::Tensor,torch::Tensor> unbind_sort(torch::Tensor & interactions){
    interactions = interactions.index({torch::argsort(interactions.slice(1,0,1).squeeze()),torch::indexing::Slice()});
    std::vector<torch::Tensor> tmp = interactions.unbind(1);
    return std::make_tuple(tmp[0],tmp[1]);
}

template <typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor> separate_interactions(
        torch::Tensor & interactions,
        torch::Tensor & centers_X,
        torch::Tensor & centers_Y,
        torch::Tensor & edge,
        const std::string & gpu_device
){
    dim3 blockSize,gridSize;
    int mem,n;
    n = interactions.size(0);
    std::tie(blockSize,gridSize,mem)=get_kernel_launch_params<scalar_t>(nd,n);
    torch::Tensor far_field_mask = torch::zeros({n}).toType(torch::kBool).to(gpu_device);
    boolean_separate_interactions<scalar_t,nd><<<gridSize,blockSize>>>(
            centers_X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            centers_Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            interactions.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            edge.data_ptr<scalar_t>(),
            far_field_mask.packed_accessor32<bool,1,torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();
    return std::make_tuple(interactions.index({far_field_mask}),interactions.index({far_field_mask.logical_not()}));
}
template <typename scalar_t, int nd>
torch::Tensor far_field_run(
                            n_tree_cuda<scalar_t,nd> & ntree_X,
                            n_tree_cuda<scalar_t,nd> & ntree_Y,
                            torch::Tensor & near_field,
                            torch::Tensor & cheb_data_X,
                            torch::Tensor & output,
                            torch::Tensor & b,
                            torch::Tensor & chebnodes_1D,
                            torch::Tensor & laplace_combinations,
                            scalar_t & ls,
                            const std::string & gpu_device
){
    torch::Tensor interactions,far_field,interactions_x,interactions_y,interactions_x_parsed;
    interactions = get_new_interactions(near_field,ntree_X.dim_fac,gpu_device); //Doesn't work for new setup since the division is changed...
    std::tie(far_field,near_field) =
            separate_interactions<scalar_t,nd>(
                    interactions,
                    ntree_X.centers,
                    ntree_Y.centers,
                    ntree_X.edge,
                    gpu_device);
    if (far_field.numel()>0) {
//        std::cout<<"near field: "<<near_field.size(0)<<std::endl;
//        std::cout<<"far field: "<<far_field.size(0)<<std::endl;
        torch::Tensor cheb_data = cheb_data_X*ntree_X.edge/2.+ntree_X.edge/2.;
        std::tie(interactions_x,interactions_y) = unbind_sort(far_field);
        interactions_x_parsed = process_interactions<nd>(interactions_x,ntree_X.centers.size(0),gpu_device);
        far_field_compute_v2<scalar_t,nd>( //Very many far field interactions quite fast...
                interactions_x_parsed,
                interactions_y,
                ntree_X,
                ntree_Y,
                output,
                b,
                gpu_device,
                ls,
                chebnodes_1D,
                laplace_combinations,
                cheb_data
        ); //far field compute
    }
    return near_field;
}

template <typename scalar_t, int nd>
void near_field_run(
        n_tree_cuda<scalar_t,nd> & ntree_X,
        n_tree_cuda<scalar_t,nd> & ntree_Y,
        torch::Tensor & near_field,
        torch::Tensor & output,
        torch::Tensor & b,
        scalar_t & ls,
        const std::string & gpu_device
){
    torch::Tensor interactions_x,interactions_y,interactions_x_parsed;
    std::tie(interactions_x,interactions_y) = unbind_sort(near_field);
    interactions_x_parsed = process_interactions<nd>(interactions_x,ntree_X.centers.size(0),gpu_device);
    near_field_compute_v2<scalar_t,nd>(interactions_x_parsed,interactions_y,ntree_X, ntree_Y, output, b, gpu_device,ls); //Make sure this thing works first!
}

template <typename scalar_t, int nd>
torch::Tensor FFM_XY(
        torch::Tensor &X_data,
        torch::Tensor &Y_data,
        torch::Tensor &b,
        const std::string & gpu_device,
        scalar_t & ls,
        int &laplace_n,
        float &min_points,
        scalar_t & lambda

) {
    torch::Tensor output = torch::zeros({X_data.size(0),b.size(1)}).to(gpu_device); //initialize empty output
    torch::Tensor edge,xmin,ymin,interactions,far_field,near_field,interactions_x,interactions_y,interactions_x_parsed,cheb_data_X,laplace_combinations; //these are needed to figure out which interactions are near/far field
    near_field = torch::zeros({1,2}).toType(torch::kInt32).to(gpu_device);
    torch::Tensor chebnodes_1D = chebyshev_nodes_1D<scalar_t>(laplace_n).to(gpu_device); //get chebyshev nodes, laplace_nodes is fixed now, should probably be a variable
    std::tie(laplace_combinations,cheb_data_X)=parse_cheb_data<scalar_t,nd>(chebnodes_1D,gpu_device);
    std::tie(edge,xmin,ymin) = calculate_edge<scalar_t,nd>(X_data,Y_data,gpu_device); //actually calculate them
    n_tree_cuda<scalar_t,nd> ntree_X = n_tree_cuda<scalar_t,nd>(edge,X_data,xmin,gpu_device);
    n_tree_cuda<scalar_t,nd> ntree_Y = n_tree_cuda<scalar_t,nd>(edge,Y_data,ymin,gpu_device);

    while (near_field.numel()>0 and ntree_X.avg_nr_points > min_points and ntree_Y.avg_nr_points > min_points){
        ntree_X.divide();//needs to be fixed... Should get 451 errors, OK. Memory issue is consistent
        ntree_Y.divide();//divide ALL boxes recursively once
        near_field = far_field_run<scalar_t,nd>(ntree_X,ntree_Y,near_field,cheb_data_X,output,b,chebnodes_1D,laplace_combinations,ls,gpu_device);

        }

    if (near_field.numel()>0){
        near_field_run<scalar_t,nd>(ntree_X,ntree_Y,near_field,output,b,ls,gpu_device);
    }
    return output;
//    +lambda*b;
}

template <typename scalar_t, int nd>
torch::Tensor FFM_X(
        torch::Tensor &X_data,
        torch::Tensor &b,
        const std::string & gpu_device,
        scalar_t & ls,
        int &laplace_n,
        float &min_points,
        scalar_t & lambda

) {
    torch::Tensor output = torch::zeros({X_data.size(0),b.size(1)}).to(gpu_device); //initialize empty output
    torch::Tensor output_ref =torch::zeros({X_data.size(0),b.size(1)}).to(gpu_device); //initialize empty output
    torch::Tensor edge,
    xmin,
    near_field,
    cheb_data_X,
    laplace_combinations;//these are needed to figure out which interactions are near/far field
    near_field = torch::zeros({1,2}).toType(torch::kInt32).to(gpu_device);
//    near_field_ref = torch::zeros({1,2}).toType(torch::kInt32).to(gpu_device);
    torch::Tensor chebnodes_1D = chebyshev_nodes_1D<scalar_t>(laplace_n).to(gpu_device); //get chebyshev nodes, laplace_nodes is fixed now, should probably be a variable
    std::tie(laplace_combinations,cheb_data_X)=parse_cheb_data<scalar_t,nd>(chebnodes_1D,gpu_device);
    std::tie(edge,xmin) = calculate_edge_X<scalar_t,nd>(X_data,gpu_device); //actually calculate them
    n_tree_cuda<scalar_t,nd> ntree_X = n_tree_cuda<scalar_t,nd>(edge,X_data,xmin,gpu_device);
//    n_tree_cuda<scalar_t,nd> ntree_X_ref =  n_tree_cuda<scalar_t,nd>(edge,X_data,xmin,gpu_device);
    while (near_field.numel()>0 and ntree_X.avg_nr_points > min_points){
        ntree_X.divide();//needs to be fixed... Should get 451 errors, OK. Memory issue is consistent
        near_field = far_field_run<scalar_t,nd>(ntree_X,ntree_X,near_field,cheb_data_X,output,b,chebnodes_1D,laplace_combinations,ls,gpu_device);
        //std::cout<<output.slice(0,0,5)<<std::endl;
    }
    if (near_field.numel()>0){
        near_field_run<scalar_t,nd>(ntree_X,ntree_X,near_field,output,b,ls,gpu_device);
    }

//    while (near_field_ref.numel()>0 and ntree_X_ref.avg_nr_points > min_points){
//        ntree_X_ref.divide_old();//needs to be fixed... Should get 451 errors, OK. Memory issue is consistent
//        near_field_ref = far_field_run<scalar_t,nd>(ntree_X_ref,ntree_X_ref,near_field_ref,cheb_data_X,output_ref,b,chebnodes_1D,laplace_combinations,ls,gpu_device);
//        //std::cout<<output_ref.slice(0,0,5)<<std::endl;
//    }
//    if (near_field_ref.numel()>0){
//        near_field_run<scalar_t,nd>(ntree_X_ref,ntree_X_ref,near_field_ref,output_ref,b,ls,gpu_device);
//    }
    return output;
    //+lambda*b;
}
template <typename scalar_t, int nd>
struct FFM_object{
    torch::Tensor & X_data;
    torch::Tensor & Y_data;
    scalar_t & ls;
    scalar_t &lambda;
    const std::string & gpu_device;
    int & laplace_n;
    float & min_points;

    FFM_object( //constructor
            torch::Tensor & X_data,
    torch::Tensor & Y_data,
    scalar_t & ls,
    scalar_t &lambda,
    const std::string & gpu_device,
    int & laplace_n,
    float & min_points): X_data(X_data), Y_data(Y_data),ls(ls),lambda(lambda),gpu_device(gpu_device), laplace_n(laplace_n),min_points(min_points){
    };
    virtual torch::Tensor operator* (torch::Tensor & b){
        if (X_data.data_ptr()==Y_data.data_ptr()){
            return FFM_X<scalar_t,nd>(
                    X_data,
                    b,
                    gpu_device,
                    ls,
                    laplace_n,
                    min_points,
                    lambda
            );
        }else{
            return FFM_XY<scalar_t,nd>(
                    X_data,
                    Y_data,
                    b,
                    gpu_device,
                    ls,
                    laplace_n,
                    min_points,
                    lambda
            );
        }


    };
};
template <typename scalar_t, int nd>
struct exact_MV : FFM_object<scalar_t,nd>{
    exact_MV(torch::Tensor & X_data,
                         torch::Tensor & Y_data,
                         scalar_t & ls,
                         scalar_t &lambda,
                         const std::string & gpu_device,
                         int &laplace_n,
                         float &min_points)
                         : FFM_object<scalar_t,nd>(X_data, Y_data, ls, lambda, gpu_device,laplace_n,min_points){};
    torch::Tensor operator* (torch::Tensor & b) override{
        torch::Tensor output = torch::zeros({FFM_object<scalar_t,nd>::X_data.size(0), b.size(1)}).to(FFM_object<scalar_t,nd>::gpu_device);
        rbf_call<scalar_t,nd>(
                FFM_object<scalar_t,nd>::X_data,
                FFM_object<scalar_t,nd>::Y_data,
                b,
                output,
                FFM_object<scalar_t,nd>::ls,
                true
        );
        return output;
    };
};

template<typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor> CG(FFM_object<scalar_t,nd> & MV, torch::Tensor &b, float & tol, int & max_its, bool tridiag){
    int h = b.size(0);
    scalar_t delta = tol*(float)h;
    auto a = torch::zeros_like(b);
    torch::Tensor r = b;
    torch::Tensor nr2 = (torch::pow(r,2)).sum();
    if (nr2.item<scalar_t>()<delta){
        return std::make_tuple(a,torch::zeros(1));
    }
    torch::Tensor p = r;
    std::vector<torch::Tensor> lanczos_vals_list = {};
    torch::Tensor Mp,alp,beta,alp_old,beta_old,nr2new,lanczos_append,tridiag_output,tridiag_cat,z,o;
    z = torch::zeros(1);
    o = torch::ones(1);
    for (int i=0;i<max_its;i++){
        Mp = MV*p;
        alp = nr2/(p*Mp).sum();
        a = a + alp*p;
        r = r - alp*Mp;
        nr2new = (torch::pow(r,2)).sum();
        if (nr2new.item<scalar_t>()<delta){
            break;
        }
        beta = nr2new/nr2;
        p = r + beta*p;
        nr2=nr2new;
        if (tridiag){
            if (i==0){
                lanczos_append = torch::stack({z.squeeze(),(o/alp).squeeze(),(beta.sqrt()/alp).squeeze()}).unsqueeze(0);
            }else{
                lanczos_append = torch::stack({(beta_old.sqrt()/alp_old).squeeze(),
                                               (o/alp+beta_old/alp_old).squeeze(),(beta.sqrt()/alp).squeeze()}).unsqueeze(0);
            }
            lanczos_vals_list.push_back(lanczos_append);
            alp_old = alp;
            beta_old = beta;
        }

    }
    if (tridiag){
        tridiag_cat = torch::cat(lanczos_vals_list);
        tridiag_output = torch::diagflat(tridiag_cat.slice(1,1,2)) +
                torch::diagflat(tridiag_cat.slice(1,0,1).slice(0,1),-1) +
                torch::diagflat(tridiag_cat.slice(1,2,3).slice(0,0,-1),1);
        return std::make_tuple(a,tridiag_output);
    }else{
        return std::make_tuple(a,torch::zeros(1));
    }
}

torch::Tensor calculate_one_lanczos_triag(torch::Tensor & tridiag_mat){
    torch::Tensor V,P,U;
    std::tie(P,U) = torch::eig(tridiag_mat,true);
    P = P.slice(1,0,1);
    V = U.slice(0,0);
    return (V.pow_(2)*P.log_()).sum();
}
template <typename scalar_t,int nd>
std::tuple<torch::Tensor,torch::Tensor> trace_and_log_det_calc(FFM_object<scalar_t,nd> &MV, FFM_object<scalar_t,nd> &MV_grad, int& T, int &max_its, float &tol){
    std::vector<torch::Tensor> log_det_approx = {};
    std::vector<torch::Tensor> trace_approx = {};
    torch::Tensor z_sol,z,tridiag_z,log_det_cat,trace_cat;
    for (int i=0;i<T;i++){
        z = torch::randn({MV.X_data.size(0),1});
        std::tie(z_sol,tridiag_z) = CG<scalar_t>(MV,z,tol,max_its,true);
        log_det_approx.push_back(calculate_one_lanczos_triag(tridiag_z));
        trace_approx.push_back( torch::sum(z_sol*(MV_grad*z)));
    }
    log_det_cat = torch::stack(log_det_approx);
    trace_cat = torch::stack(trace_approx);
    return std::make_tuple(log_det_cat.mean(),trace_cat.mean());
}

template<typename scalar_t, int nd>
torch::Tensor ls_grad_calculate(FFM_object<scalar_t,nd> &MV_grad, torch::Tensor & b_sol, torch::Tensor &trace_est){
    return trace_est + torch::sum(b_sol*(MV_grad*b_sol));
}

torch::Tensor GP_loss(torch::Tensor &log_det,torch::Tensor &b_sol,torch::Tensor &b){
    return log_det - torch::sum(b*b_sol);
}
template<typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> calculate_loss_and_grad(FFM_object<scalar_t,nd> &MV,
                                                                              FFM_object<scalar_t,nd> &MV_grad,
                                                                              torch::Tensor & b,
                                                                              int &T,
                                                                              int &max_its,
                                                                              float &tol){
    torch::Tensor b_sol,log_det,trace_est,grad,loss,_;
    std::tie(b_sol,_) = CG(MV,b,tol,max_its,false);
    std::tie(log_det,trace_est) = trace_and_log_det_calc<scalar_t>(MV,MV_grad,T,max_its,tol);
    grad = ls_grad_calculate<scalar_t>(MV_grad,b_sol,trace_est);
    loss = GP_loss(log_det,b_sol,b);
    return std::make_tuple(loss,grad,b_sol);
}

template <int nd>
void benchmark_1(int laplace_n,int n,float min_points, int threshold){
    const std::string device_cuda = "cuda:0"; //officially retarded
    const std::string device_cpu = "cpu";
//    torch::manual_seed(0);

//    torch::Tensor X_train = read_csv<float>("X_train_PCA.csv",11619,3); something wrong with data probably...
//    torch::Tensor b_train = read_csv<float>("Y_train.csv",11619,1); something wrong with data probably...
    torch::Tensor X_train = torch::rand({n,nd}).to(device_cuda); //Something fishy going on here, probably the boxes stuff...
    torch::Tensor b_train = torch::randn({n,1}).to(device_cuda);
    float ls = 1.0; //lengthscale
    float lambda = 1e-1; // ridge parameter
    torch::Tensor res,res_ref;
    FFM_object<float,nd> ffm_obj = FFM_object<float,nd>(X_train, X_train, ls, lambda, device_cuda,laplace_n,min_points); //FMM object
//    FFM_object<float> ffm_obj_grad = FFM_object<float>(X,X,ls,op_grad,lambda,device_cuda);
//    exact_MV<float> ffm_obj_grad_exact = exact_MV<float>(X,X,ls,op_grad,lambda,device_cuda);

    if (n>threshold){
        torch::Tensor subsampled_X = X_train.slice(0,0,threshold);
        exact_MV<float,nd> exact_ref = exact_MV<float,nd>(subsampled_X, X_train, ls,  lambda, device_cuda,laplace_n,min_points); //Exact method reference
        auto start = std::chrono::high_resolution_clock::now();
        res_ref = exact_ref *b_train;
        auto end = std::chrono::high_resolution_clock::now();
        res = ffm_obj * b_train; //Fast math creates problems... fast math does a lot!!!
        auto end_2 = std::chrono::high_resolution_clock::now();
        auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
        auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2-end);
        torch::Tensor res_compare = res.slice(0,0,threshold);
        std::cout<<"------------- "<<"laplace nodes: "<<laplace_n<<" n: "<<n<<" min_points: "<< min_points <<" -------------"<<std::endl;
        std::cout<<"Full matmul time (ms): "<<duration_1.count()<<std::endl;
        std::cout<<"FFM time (ms): "<<duration_2.count()<<std::endl;
        std::cout<<"Relative error: "<<((res_ref-res_compare)/res_ref).abs_().mean()<<std::endl;
    }else{
        exact_MV<float,nd> exact_ref = exact_MV<float,nd>(X_train, X_train, ls,  lambda, device_cuda,laplace_n,min_points); //Exact method reference
        auto start = std::chrono::high_resolution_clock::now();
        res_ref = exact_ref * b_train;
        auto end = std::chrono::high_resolution_clock::now();
        res = ffm_obj * b_train; //Fast math creates problems... fast math does a lot!!!
        auto end_2 = std::chrono::high_resolution_clock::now();
        auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
        auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2-end);
        std::cout<<"------------- "<<"laplace nodes: "<<laplace_n<<" n: "<<n<<" min_points: "<< min_points <<" -------------"<<std::endl;
        std::cout<<"Full matmul time (ms): "<<duration_1.count()<<std::endl;
        std::cout<<"FFM time (ms): "<<duration_2.count()<<std::endl;
        std::cout<<"Relative error: "<<((res_ref-res)/res_ref).abs_().mean()<<std::endl;
    }





}