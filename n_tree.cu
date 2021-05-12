//
// Created by rhu on 2020-03-28.
//
#pragma once
#include <cmath>
#include <vector>
#include "1_d_conv.cu"

//template<typename T>

void print_tensor(torch::Tensor & tensor){
    std::cout<<tensor<<std::endl;
}

template<typename T>
at::ScalarType dtype() { return at::typeMetaToScalarType(caffe2::TypeMeta::Make<T>()); }


template<typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> calculate_edge(const torch::Tensor &X,const torch::Tensor &Y,const std::string & gpu_device){
    torch::Tensor Xmin,Ymin,Xmax,Ymax,tmp,x_edge,y_edge;
    std::tie(Xmin,tmp) = X.min(0);
    std::tie(Xmax,tmp) = X.max(0);
    std::tie(Ymin,tmp) = Y.min(0);
    std::tie(Ymax,tmp) = Y.max(0);
    x_edge=(Xmax - (Xmin-Xmin.abs()*0.01)).max();
    y_edge=(Ymax - (Ymin-Ymin.abs()*0.01)).max();
    torch::Tensor edge = torch::stack({x_edge,y_edge}).max();
    return std::make_tuple(edge*1.01,(Xmin-Xmin.abs()*0.01),(Ymin-Ymin.abs()*0.01),Xmax,Ymax,x_edge*1.01,y_edge*1.01);
};


template<typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> calculate_edge_X(const torch::Tensor &X, const std::string & gpu_device){
    torch::Tensor Xmin,Xmax,tmp;
    std::tie(Xmin,tmp) = X.min(0);
    std::tie(Xmax,tmp) = X.max(0);
    torch::Tensor edge = (Xmax - (Xmin-Xmin.abs()*0.01)).max();
    return std::make_tuple(edge*1.01,(Xmin-Xmin.abs()*0.01),Xmax);
};


template <typename scalar_t,int dim>
struct n_tree_cuda{
    torch::Tensor &data;
    torch::Tensor edge,
    box_max_var,
    edge_og,
    xmin,
    xmax,
    side_base,
    side,
    sorted_index,
    unique_counts,
    multiply_gpu_base,
    multiply_gpu,
    box_indices_sorted,
    box_indices_sorted_reindexed,
    centers,
    empty_box_indices_current,
    empty_box_indices,
    non_empty_mask,
    box_idxs,
    unique_counts_cum,
    unique_counts_cum_reindexed,
    coord_tensor,
    old_new_map,
    perm,
    arrange_empty,
    tmp_1,
    tmp_2;
    std::string device;
    int dim_fac,depth;
    float avg_nr_points;
    n_tree_cuda(const torch::Tensor& e, torch::Tensor &d, torch::Tensor &xm,torch::Tensor &xma, const std::string &cuda_str ):data(d){
        device = cuda_str;
        xmin = xm;
        xmax = xma;
        edge = e;
        edge_og = e;
        dim_fac = pow(2,dim);
        arrange_empty  = torch::arange({dim_fac}).toType(torch::kInt32).to(device);
        sorted_index  = torch::arange({data.size(0)}).toType(torch::kInt32).to(device);
        avg_nr_points =  (float)d.size(0);
        multiply_gpu_base = torch::pow(2, torch::arange(dim - 1, -1, -1).toType(torch::kInt32)).to(device);
        multiply_gpu = multiply_gpu_base;
        coord_tensor = torch::zeros({dim_fac,dim}).toType(torch::kInt32).to(device);
        get_centers<dim><<<8,192>>>(coord_tensor.packed_accessor64<int,2,torch::RestrictPtrTraits>());
        cudaDeviceSynchronize();
        centers = xmin + 0.5 * edge;
        centers = centers.unsqueeze(0);
        depth = 0;
        side_base = torch::tensor(2.0).toType(dtype<scalar_t>()).to(device);

        /*
         * 0 init
         * */
        box_idxs = torch::arange(centers.size(0)).toType(torch::kInt32).to(device);
        unique_counts = torch::tensor({(int)data.size(0)}).toType(torch::kInt32).to(device);
        unique_counts_cum  = torch::tensor({(int)0,(int)data.size(0)}).toType(torch::kInt32).to(device);
        unique_counts_cum_reindexed = unique_counts_cum;
        non_empty_mask = unique_counts != 0;
        box_indices_sorted = box_idxs;
        empty_box_indices_current = box_idxs.index({torch::logical_not(non_empty_mask)});
        empty_box_indices = empty_box_indices_current;
        avg_nr_points = unique_counts.toType(torch::kFloat32).max().item<float>();
        box_indices_sorted_reindexed = box_indices_sorted;
        old_new_map = box_idxs;

    }
    void natural_center_divide(){
        if (depth==0){
            centers = centers.repeat_interleave(dim_fac,0)+ 0.25 * edge * coord_tensor;
        }else{
            centers = centers.repeat_interleave(dim_fac,0)+ 0.25 * edge * coord_tensor.repeat({centers.size(0),1});
        }
    }
    void divide(){
        natural_center_divide();
        edge = edge*0.5;
        box_max_var = edge*edge/12;
        depth += 1;
        side = side_base.pow(depth);
        box_idxs = torch::arange(centers.size(0)).toType(torch::kInt32).to(device);
        unique_counts_cum = torch::zeros(centers.size(0)+1).toType(torch::kInt32).to(device);//matrix -> existing boxes * 2^dim bounded over nr or points... betting on boxing dissapears...
        unique_counts= torch::zeros(centers.size(0)).toType(torch::kInt32).to(device);
        perm = torch::zeros(pow(dim_fac,depth)).toType(torch::kInt32).to(device);
        dim3 blockSize,gridSize;
        int memory;
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(dim, centers.size(0));
        center_perm<scalar_t,dim><<<gridSize,blockSize>>>(
                        centers.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                        xmin.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
                        multiply_gpu.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                        side.data_ptr<scalar_t>(),
                        edge_og.data_ptr<scalar_t>(),
                        perm.packed_accessor64<int,1,torch::RestrictPtrTraits>()
                                ); //Apply same hack but to centers to get perm

        cudaDeviceSynchronize();
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(dim, data.size(0));
        box_division_cum<scalar_t,dim><<<gridSize,blockSize>>>(
                data.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                xmin.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
                multiply_gpu.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                side.data_ptr<scalar_t>(),
                edge_og.data_ptr<scalar_t>(),
                unique_counts_cum.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                perm.packed_accessor64<int,1,torch::RestrictPtrTraits>()
        );
        cudaDeviceSynchronize();
        unique_counts_cum = unique_counts_cum.cumsum(0).toType(torch::kInt32);
        box_division_assign<scalar_t,dim><<<gridSize,blockSize>>>(
                data.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                xmin.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
                multiply_gpu.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                side.data_ptr<scalar_t>(),
                edge_og.data_ptr<scalar_t>(),
                unique_counts_cum.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                perm.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                unique_counts.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                sorted_index.packed_accessor64<int,1,torch::RestrictPtrTraits>()

        );
        cudaDeviceSynchronize();
        old_new_map = box_idxs;
        non_empty_mask = unique_counts != 0;
        box_indices_sorted = box_idxs.index({non_empty_mask});
        unique_counts = unique_counts.index({non_empty_mask});
        centers = centers.index({non_empty_mask});
        empty_box_indices = box_idxs.index({torch::logical_not(non_empty_mask)});
//        empty_box_indices = arrange_empty.repeat(empty_box_indices.size(0))+dim_fac*empty_box_indices.repeat_interleave(dim_fac,0);
//        empty_box_indices = torch::cat({empty_box_indices,empty_box_indices_current},0);
        std::tie(empty_box_indices,tmp_1) = empty_box_indices.sort(0);
        avg_nr_points = unique_counts.toType(torch::kFloat32).max().item<float>();
        multiply_gpu = multiply_gpu*multiply_gpu_base;
        box_indices_sorted_reindexed = torch::arange(centers.size(0)).toType(torch::kInt32).to(device);
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(1, old_new_map.size(0));
        transpose_to_existing_only_tree<<<gridSize,blockSize>>>(
                old_new_map.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                empty_box_indices.packed_accessor64<int,1,torch::RestrictPtrTraits>()
        );
        std::tie(unique_counts_cum_reindexed,tmp_1,tmp_2) = torch::unique_consecutive(unique_counts_cum);

//        std::cout<<"size of removed indices: "<<empty_box_indices.size(0)<<std::endl;

    };

};

template <typename scalar_t>
scalar_t * allocate_scalar_to_cuda(scalar_t & ls){
    scalar_t *d_ls;
    cudaMalloc((void **)&d_ls, sizeof(scalar_t));
    cudaMemcpy(d_ls, &ls, sizeof(scalar_t), cudaMemcpyHostToDevice);
    return d_ls;
}


template <typename scalar_t, int nd>
torch::Tensor rbf_call(
        torch::Tensor & cuda_X_job,
        torch::Tensor & cuda_Y_job,
        torch::Tensor & cuda_b_job,
        scalar_t & ls,
        bool shared = true
        ){


    torch::Tensor output_job = torch::zeros({cuda_X_job.size(0), cuda_b_job.size(1)}).toType(dtype<scalar_t>()).to(cuda_X_job.device());
    scalar_t *d_ls;
    cudaMalloc((void **)&d_ls, sizeof(scalar_t));
    cudaMemcpy(d_ls, &ls, sizeof(scalar_t), cudaMemcpyHostToDevice);

    dim3 blockSize,gridSize;
    int memory;
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(nd, cuda_X_job.size(0));

    if(shared){
        rbf_1d_reduce_shared_torch<scalar_t,nd><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            cuda_Y_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            cuda_b_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            output_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            d_ls
                                                                            );
    }else{
        rbf_1d_reduce_simple_torch<scalar_t,nd><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            cuda_Y_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            cuda_b_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            output_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            d_ls
                                                                            );
    }
    cudaDeviceSynchronize();
    return output_job;
}


int optimal_blocksize(int &min_box_size){
    std::vector<int> candidates = {0,32,64,96,128,160,192};
    for (int i=1;i<candidates.size();i++){
        if( (min_box_size>=candidates[i-1]) and (min_box_size<=candidates[i])){
            return candidates[i];
        }
    }
    return 192;
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
        skip_conv_1d_shared<scalar_t,nd><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                        cuda_Y_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                        cuda_b_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                        output_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                        d_ls,
                                                                        x_boxes_count_cumulative.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                        y_boxes_count_cumulative.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                        block_box_indicator.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                        box_block_indicator.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                        x_idx_reordering.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                        y_idx_reordering.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                        interactions_x_parsed.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
                                                                        interactions_y.packed_accessor64<int,1,torch::RestrictPtrTraits>()
                                                                     );

    }else{
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(nd, cuda_X_job.size(0));
        torch::Tensor x_boxes_count_cumulative_alt = x_boxes_count.cumsum(0).toType(torch::kInt32).to(device_gpu);
        torch::Tensor y_boxes_count_cumulative_alt = y_boxes_count.cumsum(0).toType(torch::kInt32).to(device_gpu);
        skip_conv_1d<scalar_t,nd><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              cuda_Y_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              cuda_b_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              output_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              d_ls,
                                                                 x_boxes_count_cumulative_alt.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                 y_boxes_count_cumulative_alt.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                              x_box_idx.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                              interactions_x_parsed.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
                                                              interactions_y.packed_accessor64<int,1,torch::RestrictPtrTraits>()
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
                                x_box.box_indices_sorted_reindexed,
                             device_gpu,
                             x_box.sorted_index,
                                y_box.sorted_index,
                             x_box.unique_counts_cum_reindexed,
                             y_box.unique_counts_cum_reindexed,
                             interactions_x_parsed,
                             interactions_y,
                             true);
};

template <typename scalar_t>
torch::Tensor chebyshev_nodes_1D(const int & nodes){
    float PI = atan(1.)*4.;
    torch::Tensor chebyshev_nodes = torch::arange(0, nodes).toType(dtype<scalar_t>());
    chebyshev_nodes = torch::cos((chebyshev_nodes*2.+1.)*PI/(2*(nodes-1)+2));
    return chebyshev_nodes;
}

template <typename scalar_t>
torch::Tensor chebyshev_nodes_1D_second_kind(const int & nodes){
    float PI = atan(1.)*4.;
    torch::Tensor chebyshev_nodes = torch::arange(0, nodes).toType(dtype<scalar_t>());
    chebyshev_nodes = torch::cos(chebyshev_nodes*PI/(nodes-1));
    return chebyshev_nodes;
}

template <typename scalar_t>
torch::Tensor get_w_j(torch::Tensor & nodes){
    int n=nodes.size(0);
    torch::Tensor output = torch::zeros({n}).toType(dtype<scalar_t>());
    auto node_accessor = nodes.accessor<scalar_t,1>();
    auto output_accessor = output.accessor<scalar_t,1>();
    scalar_t tmp;
    for (int i=0; i<n;i++){
        tmp=1.0;
        for (int j=0; j<n;j++){
            if (i!=j){
                tmp*=node_accessor[i]-node_accessor[j];
            }
        }
        output_accessor[i]=1/tmp;
    }
    return output;
}

template <typename scalar_t>
torch::Tensor get_w_j_first_kind(int & nr_of_nodes){
    torch::Tensor output = torch::zeros({nr_of_nodes}).toType(dtype<scalar_t>());
    auto output_accessor = output.accessor<scalar_t,1>();
    scalar_t base = -1.;
    float PI = atan(1.)*4.;
    for (int i=0;i<nr_of_nodes;i++){
        output_accessor[i] = pow(base,i)*sin(((2.*(float)i+1.)*PI)/(2.*(float)(nr_of_nodes-1.0)+2.));
    }

    return output;
}

template <typename scalar_t,int nd>
torch::Tensor interactions_readapt_indices(torch::Tensor & interactions,
                                           n_tree_cuda<scalar_t,nd>& ntree_X,
                                           n_tree_cuda<scalar_t,nd>& ntree_Y

){
    dim3 blockSize,gridSize;
    int memory;
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(2, interactions.size(0));
    torch::Tensor &removed_idx_X = ntree_X.old_new_map;
    torch::Tensor &removed_idx_Y = ntree_Y.old_new_map;

    if(ntree_X.depth==ntree_Y.depth){
        transpose_to_existing_only<<<gridSize,blockSize>>>(
                interactions.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
                removed_idx_X.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                removed_idx_Y.packed_accessor64<int,1,torch::RestrictPtrTraits>()
        );
    }
    if(ntree_X.depth<ntree_Y.depth){
        transpose_to_existing_only_Y<<<gridSize,blockSize>>>(
                interactions.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
                removed_idx_Y.packed_accessor64<int,1,torch::RestrictPtrTraits>()
        );
    }
    if(ntree_X.depth>ntree_Y.depth){
        transpose_to_existing_only_X<<<gridSize,blockSize>>>(
                interactions.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
                removed_idx_X.packed_accessor64<int,1,torch::RestrictPtrTraits>()
        );
    }

    return interactions;

}

template <typename scalar_t>
torch::Tensor get_w_j_second_kind(int & nr_of_nodes){
    torch::Tensor output = torch::zeros({nr_of_nodes}).toType(dtype<scalar_t>());
    auto output_accessor = output.accessor<scalar_t,1>();
    scalar_t base = -1.;
    scalar_t delta;
    for (int i=0;i<nr_of_nodes;i++){
        if (i==0 or i==(nr_of_nodes-1)){
            delta = 0.5;
        }else{
            delta = 1.0;
        }
        output_accessor[i] = pow(base,i)*delta;
    }

    return output;
}

template <typename scalar_t>
std::tuple<torch::Tensor,torch::Tensor> concat_many_nodes(torch::Tensor & node_list){
    std::vector<torch::Tensor> list_of_cheb = {};
    std::vector<torch::Tensor> list_of_w_j = {};
    torch::Tensor tmp_cheb,tmp_w_j;
    auto node_list_accessor = node_list.accessor<int,1>();
    for (int i=0;i<node_list.size(0);i++){
        tmp_cheb = chebyshev_nodes_1D<scalar_t>(node_list_accessor[i]);
        tmp_w_j = get_w_j_first_kind<scalar_t>(node_list_accessor[i]);
        list_of_cheb.push_back(tmp_cheb);
        list_of_w_j.push_back(tmp_w_j);
    }
    torch::Tensor chebyshev_nodes = torch::cat({list_of_cheb},0);
    torch::Tensor w_j_cheb = torch::cat({list_of_w_j},0);
    return std::make_tuple(chebyshev_nodes,w_j_cheb);
};

template<int nd>
torch::Tensor get_node_list(int & max_nodes){ //do box_variance weighted selective interpolation?
    float ref_val = pow(max_nodes, 1.0/(float)nd);
    int highest_mult = (int) std::ceil(ref_val);
    int safe_mult = (int) std::floor(ref_val);
    int acc = 1;
    int future;
    torch::Tensor torch_node_list = torch::zeros({nd}).toType(torch::kInt32);
    auto torch_node_list_accessor = torch_node_list.accessor<int,1>();
    for (int i=0;i<nd;i++){
        future = acc*pow(safe_mult,nd-(i+1))*highest_mult;
        if (future<=max_nodes){
            torch_node_list_accessor[i]=highest_mult;
            acc*=highest_mult;
        }else{
            torch_node_list_accessor[i]=safe_mult;
            acc*=safe_mult;
        }
    }
    torch_node_list = torch_node_list.index({torch::randperm(nd).toType(torch::kLong)});
    return torch_node_list;
}

template <typename scalar_t,int nd>
void apply_laplace_interpolation_v2(
        n_tree_cuda<scalar_t,nd>& n_tree,
        torch::Tensor &b,
        const std::string & device_gpu,
        torch::Tensor & nodes,
        torch::Tensor & laplace_indices,
        torch::Tensor & node_list_cum,
        torch::Tensor & cheb_w,
        const bool & transpose,
        torch::Tensor & output

        ){
    torch::Tensor & boxes_count = n_tree.unique_counts;
    torch::Tensor & idx_reordering = n_tree.sorted_index;
    torch::Tensor & data = n_tree.data;
    torch::Tensor & centers = n_tree.centers;
    torch::Tensor & edge = n_tree.edge;

    dim3 blockSize,gridSize;
    int memory,blkSize;
    torch::Tensor indicator,box_block;
    int min_size=boxes_count.min().item<int>();
    blkSize = optimal_blocksize(min_size);
    std::tie(blockSize,gridSize,memory,indicator,box_block) = skip_kernel_launch<scalar_t>(nd,blkSize,boxes_count,n_tree.box_indices_sorted_reindexed);
    memory = memory+2*nodes.size(0)*sizeof(scalar_t)+(nd+1)*sizeof(int); //Seems the last write is where the trouble is...
    torch::Tensor boxes_count_cumulative = n_tree.unique_counts_cum_reindexed;

    if (transpose){

        lagrange_shared<scalar_t, nd><<<gridSize, blockSize, memory>>>(
                data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                nodes.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                laplace_indices.packed_accessor64<int, 2, torch::RestrictPtrTraits>(),
                output.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                indicator.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
                box_block.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
                boxes_count_cumulative.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
                centers.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                edge.data_ptr<scalar_t>(),
                idx_reordering.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
                node_list_cum.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
                cheb_w.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>()
        );
        cudaDeviceSynchronize();

    }else{
        laplace_shared_transpose<scalar_t,nd><<<gridSize,blockSize,memory>>>(
                data.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                b.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                nodes.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
                laplace_indices.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
                output.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                indicator.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                box_block.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                boxes_count_cumulative.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                centers.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                edge.data_ptr<scalar_t>(),
                idx_reordering.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                node_list_cum.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                cheb_w.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>()
        );
        cudaDeviceSynchronize();
    }
}


template <typename scalar_t,int nd>
torch::Tensor setup_skip_conv(
                            torch::Tensor &cheb_data_X,
                            torch::Tensor &cheb_data_Y,
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
    cudaMalloc((void **)&d_ls, sizeof(scalar_t));
    cudaMemcpy(d_ls, &ls, sizeof(scalar_t), cudaMemcpyHostToDevice);
    torch::Tensor indicator,box_block,output;
    int cheb_data_size=cheb_data_X.size(0);
    int blkSize = optimal_blocksize(cheb_data_size);
//    torch::Tensor boxes_count = min_size*torch::ones({unique_sorted_boxes_idx.size(0)+1}).toType(torch::kInt32);
    torch::Tensor boxes_count = cheb_data_size * torch::ones(unique_sorted_boxes_idx.size(0)).toType(torch::kInt32).to(device_gpu);
    dim3 blockSize,gridSize;
    int memory;
    unique_sorted_boxes_idx = unique_sorted_boxes_idx.toType(torch::kInt32);
    std::tie(blockSize,gridSize,memory,indicator,box_block) = skip_kernel_launch<scalar_t>(nd,blkSize,boxes_count,unique_sorted_boxes_idx);
    output = torch::zeros({ centers_X.size(0)*cheb_data_X.size(0),b_data.size(1)}).toType(dtype<scalar_t>()).to(device_gpu);
    skip_conv_far_boxes_opt<scalar_t,nd><<<gridSize,blockSize,memory>>>(
            cheb_data_X.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
            cheb_data_Y.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
            b_data.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
            output.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
            d_ls,
            centers_X.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
            centers_Y.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
            indicator.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
            box_block.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
            interactions_x_parsed.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
            interactions_y.packed_accessor64<int,1,torch::RestrictPtrTraits>()
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
                        torch::Tensor & cheb_data,
                        torch::Tensor & node_list_cum,
                       torch::Tensor & cheb_w
){
    torch::Tensor low_rank_y;
    torch::Tensor cheb_data_X = cheb_data*x_box.edge/2.+x_box.edge/2.; //scaling lagrange nodes to edge scale
    torch::Tensor cheb_data_Y = cheb_data*y_box.edge/2.+y_box.edge/2.; //scaling lagrange nodes to edge scale
    low_rank_y = torch::zeros({cheb_data.size(0)*y_box.centers.size(0),b.size(1)}).toType(dtype<scalar_t>()).to(device_gpu);
    //Found the buggie, the low_rank_y is proportioned towards x, but when Y has more non empty boxes things implode!!!

    //if bottom mode i.e. max points = 1 skip y interpolation

    apply_laplace_interpolation_v2<scalar_t,nd>(y_box,
                                            b,
                                            device_gpu,
                                            chebnodes_1D,
                                            laplace_combinations,
                                                node_list_cum,
                                                cheb_w,
                                            true,
                                            low_rank_y
                                            ); //no problems here!

    // replace cheb_data_Y = 0 points, low_rank_y = b, cheb_Data_Y = 0
    low_rank_y =  setup_skip_conv<scalar_t,nd>( //error happens here
            cheb_data_X,
            cheb_data_Y,
            low_rank_y,
            x_box.centers,
            y_box.centers,
            x_box.box_indices_sorted_reindexed,
            ls,
            device_gpu,
            interactions_x_parsed,
            interactions_y
    );
    apply_laplace_interpolation_v2<scalar_t,nd>(x_box,
                                            low_rank_y,
                                            device_gpu,
                                            chebnodes_1D,
                                            laplace_combinations,
                                                node_list_cum,
                                                cheb_w,
                                                false,
                                            output
    );
};
torch::Tensor get_new_interactions(
        int & x_div_num,int & y_div_num,
        torch::Tensor & old_near_interactions, int & p,const std::string & gpu_device){//fix tmrw
    int n = old_near_interactions.size(0);
    if(x_div_num==y_div_num){
        torch::Tensor arr = torch::arange(p).toType(torch::kInt32).to(gpu_device);
        torch::Tensor new_interactions_vec = torch::stack({arr.repeat_interleave(p).repeat(n),arr.repeat(p*n)},1)+p*old_near_interactions.repeat_interleave(p*p,0);
        return new_interactions_vec;
    }
    if(x_div_num<y_div_num){
        std::vector<torch::Tensor>  tmp = old_near_interactions.repeat_interleave(p,0).unbind(1);
        torch::Tensor & keep = tmp[0];
        torch::Tensor & expand = tmp[1];
        torch::Tensor arr = torch::arange(p).repeat(n).toType(torch::kInt32).to(gpu_device);
        torch::Tensor new_interactions_vec = torch::stack({keep,arr+p*expand},1);
        return new_interactions_vec;
    }
    if(x_div_num>y_div_num){
        std::vector<torch::Tensor>  tmp = old_near_interactions.repeat_interleave(p,0).unbind(1);
        torch::Tensor  & keep = tmp[1];
        torch::Tensor & expand = tmp[0];
        torch::Tensor arr = torch::arange(p).repeat(n).toType(torch::kInt32).to(gpu_device);
        torch::Tensor new_interactions_vec = torch::stack({arr+p*expand,keep},1);
        return new_interactions_vec;
    }


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
            count_cumsum.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
            results.packed_accessor64<int,2,torch::RestrictPtrTraits>()
            );
    cudaDeviceSynchronize();
    return results;
}



template<typename scalar_t,int d>
std::tuple<torch::Tensor,torch::Tensor> parse_cheb_data_smolyak(
        torch::Tensor & cheb_nodes,
        const std::string & gpu_device,
        torch::Tensor & nodes_per_dim
){
    torch::Tensor nodes_cum_prod = nodes_per_dim.cumprod(0).toType(torch::kInt32);
    int n = nodes_per_dim.prod(0).toType(torch::kInt32).item<int>();
    torch::Tensor cheb_idx = torch::zeros({n,d}).toType(torch::kInt32).to(gpu_device);
    torch::Tensor cheb_data = torch::zeros({n,d}).toType(dtype<scalar_t>()).to(gpu_device);
    dim3 block,grid;
    int shared;
    std::tie(block,grid,shared) =  get_kernel_launch_params<scalar_t>(d,n);
    get_smolyak_indices<scalar_t,d><<<grid,block,shared>>>(
            cheb_nodes.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
            cheb_data.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
            cheb_idx.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
            nodes_per_dim.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
            nodes_cum_prod.packed_accessor64<int,1,torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();
    return std::make_tuple(cheb_idx,cheb_data);


}

template<typename scalar_t,int d>
std::tuple<torch::Tensor,torch::Tensor> parse_cheb_data(
        torch::Tensor & cheb_nodes,
        const std::string & gpu_device,
        int & nr_of_samples
        ){
    int n = (int) pow(cheb_nodes.size(0),d);
    torch::Tensor tmp = torch::randperm(n).slice(0,0,nr_of_samples);
    torch::Tensor sampled_indices = tmp.toType(torch::kInt32).to(gpu_device);
    torch::Tensor cheb_idx = torch::zeros({nr_of_samples,d}).toType(torch::kInt32).to(gpu_device);
    torch::Tensor cheb_data = torch::zeros({nr_of_samples,d}).toType(dtype<scalar_t>()).to(gpu_device);
    dim3 block,grid;
    int shared;
    std::tie(block,grid,shared) =  get_kernel_launch_params<scalar_t>(d,n);
    get_cheb_idx_data<scalar_t,d><<<grid,block,shared>>>(
            cheb_nodes.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
            cheb_data.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
            cheb_idx.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
            sampled_indices.packed_accessor64<int,1,torch::RestrictPtrTraits>()
            );
    cudaDeviceSynchronize();
    return std::make_tuple(cheb_idx,cheb_data);

}

std::tuple<torch::Tensor,torch::Tensor> unbind_sort(torch::Tensor & interactions){
    if (interactions.dim()<2){
        interactions = interactions.unsqueeze(0);
    }
    std::vector<torch::Tensor>  tmp = interactions.unbind(1);
    return std::make_tuple(tmp[0],tmp[1]);
}

template <typename scalar_t, int nd>
torch::Tensor get_low_variance_pairs(
        n_tree_cuda<scalar_t,nd> & ntree_X,
        torch::Tensor & big_enough
){
    torch::Tensor box_variance_1 = torch::zeros({big_enough.size(0),nd}).toType(dtype<scalar_t>()).to(big_enough.device());
    torch::Tensor & x_dat = ntree_X.data;
    torch::Tensor & box_ind = ntree_X.sorted_index;
    torch::Tensor & x_box_cum = ntree_X.unique_counts_cum;
    dim3 blockSize,gridSize;
    int mem;
    std::tie(blockSize,gridSize,mem)=get_kernel_launch_params<scalar_t>(nd,big_enough.size(0));
    box_variance<scalar_t, nd><<<gridSize, blockSize, mem>>>(
            x_dat.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            box_ind.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
            x_box_cum.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
            big_enough.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
            box_variance_1.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();
    return box_variance_1;

}

template <typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> separate_interactions(
        torch::Tensor & interactions,
        n_tree_cuda<scalar_t,nd> & ntree_X,
        n_tree_cuda<scalar_t,nd> & ntree_Y,
        const std::string & gpu_device,
        int & small_field_limit,
        int & nr_of_interpolation_points,
        scalar_t & ls,
        bool & var_comp,
        scalar_t & eff_var_limit
){
    auto *d_small_field_limit = allocate_scalar_to_cuda<int>(small_field_limit);
    auto *d_nr_of_interpolation_points = allocate_scalar_to_cuda<int>(nr_of_interpolation_points);

    dim3 blockSize,gridSize;
    int mem,n;
    n = interactions.size(0);
    std::tie(blockSize,gridSize,mem)=get_kernel_launch_params<scalar_t>(nd,n);
    torch::Tensor far_field_mask = torch::zeros({n}).toType(torch::kBool).to(gpu_device);
    torch::Tensor small_field_mask = torch::zeros({n}).toType(torch::kBool).to(gpu_device);
    torch::Tensor &centers_X  = ntree_X.centers;
    torch::Tensor &centers_Y  = ntree_Y.centers;
    torch::Tensor &unique_X  = ntree_X.unique_counts;
    torch::Tensor &unique_Y  = ntree_Y.unique_counts;
    torch::Tensor edge_X = ntree_X.edge;
    torch::Tensor edge_Y = ntree_Y.edge;
    torch::Tensor edge  = torch::stack({edge_X,edge_Y}).max();
    if(var_comp){
        torch::Tensor x_var,max_var_x,max_var_y,tmp;
        auto *d_eff_var_limit = allocate_scalar_to_cuda<scalar_t>(eff_var_limit);
        x_var = get_low_variance_pairs<scalar_t,nd>(ntree_X,ntree_X.box_indices_sorted);
        std::tie(max_var_x,tmp) = x_var.max(1);
        max_var_x = max_var_x/ls;
        if (ntree_X.data.data_ptr()!=ntree_Y.data.data_ptr()){
            torch::Tensor y_var = get_low_variance_pairs<scalar_t,nd>(ntree_Y,ntree_Y.box_indices_sorted);
            std::tie(max_var_y,tmp) = y_var.max(1);
            max_var_y = max_var_y/ls;
        }else{
            max_var_y = max_var_x;
        }

        boolean_separate_interactions_small_var_comp<scalar_t,nd><<<gridSize,blockSize>>>(
                centers_X.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                centers_Y.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                unique_X.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                unique_Y.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                max_var_x.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
                max_var_y.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
                interactions.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
                edge.data_ptr<scalar_t>(),
                far_field_mask.packed_accessor64<bool,1,torch::RestrictPtrTraits>(),
                small_field_mask.packed_accessor64<bool,1,torch::RestrictPtrTraits>(),
                d_nr_of_interpolation_points,
                d_small_field_limit,
                d_eff_var_limit
        );
        cudaDeviceSynchronize();
    }else{
        boolean_separate_interactions_small<scalar_t,nd><<<gridSize,blockSize>>>(
                centers_X.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                centers_Y.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                unique_X.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                unique_Y.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                interactions.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
                edge.data_ptr<scalar_t>(),
                far_field_mask.packed_accessor64<bool,1,torch::RestrictPtrTraits>(),
                small_field_mask.packed_accessor64<bool,1,torch::RestrictPtrTraits>(),
                d_nr_of_interpolation_points,
                d_small_field_limit
        );
        cudaDeviceSynchronize();

    }

    return std::make_tuple(interactions.index({far_field_mask}),
                           interactions.index({small_field_mask}),
                           interactions.index({ torch::logical_not(torch::logical_or(far_field_mask,small_field_mask))})
    );
}
template <typename scalar_t, int nd>
torch::Tensor filter_out_interactions(torch::Tensor & interactions,
                                      n_tree_cuda<scalar_t,nd> & ntree_X,
                                      n_tree_cuda<scalar_t,nd> & ntree_Y){
    dim3 blockSize,gridSize;
    int memory;
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(nd, interactions.size(0));
    torch::Tensor mask = torch::zeros(interactions.size(0)).toType(torch::kBool).to(interactions.device());
    torch::Tensor &x_keep = ntree_X.non_empty_mask;
    torch::Tensor &y_keep = ntree_Y.non_empty_mask;
    get_keep_mask<<<gridSize,blockSize>>>(
            interactions.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
            x_keep.packed_accessor64<bool,1,torch::RestrictPtrTraits>(),
            y_keep.packed_accessor64<bool,1,torch::RestrictPtrTraits>(),
            mask.packed_accessor64<bool,1,torch::RestrictPtrTraits>()
            );
    interactions = interactions.index({mask});
    interactions = interactions_readapt_indices<scalar_t, nd>(interactions, ntree_X, ntree_Y);
    interactions = interactions.index({torch::argsort(interactions.slice(1,0,1).squeeze()),torch::indexing::Slice()});
    if (interactions.dim()<2){
        interactions = interactions.unsqueeze(0);
    }
    return interactions;

}

template <typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> smolyak_grid(int nr_of_interpolation_points, const std::string & gpu_device){
    torch::Tensor cheb_data_X,
            laplace_combinations,
            node_list,
            node_list_cum,
            chebnodes_1D,
            cheb_w;
    node_list = get_node_list<nd>(nr_of_interpolation_points);
    std::tie(chebnodes_1D,cheb_w) = concat_many_nodes<scalar_t>(node_list);
    node_list = node_list.to(gpu_device);
    chebnodes_1D = chebnodes_1D.to(gpu_device);
    cheb_w = cheb_w.to(gpu_device);
    std::tie(laplace_combinations,cheb_data_X) = parse_cheb_data_smolyak<scalar_t,nd>(chebnodes_1D,gpu_device,node_list);
    node_list_cum = node_list.cumsum(0).toType(torch::kInt32);
    return std::make_tuple(cheb_data_X,laplace_combinations,node_list_cum,chebnodes_1D,cheb_w);
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
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> get_field(
        torch::Tensor & near_field,
        n_tree_cuda<scalar_t,nd> & ntree_X,
        n_tree_cuda<scalar_t,nd> & ntree_Y,
        scalar_t & ls,
        int & nr_of_interpolation_points,
        const std::string & gpu_device,
        bool & var_compression,
        scalar_t  & eff_var_limit,
        int & small_field_limit
        ){

    torch::Tensor interactions, far_field,small_field;
//    std::cout<<"active boxes X: "<<ntree_X.box_indices_sorted_reindexed.size(0)<<std::endl;
//    std::cout<<"active boxes Y: "<<ntree_Y.box_indices_sorted_reindexed.size(0)<<std::endl;
    if ((near_field.size(0)*ntree_X.dim_fac*ntree_Y.dim_fac)>1e8){
        std::vector<torch::Tensor> interaction_list ={};
        std::vector<torch::Tensor>chunked_near_field=torch::chunk(near_field,10,0);
        for (torch::Tensor &inter_subset : chunked_near_field){
            inter_subset = get_new_interactions(ntree_X.depth,ntree_Y.depth,inter_subset,ntree_X.dim_fac,gpu_device); //Doesn't work for new setup since the division is changed...
//            std::cout<<"interactions_1: "<<inter_subset.size(0)<<std::endl;
            inter_subset = filter_out_interactions(inter_subset,ntree_X,ntree_Y);
//            std::cout<<"interactions_2: "<<inter_subset.size(0)<<std::endl;
            interaction_list.push_back(inter_subset);
        }
        interactions = torch::cat(interaction_list,0);

    }else{
        interactions = get_new_interactions(ntree_X.depth,ntree_Y.depth,near_field,ntree_X.dim_fac,gpu_device); //Doesn't work for new setup since the division is changed...
//        std::cout<<"interactions_1: "<<interactions.size(0)<<std::endl;
        interactions = filter_out_interactions(interactions,ntree_X,ntree_Y);
//        std::cout<<"interactions_2: "<<interactions.size(0)<<std::endl;
    }

    std::tie(far_field,small_field,near_field) =
            separate_interactions<scalar_t,nd>(
                    interactions,
                    ntree_X,
                    ntree_Y,
                    gpu_device,
                    small_field_limit,
                    nr_of_interpolation_points,
                    ls,
                    var_compression,
                    eff_var_limit
            );
    return std::make_tuple(far_field,small_field,near_field);

}



template <typename scalar_t, int nd>
torch::Tensor far_field_run(
        n_tree_cuda<scalar_t,nd> & ntree_X,
        n_tree_cuda<scalar_t,nd> & ntree_Y,
        torch::Tensor & near_field,
        torch::Tensor & output,
        torch::Tensor & b,
        scalar_t & ls,
        int & nr_of_interpolation_points,
        const std::string & gpu_device,
        bool & var_compression,
        scalar_t  & eff_var_limit,
        int & small_field_limit
){
    torch::Tensor far_field,small_field,interactions_x,interactions_y,interactions_x_parsed,cheb_data,
            laplace_combinations,
            chebnodes_1D,
            node_list_cum,
            cheb_w,
            x_var,
            max_var
            ;

//    std::cout<<"near_field: "<<near_field.size(0)<<std::endl;
    std::tie(far_field,small_field,near_field) = get_field<scalar_t,nd>(
            near_field,
            ntree_X,
            ntree_Y,
            ls,
            nr_of_interpolation_points,
            gpu_device,
            var_compression,
            eff_var_limit,
            small_field_limit
            );
    if (small_field.numel()>0) {
        near_field_run<scalar_t, nd>(ntree_X, ntree_Y, small_field, output, b, ls, gpu_device);
    }
    if (far_field.numel()>0) {

        std::tie(cheb_data,laplace_combinations,node_list_cum,chebnodes_1D,cheb_w) = smolyak_grid<scalar_t,nd>(nr_of_interpolation_points,gpu_device);
//        torch::Tensor cheb_data_X = cheb_data*ntree_X.edge/2.+ntree_X.edge/2.; //scaling lagrange nodes to edge scale
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
                cheb_data,
                node_list_cum,
                cheb_w
        ); //far field compute
    }
    return near_field;
}


template <typename scalar_t, int nd>
torch::Tensor FFM_XY(
        torch::Tensor &X_data,
        torch::Tensor &Y_data,
        torch::Tensor &b,
        const std::string & gpu_device,
        scalar_t & ls,
        float &min_points,
        int & nr_of_interpolation_points,
        bool &var_compression,
        scalar_t  & eff_var_limit,
        int & small_field_limit
) {

    torch::Tensor output = torch::zeros({X_data.size(0),b.size(1)}).to(gpu_device); //initialize empty output
    torch::Tensor edge,
            xmin,
            ymin,
            xmax,
            ymax,
            near_field,
            x_edge,
            y_edge
            ;
    near_field = torch::zeros({1,2}).toType(torch::kInt32).to(gpu_device);
    if (X_data.data_ptr()==Y_data.data_ptr()){
        std::tie(edge,xmin,xmax) = calculate_edge_X<scalar_t,nd>(X_data,gpu_device); //actually calculate them
        n_tree_cuda<scalar_t,nd> ntree_X = n_tree_cuda<scalar_t,nd>(edge,X_data,xmin,xmax,gpu_device);
        while (near_field.numel()>0 and ntree_X.avg_nr_points > min_points){
            ntree_X.divide();//needs to be fixed... Should get 451 errors, OK. Memory issue is consistent
            near_field = far_field_run<scalar_t, nd>(
                    ntree_X,
                    ntree_X,
                    near_field,
                    output,
                    b,
                    ls,
                    nr_of_interpolation_points,
                    gpu_device,
                    var_compression,
                    eff_var_limit,
                    small_field_limit
            );
        }

        if (near_field.numel()>0){
            near_field_run<scalar_t,nd>(ntree_X,ntree_X,near_field,output,b,ls,gpu_device);
        }
    }else{
        std::tie(edge,xmin,ymin,xmax,ymax,x_edge,y_edge) = calculate_edge<scalar_t,nd>(X_data,Y_data,gpu_device); //actually calculate them
        n_tree_cuda<scalar_t,nd> ntree_X = n_tree_cuda<scalar_t,nd>(edge,X_data,xmin,xmax,gpu_device);
        n_tree_cuda<scalar_t,nd> ntree_Y = n_tree_cuda<scalar_t,nd>(edge,Y_data,ymin,ymax,gpu_device);

        float min_points_X,min_points_Y;
        if (X_data.size(0)>Y_data.size(0)){
            min_points_X = min_points;
            min_points_Y = max((int)ceil(((float)Y_data.size(0)/(float)X_data.size(0))*min_points),(int)1);
        }else{
            min_points_Y = min_points;
            min_points_X = max((int)ceil(((float)X_data.size(0)/(float)Y_data.size(0))*min_points),(int)1);
        }



        while (near_field.numel()>0 and (ntree_X.avg_nr_points > min_points_X and ntree_Y.avg_nr_points > min_points_Y)){
            ntree_X.divide();//needs to be fixed... Should get 451 errors, OK. Memory issue is consistent
            ntree_Y.divide();//needs to be fixed... Should get 451 errors, OK. Memory issue is consistent
            near_field = far_field_run<scalar_t, nd>(
                    ntree_X,
                    ntree_Y,
                    near_field,
                    output,
                    b,
                    ls,
                    nr_of_interpolation_points,
                    gpu_device,
                    var_compression,
                    eff_var_limit,
                    small_field_limit
            );
        }
        if (near_field.numel()>0){
            near_field_run<scalar_t,nd>(ntree_X,ntree_Y,near_field,output,b,ls,gpu_device);
        }
    }

    return output;
}

template <typename scalar_t, int nd>
struct FFM_object{
    torch::Tensor & X_data;
    torch::Tensor & Y_data;
    scalar_t & ls;
    const std::string & gpu_device;
    int & nr_of_interpolation_points;
    float & min_points;
    bool &var_compression;
    scalar_t  & eff_var_limit;
    int & small_field_limit;

    FFM_object( //constructor
            torch::Tensor & X_data,
    torch::Tensor & Y_data,
    scalar_t & ls,
    const std::string & gpu_device,
    float & min_points,
    int & nr_of_interpolation_points,
    bool & var_comp,
    scalar_t  & eff_var,
    int & small_field_limit
    ): X_data(X_data), Y_data(Y_data),ls(ls),gpu_device(gpu_device),min_points(min_points),nr_of_interpolation_points(nr_of_interpolation_points),
    var_compression(var_comp),eff_var_limit(eff_var),small_field_limit(small_field_limit){
    };
    virtual torch::Tensor operator* (torch::Tensor & b){
        if (X_data.data_ptr()==Y_data.data_ptr()){
            return FFM_XY<scalar_t,nd>(
                    X_data,
                    X_data,
                    b,
                    gpu_device,
                    ls,
                    min_points,
                    nr_of_interpolation_points,
                    var_compression,
                    eff_var_limit,
                    small_field_limit
            );
//            }
        }else{
                return FFM_XY<scalar_t, nd>(
                        X_data,
                        Y_data,
                        b,
                        gpu_device,
                        ls,
                        min_points,
                        nr_of_interpolation_points,
                        var_compression,
                        eff_var_limit,
                        small_field_limit
                );
        }
    };
};
template <typename scalar_t, int nd>
struct exact_MV : FFM_object<scalar_t,nd>{
                    exact_MV(
                        torch::Tensor & X_data,
                         torch::Tensor & Y_data,
                         scalar_t & ls,
                         const std::string & gpu_device,
                        float & min_points,
                         int & nr_of_interpolation_points,
                         bool &var_compression,
                        scalar_t  & eff_var_limit,
                        int & small_field_limit
                    )
                         : FFM_object<scalar_t,nd>(X_data, Y_data, ls, gpu_device,min_points,nr_of_interpolation_points,var_compression,eff_var_limit,small_field_limit){};
    torch::Tensor operator* (torch::Tensor & b) override{
        return  rbf_call<scalar_t,nd>(
                FFM_object<scalar_t,nd>::X_data,
                FFM_object<scalar_t,nd>::Y_data,
                b,
                FFM_object<scalar_t,nd>::ls,
                true
        );
    };
};


