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

template <typename scalar_t>
scalar_t * allocate_scalar_to_cuda(scalar_t & ls){
    scalar_t *d_ls;
    cudaMalloc((void **)&d_ls, sizeof(scalar_t));
    cudaMemcpy(d_ls, &ls, sizeof(scalar_t), cudaMemcpyHostToDevice);
    return d_ls;
}

template<typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> calculate_edge(const torch::Tensor &X,const torch::Tensor &Y,const std::string & gpu_device){
    torch::Tensor Xmin,Ymin,Xmax,Ymax,tmp,x_edge,y_edge;
    std::tie(Xmin,tmp) = X.min(0);
    std::tie(Xmax,tmp) = X.max(0);
    std::tie(Ymin,tmp) = Y.min(0);
    std::tie(Ymax,tmp) = Y.max(0);
    x_edge=(Xmax - Xmin).max();
    y_edge=(Ymax - Ymin).max();
    torch::Tensor edge = torch::stack({x_edge,y_edge}).max();
    return std::make_tuple(edge*1.01,Xmin,Ymin,Xmax,Ymax,x_edge*1.01,y_edge*1.01);
};


template<typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> calculate_edge_X(const torch::Tensor &X, const std::string & gpu_device){
    torch::Tensor Xmin,Xmax,tmp;
    std::tie(Xmin,tmp) = X.min(0);
    std::tie(Xmax,tmp) = X.max(0);
    torch::Tensor edge = (Xmax - Xmin).max();
    return std::make_tuple(edge*1.01,Xmin,Xmax);
};



template <typename scalar_t,int dim>
struct n_tree_cuda{
    torch::Tensor &data;
    torch::Tensor edge,
            edge_og,
            xmin,
            xmax,
            side,
            side_base,
            sorted_index,
            unique_counts,
            multiply_gpu_base,
            multiply_gpu,
            box_indices_sorted,
            box_indices_sorted_reindexed,
            centers,
            empty_box_indices,
            non_empty_mask,
            box_idxs,
            unique_counts_cum,
            unique_counts_cum_reindexed,
            coord_tensor,
            old_new_map,
            perm,
            tmp_1,
            tmp_2;
    std::string device;
    float avg_nr_points;
    int dim_fac,depth,hash_table_size;
    n_tree_cuda(const torch::Tensor& e, torch::Tensor &d, torch::Tensor &xm,torch::Tensor &xma, const std::string &cuda_str ):data(d){
        device = cuda_str;
        xmin = xm;
        xmax = xma;
        edge = e;
        edge_og = e;
        dim_fac = pow(2,dim);
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
        depth += 1;
        side = side_base.pow(depth);
        box_idxs = torch::arange(centers.size(0)).toType(torch::kInt32).to(device);
        unique_counts_cum = torch::zeros(centers.size(0)+1).toType(torch::kInt32).to(device);//matrix -> existing boxes * 2^dim bounded over nr or points... betting on boxing dissapears...
        unique_counts= torch::zeros(centers.size(0)).toType(torch::kInt32).to(device);
        hash_table_size=1024*1024;
        while(hash_table_size<(int)(2*centers.size(0))){
            hash_table_size*=2;
        }
        auto hash_table_size_pointer= allocate_scalar_to_cuda<int>(hash_table_size);
        KeyValue* perm_hash = create_hashtable_size(hash_table_size);
//        std::cout<<"allocated memory: "<< sizeof(KeyValue) * hash_table_size<<std::endl;
        dim3 blockSize,gridSize;
        int memory;
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(dim, centers.size(0));
        center_perm_hash<scalar_t,dim><<<gridSize,blockSize>>>(
                centers.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                xmin.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
                multiply_gpu.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                side.data_ptr<scalar_t>(),
                edge_og.data_ptr<scalar_t>(),
                perm_hash,
                hash_table_size_pointer
        ); //Apply same hack but to centers to get perm

        cudaDeviceSynchronize();
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(dim, data.size(0));
        box_division_cum_hash<scalar_t,dim><<<gridSize,blockSize>>>(
                data.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                xmin.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
                multiply_gpu.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                side.data_ptr<scalar_t>(),
                edge_og.data_ptr<scalar_t>(),
                unique_counts_cum.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                perm_hash,
                hash_table_size_pointer

        );
        cudaDeviceSynchronize();
        unique_counts_cum = unique_counts_cum.cumsum(0).toType(torch::kInt32);
        box_division_assign_hash<scalar_t,dim><<<gridSize,blockSize>>>(
                data.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                xmin.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
                multiply_gpu.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                side.data_ptr<scalar_t>(),
                edge_og.data_ptr<scalar_t>(),
                unique_counts_cum.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                perm_hash,
                hash_table_size_pointer,
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
        destroy_hashtable(perm_hash);
    };
};




template <typename scalar_t, int nd>
torch::Tensor rbf_call(
        torch::Tensor & cuda_X_job,
        torch::Tensor & cuda_Y_job,
        torch::Tensor & cuda_b_job,
        scalar_t & ls,
        bool shared = true
){

    scalar_t lcs = 1/(2*ls*ls);
    torch::Tensor output_job = torch::zeros({cuda_X_job.size(0), cuda_b_job.size(1)}).toType(dtype<scalar_t>()).to(cuda_X_job.device());
    auto d_ls = allocate_scalar_to_cuda<scalar_t>(lcs);

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



//Could represent indices with [box_id, nr of points], make sure its all concatenated correctly and in order.
//local variable: x_i, box belonging. How to get loading scheme. [nr of points].cumsum().  Iteration schedual...
// calculate box belonging.





template <typename scalar_t>
torch::Tensor chebyshev_nodes_1D(const int & nodes){
    float PI = atan(1.)*4.;
    torch::Tensor chebyshev_nodes = torch::arange(0, nodes).toType(dtype<scalar_t>());
    chebyshev_nodes = torch::cos((chebyshev_nodes*2.+1.)*PI/(2*(nodes-1)+2));
    return chebyshev_nodes;
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

    transpose_to_existing_only<<<gridSize,blockSize>>>(
            interactions.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
            removed_idx_X.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
            removed_idx_Y.packed_accessor64<int,1,torch::RestrictPtrTraits>()
    );


    return interactions;

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
        torch::Tensor & output,
        torch::Tensor & unique_vec,
        KeyValue* x_to_hash,
        int * hash_table_size_pointer
){
    dim3 blockSize,gridSize;
    int memory,blkSize;
    torch::Tensor & counts_ref_const = n_tree.unique_counts;
    torch::Tensor boxes_count = torch::zeros_like(unique_vec);
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(1, boxes_count.size(0));
    int_indexing<<<gridSize, blockSize>>>(
            counts_ref_const.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
            unique_vec.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
            boxes_count.packed_accessor64<int, 1, torch::RestrictPtrTraits>()
    );

    torch::Tensor & idx_reordering = n_tree.sorted_index;
    torch::Tensor & data = n_tree.data;
    torch::Tensor & centers = n_tree.centers;
    torch::Tensor & edge = n_tree.edge;

    torch::Tensor indicator,box_block;
    int min_size=boxes_count.min().item<int>();
    blkSize = optimal_blocksize(min_size);
    std::tie(blockSize,gridSize,memory,indicator,box_block) = skip_kernel_launch<scalar_t>(nd,blkSize,boxes_count,unique_vec);
    memory = memory+2*nodes.size(0)*sizeof(scalar_t)+(nd+3)*sizeof(int); //Seems the last write is where the trouble is...
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
                cheb_w.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                x_to_hash,
                hash_table_size_pointer
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
                cheb_w.packed_accessor64<scalar_t,1,torch::RestrictPtrTraits>(),
                x_to_hash,
                hash_table_size_pointer
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
        torch::Tensor & unique_x,
        scalar_t & ls,
        const std::string & device_gpu,
        torch::Tensor & interactions_x_parsed,
        torch::Tensor & interactions_y,
        KeyValue* hash_list_x,
        int * hash_list_size_x,
        KeyValue* hash_list_y,
        int * hash_list_size_y
){ ////SOMETHING WRONG HERE
    dim3 blockSize,gridSize;
    int memory;
    torch::Tensor interactions_y_hash = interactions_y.clone();
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(1,interactions_y_hash.size(0));
    create_interactions_hash_y<<<gridSize,blockSize>>>(hash_list_y,
                                                       hash_list_size_y,
                                                       interactions_y_hash.packed_accessor64<int,1,torch::RestrictPtrTraits>()
                                                               );
    cudaDeviceSynchronize();
    scalar_t *d_ls = allocate_scalar_to_cuda<scalar_t>(ls);
    torch::Tensor indicator,box_block,output;
    int cheb_data_size=cheb_data_X.size(0);
    int blkSize = optimal_blocksize(cheb_data_size);
    torch::Tensor boxes_count = cheb_data_size * torch::ones(unique_x.size(0)).toType(torch::kInt32).to(device_gpu);

    std::tie(blockSize,gridSize,memory,indicator,box_block) = skip_kernel_launch<scalar_t>(nd, blkSize, boxes_count, unique_x);
    memory = memory+sizeof(int);
    output = torch::zeros({ unique_x.size(0)*cheb_data_X.size(0),b_data.size(1)}).toType(dtype<scalar_t>()).to(device_gpu);
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
            interactions_y.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
            hash_list_x,
            hash_list_size_x,
            interactions_y_hash.packed_accessor64<int,1,torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();
    return output;
}
template <typename scalar_t,int nd>
void apply_laplace_interpolation_v2_old(
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

        lagrange_shared_old<scalar_t, nd><<<gridSize, blockSize, memory>>>(
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
        laplace_shared_transpose_old<scalar_t,nd><<<gridSize,blockSize,memory>>>(
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
torch::Tensor setup_skip_conv_old(
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
    skip_conv_far_boxes_opt_old<scalar_t,nd><<<gridSize,blockSize,memory>>>(
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
        torch::Tensor & cheb_w,
        torch::Tensor & unique_x,
        torch::Tensor & unique_y
){
    dim3 blockSize,gridSize;
    int memory;
    torch::Tensor low_rank_y;
    torch::Tensor cheb_data_X = cheb_data*x_box.edge/2.+x_box.edge/2.; //scaling lagrange nodes to edge scale
    low_rank_y = torch::zeros({cheb_data.size(0)*unique_y.size(0),b.size(1)}).toType(dtype<scalar_t>()).to(device_gpu);
    //ok might want to adapt things a bit so they don't explode at the end!

    int hash_table_size = 1024 * 1024;
    while (hash_table_size < (int) (2 * unique_x.size(0))) {
        hash_table_size *= 2;
    }
    int * hash_list_size_x = allocate_scalar_to_cuda<int>(hash_table_size);
    KeyValue *hash_list_x = create_hashtable_size(hash_table_size);
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(1, unique_x.size(0));
    hash_into_natural_order<<<gridSize, blockSize>>>(
            hash_list_x,
            hash_list_size_x,
            unique_x.packed_accessor64<int, 1, torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();

    hash_table_size = 1024 * 1024;
    while (hash_table_size < (int) (2 * unique_y.size(0))) {
        hash_table_size *= 2;
    }
    int * hash_list_size_y = allocate_scalar_to_cuda<int>(hash_table_size);
    KeyValue *hash_list_y = create_hashtable_size(hash_table_size);
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(1, unique_y.size(0));
    hash_into_natural_order<<<gridSize, blockSize>>>(
            hash_list_y,
            hash_list_size_y,
            unique_y.packed_accessor64<int, 1, torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();

//    apply_laplace_interpolation_v2<scalar_t,nd>(y_box,
//                                                b,
//                                                device_gpu,
//                                                chebnodes_1D,
//                                                laplace_combinations,
//                                                node_list_cum,
//                                                cheb_w,
//                                                true,
//                                                low_rank_y,
//                                                unique_y,
//                                                hash_list_y,
//                                                hash_list_size_y
//    ); //no problems here!
    apply_laplace_interpolation_v2_old<scalar_t,nd>(y_box,
                                                b,
                                                device_gpu,
                                                chebnodes_1D,
                                                laplace_combinations,
                                                node_list_cum,
                                                cheb_w,
                                                true,
                                                low_rank_y
    ); //no problems here!

//    low_rank_y =  setup_skip_conv<scalar_t,nd>( //error happens here
//            cheb_data_X,
//            cheb_data_X,
//            low_rank_y,
//            x_box.centers,
//            y_box.centers,
//            unique_x,
//            ls,
//            device_gpu,
//            interactions_x_parsed,
//            interactions_y,
//            hash_list_x,
//            hash_list_size_x,
//            hash_list_y,
//            hash_list_size_y
//
//    );

    low_rank_y =  setup_skip_conv_old<scalar_t,nd>( //error happens here
            cheb_data_X,
            cheb_data_X,
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
                                                output,
                                                unique_x,
                                                hash_list_x,
                                                hash_list_size_x
                                                );
    destroy_hashtable(hash_list_x);
    destroy_hashtable(hash_list_y);

};
torch::Tensor get_new_interactions(
        int & x_div_num,int & y_div_num,
        torch::Tensor & old_near_interactions, int & p,const std::string & gpu_device){//fix tmrw
    int n = old_near_interactions.size(0);
    torch::Tensor left;
    torch::Tensor right;
    torch::Tensor arr=torch::arange(p).toType(torch::kInt32).to(gpu_device);

    if (x_div_num==1){
        torch::Tensor new_interactions_vec;
        left = arr.repeat_interleave(p).repeat(n);
        right = arr.repeat(p*n);
        torch::Tensor add = p*old_near_interactions.repeat_interleave(p*p,0);
        new_interactions_vec = torch::stack({left,right},1)+add;
        return new_interactions_vec;

    }else{
        torch::Tensor output,tmp,counts;
        std::vector<torch::Tensor> ubind = old_near_interactions.unbind(1);
        torch::Tensor & old_left = ubind[0];
        torch::Tensor & old_right = ubind[1];
        std::tie(output,tmp,counts)=torch::unique_consecutive(old_left,false,true);
        torch::Tensor right_interleaved = old_right.repeat_interleave(p);
        torch::Tensor new_interactions_vec = arr.repeat_interleave(p).repeat(output.size(0)).repeat_interleave(counts.repeat_interleave(p*p))+p*old_left.repeat_interleave(p*p,0);
        new_interactions_vec  = new_interactions_vec.unsqueeze(1).repeat({1,2});
        auto p_pointer = allocate_scalar_to_cuda<int>( p);
        torch::Tensor short_cumsum = torch::cumsum(counts*p,0).toType(torch::kInt32);
        counts = counts.toType(torch::kInt32);
        dim3 blockSize,gridSize;
        int memory;
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(1, counts.size(0));
        repeat_within<<<gridSize,blockSize>>>(
                short_cumsum.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                right_interleaved.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                counts.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                new_interactions_vec.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
                p_pointer
        );
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(1, new_interactions_vec.size(0));
        repeat_add<<<gridSize,blockSize>>>(
                arr.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                new_interactions_vec.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
                p_pointer
        );
        return new_interactions_vec;

    }
}

template <int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> process_interactions(torch::Tensor & interactions_x,torch::Tensor & interactions_y, const std::string & gpu_device){
    torch::Tensor unique_x,tmp,counts,count_cumsum,results,unique_y;
    std::tie(unique_x, tmp, counts) = torch::unique_consecutive(interactions_x, false, true);
    std::tie(unique_y,tmp)=torch::_unique(interactions_y,true,false);
    count_cumsum = counts.cumsum(0).toType(torch::kInt32).to(gpu_device);  //64+1 vec
    results = torch::zeros({unique_x.size(0), 2}).toType(torch::kInt32).to(gpu_device);
    dim3 blockSize,gridSize;
    int memory;
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(nd, count_cumsum.size(0));
    parse_x_boxes<<<gridSize,blockSize>>>(
            count_cumsum.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
            results.packed_accessor64<int,2,torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();

    return std::make_tuple(unique_x,results,unique_y);
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


std::tuple<torch::Tensor,torch::Tensor> unbind_sort(torch::Tensor & interactions){
    if (interactions.dim()<2){
        interactions = interactions.unsqueeze(0);
    }
    std::vector<torch::Tensor>  tmp = interactions.unbind(1);
    return std::make_tuple(tmp[0],tmp[1]);
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
    torch::Tensor small_field_mask = far_field_mask.clone();
    torch::Tensor keep_mask = torch::ones({n}).toType(torch::kBool).to(gpu_device);
    torch::Tensor &centers_X  = ntree_X.centers;
    torch::Tensor &centers_Y  = ntree_Y.centers;
    torch::Tensor &unique_X  = ntree_X.unique_counts;
    torch::Tensor &unique_Y  = ntree_Y.unique_counts;
    torch::Tensor edge = ntree_X.edge;
    scalar_t square_edge = edge.item<scalar_t>()*edge.item<scalar_t>()*ls;
    bool do_inital_check;
    if (var_comp){
        do_inital_check = square_edge<=3;
    }else{
        do_inital_check =true;
    }
//    bool do_inital_check = true;
    auto d_bool_check = allocate_scalar_to_cuda<bool>(do_inital_check);
    if(var_comp){
        bool enable_smooth_field_pair = (float)nd*(square_edge/4)<=eff_var_limit;
        bool enable_smooth_field_all = (float)nd*(square_edge/4)<=(eff_var_limit/2);
        scalar_t crit_distance = sqrt(eff_var_limit*2/((float)nd*ls));
        auto d_enable_smooth_field_pair = allocate_scalar_to_cuda<bool>(enable_smooth_field_pair);
        auto d_enable_smooth_field_all = allocate_scalar_to_cuda<bool>(enable_smooth_field_all);
        auto d_crit_distance = allocate_scalar_to_cuda<scalar_t>(crit_distance);
        auto d_ls = allocate_scalar_to_cuda<scalar_t>(ls);
        boolean_separate_interactions_small_var_comp<scalar_t,nd><<<gridSize,blockSize>>>(
                centers_X.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                centers_Y.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                unique_X.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                unique_Y.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                interactions.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
                edge.data_ptr<scalar_t>(),
                far_field_mask.packed_accessor64<bool,1,torch::RestrictPtrTraits>(),
                small_field_mask.packed_accessor64<bool,1,torch::RestrictPtrTraits>(),
                keep_mask.packed_accessor64<bool,1,torch::RestrictPtrTraits>(),
                d_nr_of_interpolation_points,
                d_small_field_limit,
                d_bool_check,
                d_enable_smooth_field_pair,
                d_enable_smooth_field_all,
                d_crit_distance,
                d_ls
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
                keep_mask.packed_accessor64<bool,1,torch::RestrictPtrTraits>(),
                d_nr_of_interpolation_points,
                d_small_field_limit,
                d_bool_check
        );
        cudaDeviceSynchronize();

    }

    return std::make_tuple(interactions.index({far_field_mask}),
                           interactions.index({small_field_mask}),
                           interactions.index({ keep_mask})
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
    torch::Tensor interactions_x,interactions_y,interactions_x_parsed,unique_x,unique_y;
    std::tie(interactions_x,interactions_y) = unbind_sort(near_field);
    std::tie(unique_x,interactions_x_parsed,unique_y) = process_interactions<nd>(interactions_x,interactions_y,gpu_device);
    scalar_t *d_ls;
    cudaMalloc((void **)&d_ls, sizeof(scalar_t));
    cudaMemcpy(d_ls, &ls, sizeof(scalar_t), cudaMemcpyHostToDevice);
    dim3 blockSize,gridSize;
    int memory,blkSize;
    torch::Tensor & counts_ref_const = ntree_X.unique_counts;
    torch::Tensor x_boxes_count = torch::zeros_like(unique_x);
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(1, x_boxes_count.size(0));
    int_indexing<<<gridSize, blockSize>>>(
            counts_ref_const.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
            unique_x.packed_accessor64<int, 1, torch::RestrictPtrTraits>(),
            x_boxes_count.packed_accessor64<int, 1, torch::RestrictPtrTraits>()
    );
//    torch::Tensor & x_box_idx = ntree_X.box_indices_sorted_reindexed;
    torch::Tensor & cuda_X_job = ntree_X.data;
    torch::Tensor & cuda_Y_job = ntree_Y.data;
    torch::Tensor & x_boxes_count_cumulative = ntree_X.unique_counts_cum_reindexed;
    torch::Tensor & y_boxes_count_cumulative = ntree_Y.unique_counts_cum_reindexed;
    torch::Tensor & x_idx_reordering = ntree_X.sorted_index;
    torch::Tensor & y_idx_reordering = ntree_Y.sorted_index;
    torch::Tensor block_box_indicator,box_block_indicator;

    int hash_table_size = 1024 * 1024;
    while (hash_table_size < (int) (2 * unique_x.size(0))) {
        hash_table_size *= 2;
    }
    int * hash_table_size_pointer = allocate_scalar_to_cuda<int>(hash_table_size);
    KeyValue *x_to_hash = create_hashtable_size(hash_table_size);
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(1, unique_x.size(0));
    hash_into_natural_order<<<gridSize, blockSize>>>(
            x_to_hash,
            hash_table_size_pointer,
            unique_x.packed_accessor64<int, 1, torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();
    int min_size=x_boxes_count.min().item<int>();
    blkSize = optimal_blocksize(min_size);
    std::tie(blockSize, gridSize, memory, block_box_indicator, box_block_indicator) = skip_kernel_launch<scalar_t>(nd, blkSize, x_boxes_count, unique_x);
    memory = memory+sizeof(int);
    skip_conv_1d_shared<scalar_t,nd><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                    cuda_Y_job.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                    b.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                    output.packed_accessor64<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                    d_ls,
                                                                    x_boxes_count_cumulative.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                    y_boxes_count_cumulative.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                    block_box_indicator.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                    box_block_indicator.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                    x_idx_reordering.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                    y_idx_reordering.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                    interactions_x_parsed.packed_accessor64<int,2,torch::RestrictPtrTraits>(),
                                                                    interactions_y.packed_accessor64<int,1,torch::RestrictPtrTraits>(),
                                                                    x_to_hash,
                                                                    hash_table_size_pointer
    );
    cudaDeviceSynchronize();
    destroy_hashtable(x_to_hash);

}
template <typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> get_field(
        torch::Tensor & near_field_old,
        n_tree_cuda<scalar_t,nd> & ntree_X,
        n_tree_cuda<scalar_t,nd> & ntree_Y,
        scalar_t & ls,
        int & nr_of_interpolation_points,
        const std::string & gpu_device,
        bool & var_compression,
        scalar_t  & eff_var_limit,
        int & small_field_limit
){

    torch::Tensor interactions, far_field,small_field,near_field;
//    std::cout<<"active boxes X: "<<ntree_X.box_indices_sorted_reindexed.size(0)<<std::endl;
//    std::cout<<"active boxes Y: "<<ntree_Y.box_indices_sorted_reindexed.size(0)<<std::endl;
    if ((near_field_old.size(0)*ntree_X.dim_fac*ntree_Y.dim_fac)>1e9){
        far_field = torch::empty({0,2}).toType(torch::kInt32).to(near_field_old.device());
        small_field = torch::empty({0,2}).toType(torch::kInt32).to(near_field_old.device());
        near_field = torch::empty({0,2}).toType(torch::kInt32).to(near_field_old.device());
        std::vector<torch::Tensor>chunked_near_field=torch::chunk(near_field_old,10,0);
        torch::Tensor far_field_small,small_field_small,near_field_small;
        for (torch::Tensor &inter_subset : chunked_near_field){
            inter_subset = get_new_interactions(ntree_X.depth,ntree_Y.depth,inter_subset,ntree_X.dim_fac,gpu_device); //Doesn't work for new setup since the division is changed...
//            std::cout<<"interactions_1: "<<inter_subset.size(0)<<std::endl;
            inter_subset = filter_out_interactions(inter_subset,ntree_X,ntree_Y);
//            std::cout<<"interactions_2: "<<inter_subset.size(0)<<std::endl;
            std::tie(far_field_small,small_field_small,near_field_small) =
                    separate_interactions<scalar_t,nd>(
                            inter_subset,
                            ntree_X,
                            ntree_Y,
                            gpu_device,
                            small_field_limit,
                            nr_of_interpolation_points,
                            ls,
                            var_compression,
                            eff_var_limit
                    );
            far_field = torch::cat({far_field,far_field_small},0);
            small_field = torch::cat({small_field,small_field_small},0);
            near_field = torch::cat({near_field,near_field_small},0);
        }
        far_field = far_field.index({torch::argsort(far_field.slice(1,0,1).squeeze())});
        small_field = small_field.index({torch::argsort(small_field.slice(1,0,1).squeeze())});
        near_field = near_field.index({torch::argsort(near_field.slice(1,0,1).squeeze())});
    }else{
        interactions = get_new_interactions(ntree_X.depth,ntree_Y.depth,near_field_old,ntree_X.dim_fac,gpu_device); //Doesn't work for new setup since the division is changed...
        interactions = filter_out_interactions(interactions,ntree_X,ntree_Y);
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
    }


    return std::make_tuple(far_field,small_field,near_field);

}

template <typename scalar_t,int nd>
int get_interpolation_rule(scalar_t & effective_edge,int & nr_of_interpolation){
    if (effective_edge<0.01){
        return min((int)pow(3,nd),nr_of_interpolation);
    }
    return nr_of_interpolation;

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
        torch::Tensor effective_far_field_distance_tensor =  ntree_X.edge*ntree_X.edge*ls;
        scalar_t effective_far_field_distance = effective_far_field_distance_tensor.item<scalar_t>();
        int num_nodes = get_interpolation_rule<scalar_t,nd>(effective_far_field_distance,nr_of_interpolation_points);
        std::tie(cheb_data, laplace_combinations, node_list_cum, chebnodes_1D, cheb_w) = smolyak_grid<scalar_t, nd>(
                num_nodes, gpu_device);
        std::tie(interactions_x, interactions_y) = unbind_sort(far_field);
        torch::Tensor unique_x,unique_y;
        std::tie(unique_x,interactions_x_parsed,unique_y) = process_interactions<nd>(interactions_x, interactions_y, gpu_device);
        far_field_compute_v2<scalar_t, nd>( //Very many far field interactions quite fast...
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
                cheb_w,
                unique_x,
                unique_y
        ); //far field compute
    }
    return near_field;
}


template <typename scalar_t, int nd>
torch::Tensor FFM_XY(torch::Tensor &X_data, torch::Tensor &Y_data, torch::Tensor &b, const std::string &gpu_device, scalar_t &ls,
       float &min_points, int &nr_of_interpolation_points, bool &var_compression, scalar_t &eff_var_limit,
       int &small_field_limit) {
    scalar_t lcs = 1/(2*ls*ls);
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
                    lcs,
                    nr_of_interpolation_points,
                    gpu_device,
                    var_compression,
                    eff_var_limit,
                    small_field_limit
            );
        }

        if (near_field.numel()>0){
            near_field_run<scalar_t,nd>(ntree_X,ntree_X,near_field,output,b,lcs,gpu_device);
        }
    }else{
        std::tie(edge,xmin,ymin,xmax,ymax,x_edge,y_edge) = calculate_edge<scalar_t,nd>(X_data,Y_data,gpu_device); //actually calculate them
        n_tree_cuda<scalar_t,nd> ntree_X = n_tree_cuda<scalar_t,nd>(edge,X_data,xmin,xmax,gpu_device);
        n_tree_cuda<scalar_t,nd> ntree_Y = n_tree_cuda<scalar_t,nd>(edge,Y_data,ymin,ymax,gpu_device);
        while (near_field.numel()>0 and (ntree_X.avg_nr_points > min_points and ntree_Y.avg_nr_points > min_points)){

            ntree_X.divide();//needs to be fixed... Should get 451 errors, OK. Memory issue is consistent
            ntree_Y.divide();//needs to be fixed... Should get 451 errors, OK. Memory issue is consistent
            near_field = far_field_run<scalar_t, nd>(
                    ntree_X,
                    ntree_Y,
                    near_field,
                    output,
                    b,
                    lcs,
                    nr_of_interpolation_points,
                    gpu_device,
                    var_compression,
                    eff_var_limit,
                    small_field_limit
            );

        }
        if (near_field.numel()>0){
            near_field_run<scalar_t,nd>(ntree_X,ntree_Y,near_field,output,b,lcs,gpu_device);
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
    FFM_object(torch::Tensor &X_data, torch::Tensor &Y_data, scalar_t &ls, const std::string &gpu_device,
               float &min_points, int &nr_of_interpolation_points, bool &var_comp, scalar_t &eff_var,
               int &small_field_limit)
            : X_data(X_data), Y_data(Y_data), ls(ls), gpu_device(gpu_device), min_points(min_points), nr_of_interpolation_points(nr_of_interpolation_points),
              var_compression(var_comp), eff_var_limit(eff_var), small_field_limit(small_field_limit){
    };
    virtual torch::Tensor operator* (torch::Tensor & b){
        if (X_data.data_ptr()==Y_data.data_ptr()){
            return FFM_XY<scalar_t, nd>(
                    X_data,
                    X_data,
                    b,
                    gpu_device,
                    ls,
                    min_points,
                    nr_of_interpolation_points,
                    var_compression,
                    eff_var_limit,
                    small_field_limit);
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
                    small_field_limit);
        }
    };
};
template <typename scalar_t, int nd>
struct exact_MV{
    torch::Tensor & X_data;
    torch::Tensor & Y_data;
    scalar_t & ls;
    exact_MV( //constructor
            torch::Tensor & X_data,
            torch::Tensor & Y_data,
            scalar_t & ls
    ): X_data(X_data), Y_data(Y_data),ls(ls){
    };
    torch::Tensor operator* (torch::Tensor & b) {
        return  rbf_call<scalar_t,nd>(
                X_data,
                Y_data,
                b,
                ls,
                true
        );
    };
};

