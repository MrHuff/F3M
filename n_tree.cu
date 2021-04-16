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
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> calculate_edge(const torch::Tensor &X,const torch::Tensor &Y,const std::string & gpu_device){
    torch::Tensor Xmin = torch::ones(X.size(1)).toType(dtype<scalar_t>()).to(gpu_device)*NPP_MAXABS_32F;
    torch::Tensor Xmax = -torch::ones(X.size(1)).toType(dtype<scalar_t>()).to(gpu_device)*NPP_MAXABS_32F;
    torch::Tensor Ymin = torch::ones(X.size(1)).toType(dtype<scalar_t>()).to(gpu_device)*NPP_MAXABS_32F;
    torch::Tensor Ymax = -torch::ones(X.size(1)).toType(dtype<scalar_t>()).to(gpu_device)*NPP_MAXABS_32F;
    dim3 blockSize,gridSize;
    blockSize.x = 1024;
    gridSize.x = 20;
    reduceMaxMinOptimizedWarpMatrix<scalar_t,nd><<<gridSize,blockSize,64>>>(X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                        Xmax.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                                                                        Xmin.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();
    reduceMaxMinOptimizedWarpMatrix<scalar_t,nd><<<gridSize,blockSize,64>>>(Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                        Ymax.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                                                                        Ymin.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();
    torch::Tensor edge = torch::cat({Xmax - Xmin, Ymax - Ymin}).max();
    return std::make_tuple(edge*1.01,Xmin,Ymin,Xmax,Ymax);
};

template<typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> calculate_edge_debug(const torch::Tensor &X,const torch::Tensor &Y,const std::string & gpu_device){
    torch::Tensor Xmin,Ymin,Xmax,Ymax,tmp;
    std::tie(Xmin,tmp) = X.min(0);
    std::tie(Xmax,tmp) = X.max(0);
    std::tie(Ymin,tmp) = Y.min(0);
    std::tie(Ymax,tmp) = Y.max(0);
    torch::Tensor edge = torch::cat({Xmax - Xmin, Ymax - Ymin}).max();
    return std::make_tuple(edge*1.01,Xmin,Ymin,Xmax,Ymax);
};

template<typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> calculate_edge_X(const torch::Tensor &X,const std::string & gpu_device){
    torch::Tensor Xmin = torch::ones(nd).toType(dtype<scalar_t>()).to(gpu_device)*NPP_MAXABS_32F;
    torch::Tensor Xmax = -torch::ones(nd).toType(dtype<scalar_t>()).to(gpu_device)*NPP_MAXABS_32F;
    dim3 blockSize,gridSize;
    blockSize.x = 1024;
    gridSize.x = 10;
    reduceMaxMinOptimizedWarpMatrix<scalar_t,nd><<<gridSize,blockSize,64>>>(X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                                    Xmax.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                                                                                            Xmin.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
                            );
    cudaDeviceSynchronize();
    torch::Tensor edge = (Xmax - Xmin).max();
    return std::make_tuple(edge*1.01,Xmin,Xmax);
};

template<typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> calculate_edge_X_debug(const torch::Tensor &X,const std::string & gpu_device){
    torch::Tensor Xmin,Xmax,tmp;
    std::tie(Xmin,tmp) = X.min(0);
    std::tie(Xmax,tmp) = X.max(0);
    torch::Tensor edge = (Xmax - Xmin).max();
    return std::make_tuple(edge*1.01,Xmin,Xmax);
};


template <typename scalar_t,int dim>
struct n_tree_cuda{
    torch::Tensor &data;
    torch::Tensor edge,edge_og,
    xmin,
    xmax,
    side_base,
    side,
    sorted_index,
    unique_counts,
    unique_counts_og,
    multiply_gpu_base,
    multiply_gpu,
    box_indices_sorted,
    centers,
    non_empty_mask,
    box_idxs,
    unique_counts_cum,
    coord_tensor,
    perm;
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
        sorted_index  = torch::zeros({data.size(0)}).toType(torch::kInt32).contiguous().to(device);
        avg_nr_points =  (float)d.size(0);
        multiply_gpu_base = torch::pow(2, torch::arange(dim - 1, -1, -1).toType(torch::kInt32)).to(device);
        multiply_gpu = multiply_gpu_base;
        coord_tensor = torch::zeros({dim_fac,dim}).toType(torch::kInt32).to(device);
        get_centers<dim><<<8,192>>>(coord_tensor.packed_accessor32<int,2,torch::RestrictPtrTraits>());
        cudaDeviceSynchronize();
        centers = xmin + 0.5 * edge;
        centers = centers.unsqueeze(0);
        depth = 0;
        side_base = torch::tensor(2.0).toType(dtype<scalar_t>()).to(device);
        box_idxs = torch::arange(centers.size(0)).toType(torch::kInt32).contiguous().to(device);

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
        box_idxs = torch::arange(centers.size(0)).toType(torch::kInt32).contiguous().to(device);
        unique_counts_cum = torch::zeros(centers.size(0)+1).toType(torch::kInt32).contiguous().to(device);//matrix -> existing boxes * 2^dim bounded over nr or points... betting on boxing dissapears...
        unique_counts= torch::zeros(centers.size(0)).toType(torch::kInt32).contiguous().to(device);
        perm = torch::zeros(centers.size(0)).toType(torch::kInt32).contiguous().to(device);
        dim3 blockSize,gridSize;
        int memory;
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(dim, centers.size(0));
        center_perm<scalar_t,dim><<<gridSize,blockSize>>>(
                        centers.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                        xmin.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                        multiply_gpu.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                        side.data_ptr<scalar_t>(),
                        edge_og.data_ptr<scalar_t>(),
                        perm.packed_accessor32<int,1,torch::RestrictPtrTraits>()
                                ); //Apply same hack but to centers to get perm

        cudaDeviceSynchronize();
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(dim, data.size(0));
        box_division_cum<scalar_t,dim><<<gridSize,blockSize>>>(
                data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                xmin.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                multiply_gpu.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                side.data_ptr<scalar_t>(),
                edge_og.data_ptr<scalar_t>(),
                unique_counts_cum.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                perm.packed_accessor32<int,1,torch::RestrictPtrTraits>()
        );
        cudaDeviceSynchronize();
        unique_counts_cum = unique_counts_cum.cumsum(0).toType(torch::kInt32);
        box_division_assign<scalar_t,dim><<<gridSize,blockSize>>>(
                data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                xmin.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                multiply_gpu.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                side.data_ptr<scalar_t>(),
                edge_og.data_ptr<scalar_t>(),
                unique_counts_cum.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                perm.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                unique_counts.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                sorted_index.packed_accessor32<int,1,torch::RestrictPtrTraits>()

        );
        cudaDeviceSynchronize();
        non_empty_mask = unique_counts != 0;
        box_indices_sorted = box_idxs.index({non_empty_mask});
        unique_counts_og = unique_counts;
        unique_counts = unique_counts.index({non_empty_mask});
        avg_nr_points = unique_counts.toType(torch::kFloat32).max().item<float>();
        multiply_gpu = multiply_gpu*multiply_gpu_base;

    };
    std::tuple<torch::Tensor,torch::Tensor> get_box_sorted_data(){
        return std::make_tuple(unique_counts,sorted_index);
    }
};




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
    memory = memory+2*nodes.size(0)*sizeof(scalar_t)+(nd+1)*sizeof(int); //Seems the last write is where the trouble is...
    torch::Tensor boxes_count_cumulative = n_tree.unique_counts_cum;
    if (transpose){

        lagrange_shared<scalar_t, nd><<<gridSize, blockSize, memory>>>(
                data.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                nodes.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                laplace_indices.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                indicator.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                box_block.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                boxes_count_cumulative.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                centers.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                edge.data_ptr<scalar_t>(),
                idx_reordering.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                node_list_cum.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                cheb_w.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
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
                idx_reordering.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                node_list_cum.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                cheb_w.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
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
                        torch::Tensor & cheb_data_X,
                        torch::Tensor & node_list_cum,
                       torch::Tensor & cheb_w
){
    torch::Tensor low_rank_y;
    low_rank_y = torch::zeros({cheb_data_X.size(0)*x_box.centers.size(0),b.size(1)}).toType(dtype<scalar_t>()).to(device_gpu);
    apply_laplace_interpolation_v2<scalar_t,nd>(y_box,
                                            b,
                                            device_gpu,
                                            chebnodes_1D,
                                            laplace_combinations,
                                                node_list_cum,
                                                cheb_w,
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
                                                node_list_cum,
                                                cheb_w,
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
            cheb_nodes.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            cheb_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            cheb_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            nodes_per_dim.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            nodes_cum_prod.packed_accessor32<int,1,torch::RestrictPtrTraits>()
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
            cheb_nodes.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            cheb_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            cheb_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            sampled_indices.packed_accessor32<int,1,torch::RestrictPtrTraits>()
            );
    cudaDeviceSynchronize();
    return std::make_tuple(cheb_idx,cheb_data);

}

std::tuple<torch::Tensor,torch::Tensor> unbind_sort(torch::Tensor & interactions){
    interactions = interactions.index({torch::argsort(interactions.slice(1,0,1).squeeze()),torch::indexing::Slice()});
    if (interactions.dim()<2){
        interactions = interactions.unsqueeze(0);
    }
    std::vector<torch::Tensor> tmp = interactions.unbind(1);
    return std::make_tuple(tmp[0],tmp[1]);
}

template <typename scalar_t, int nd>
std::tuple<torch::Tensor,torch::Tensor> separate_interactions(
        torch::Tensor & interactions,
        torch::Tensor & centers_X,
        torch::Tensor & centers_Y,
        torch::Tensor & edge,
        scalar_t & ls,
        const std::string & gpu_device
){
    dim3 blockSize,gridSize;
    int mem,n;
    scalar_t *d_ls;
    cudaMalloc((void **)&d_ls, sizeof(scalar_t));
    cudaMemcpy(d_ls, &ls, sizeof(scalar_t), cudaMemcpyHostToDevice);
    n = interactions.size(0);
    std::tie(blockSize,gridSize,mem)=get_kernel_launch_params<scalar_t>(nd,n);
    torch::Tensor far_field_mask = torch::zeros({n}).toType(torch::kBool).to(gpu_device);
    torch::Tensor impact = torch::zeros({n}).toType(dtype<scalar_t>()).to(gpu_device);
    boolean_separate_interactions<scalar_t,nd><<<gridSize,blockSize>>>(
            centers_X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            centers_Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            interactions.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            edge.data_ptr<scalar_t>(),
            far_field_mask.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
            impact.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            d_ls
    );
    cudaDeviceSynchronize();
//    torch::Tensor far_impact = impact.index({far_field_mask});
//    scalar_t largest_effect = far_impact.max().item<scalar_t>();
    return std::make_tuple(interactions.index({far_field_mask}),interactions.index({far_field_mask.logical_not()}));
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
            interactions.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            x_keep.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
            y_keep.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
            mask.packed_accessor32<bool,1,torch::RestrictPtrTraits>()
            );
    return interactions.index({mask});

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
torch::Tensor far_field_run(
                            n_tree_cuda<scalar_t,nd> & ntree_X,
                            n_tree_cuda<scalar_t,nd> & ntree_Y,
                            torch::Tensor & near_field,
                            torch::Tensor & output,
                            torch::Tensor & b,
                            scalar_t & ls,
                            int & nr_of_interpolation_points,
                            const std::string & gpu_device
){
    torch::Tensor interactions,far_field,interactions_x,interactions_y,interactions_x_parsed,cheb_data_X,
            laplace_combinations,
            chebnodes_1D,
            node_list_cum,
            cheb_w;

    std::tie(cheb_data_X,laplace_combinations,node_list_cum,chebnodes_1D,cheb_w) = smolyak_grid<scalar_t,nd>(nr_of_interpolation_points,gpu_device);
    interactions = get_new_interactions(near_field,ntree_X.dim_fac,gpu_device); //Doesn't work for new setup since the division is changed...
    interactions = filter_out_interactions(interactions,ntree_X,ntree_Y);
    std::tie(far_field,near_field) =
            separate_interactions<scalar_t,nd>(
                    interactions,
                    ntree_X.centers,
                    ntree_Y.centers,
                    ntree_X.edge,
                    ls,
                    gpu_device);
    if (far_field.numel()>0) {
        //Normal 0 1 has really bad precision for some reason... also removing the far_field doesn't work here... WHY?!
//        far_field = filter_out_interactions(far_field,ntree_X,ntree_Y);
//        std::cout<<"near field: "<<near_field.size(0)<<std::endl;
//        std::cout<<"far field: "<<far_field.size(0)<<std::endl;
// Add a counter to check for interactions! Either its not scaling linearly or more interactions. Average nr of points per box.
        torch::Tensor cheb_data = cheb_data_X*ntree_X.edge/2.+ntree_X.edge/2.; //scaling lagrange nodes to edge scale
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
            x_dat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            box_ind.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            x_box_cum.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            big_enough.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            box_variance_1.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();
    return box_variance_1;

}

template <typename scalar_t, int nd>
torch::Tensor far_field_run_XX(
        n_tree_cuda<scalar_t,nd> & ntree_X,
        torch::Tensor & near_field,
        torch::Tensor & output,
        torch::Tensor & b,
        scalar_t & ls,
        int & nr_of_interpolation_points,
        const std::string & gpu_device,
        bool & var_compression,
        scalar_t  & eff_var_limit
){
    torch::Tensor interactions,far_field,interactions_x,interactions_y,interactions_x_parsed,cheb_data_X,
            laplace_combinations,
            chebnodes_1D,
            node_list_cum,
            cheb_w;

    std::tie(cheb_data_X,laplace_combinations,node_list_cum,chebnodes_1D,cheb_w) = smolyak_grid<scalar_t,nd>(nr_of_interpolation_points,gpu_device);
    interactions = get_new_interactions(near_field,ntree_X.dim_fac,gpu_device); //Doesn't work for new setup since the division is changed...
    interactions = filter_out_interactions(interactions,ntree_X,ntree_X);
    std::tie(far_field,near_field) =
            separate_interactions<scalar_t,nd>(
                    interactions,
                    ntree_X.centers,
                    ntree_X.centers,
                    ntree_X.edge,
                    ls,
                    gpu_device);
    if (var_compression){
        std::vector<torch::Tensor> vec_near_field = near_field.unbind(1);
        torch::Tensor boolean_tensor = vec_near_field[0]!=vec_near_field[1];
        torch::Tensor non_pair = near_field.index({boolean_tensor}); // get non  pair interactions
        torch::Tensor pair = vec_near_field[0].index({torch::logical_not(boolean_tensor)} ); // get non  pair interactions
        torch::Tensor bool_big_enough = ntree_X.unique_counts_og.index({pair.toType(torch::kLong)})>=10*nr_of_interpolation_points;
        torch::Tensor big_enough_boxes  = pair.index({bool_big_enough});
        if (big_enough_boxes.numel()>0){
            torch::Tensor append_back_1 = pair.index({torch::logical_not(bool_big_enough)}).unsqueeze(-1).repeat({1,2});
            torch::Tensor x_var = get_low_variance_pairs<scalar_t,nd>(ntree_X,big_enough_boxes);
            torch::Tensor bool_effective_variance = (x_var.sum(1)/ls)<=eff_var_limit;
            torch::Tensor append_back_2 = big_enough_boxes.index({torch::logical_not(bool_effective_variance)}).unsqueeze(-1).repeat({1,2});
            torch::Tensor smooth_field =big_enough_boxes.index({bool_effective_variance}).unsqueeze(-1).repeat({1,2});
            near_field = torch::cat({non_pair,append_back_1,append_back_2},0);
            far_field = torch::cat({far_field,smooth_field});
        };
    }


    if (far_field.numel()>0) {
        torch::Tensor cheb_data = cheb_data_X*ntree_X.edge/2.+ntree_X.edge/2.; //scaling lagrange nodes to edge scale
        std::tie(interactions_x,interactions_y) = unbind_sort(far_field);
        interactions_x_parsed = process_interactions<nd>(interactions_x,ntree_X.centers.size(0),gpu_device);
        far_field_compute_v2<scalar_t,nd>( //Very many far field interactions quite fast...
                interactions_x_parsed,
                interactions_y,
                ntree_X,
                ntree_X,
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
//    float work_rate = (float)near_field.size(0)/((float)ntree_X.unique_counts.size(0)*ntree_X.unique_counts.size(0));
//    std::cout<<work_rate<<std::endl;
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
        float &min_points,
        int & nr_of_interpolation_points
) {
    torch::Tensor output = torch::zeros({X_data.size(0),b.size(1)}).to(gpu_device); //initialize empty output
    torch::Tensor edge,
    xmin,
    ymin,
    xmax,
    ymax,
    near_field;
    near_field = torch::zeros({1,2}).toType(torch::kInt32).to(gpu_device);
    std::tie(edge,xmin,ymin,xmax,ymax) = calculate_edge<scalar_t,nd>(X_data,Y_data,gpu_device); //actually calculate them
//    std::tie(edge,xmin,ymin,xmax,ymax) = calculate_edge_debug<scalar_t,nd>(X_data,Y_data,gpu_device); //actually calculate them
    n_tree_cuda<scalar_t,nd> ntree_X = n_tree_cuda<scalar_t,nd>(edge,X_data,xmin,xmax,gpu_device);
    n_tree_cuda<scalar_t,nd> ntree_Y = n_tree_cuda<scalar_t,nd>(edge,Y_data,ymin,ymax,gpu_device);

    while (near_field.numel()>0 and ntree_X.avg_nr_points > min_points and ntree_Y.avg_nr_points > min_points){
        ntree_X.divide();//needs to be fixed... Should get 451 errors, OK. Memory issue is consistent
        ntree_Y.divide();//divide ALL boxes recursively once
        near_field = far_field_run<scalar_t,nd>(ntree_X,
                ntree_Y,
                near_field,
                output,
                b,
                ls,
                nr_of_interpolation_points,
                gpu_device);

        }

    if (near_field.numel()>0){
        near_field_run<scalar_t,nd>(ntree_X,ntree_Y,near_field,output,b,ls,gpu_device);
    }
    return output;
}



template <typename scalar_t, int nd>
torch::Tensor FFM_X(
        torch::Tensor &X_data,
        torch::Tensor &b,
        const std::string & gpu_device,
        scalar_t & ls,
        float &min_points,
        int & nr_of_interpolation_points,
        bool &var_compression,
        scalar_t  & eff_var_limit
) {

    torch::Tensor output = torch::zeros({X_data.size(0),b.size(1)}).toType(dtype<scalar_t>()).to(gpu_device); //initialize empty output
    torch::Tensor edge,
    xmin,
    xmax,
    near_field;//these are needed to figure out which interactions are near/far field
    near_field = torch::zeros({1,2}).toType(torch::kInt32).to(gpu_device);
    std::tie(edge,xmin,xmax) = calculate_edge_X<scalar_t,nd>(X_data,gpu_device); //actually calculate them
    n_tree_cuda<scalar_t,nd> ntree_X = n_tree_cuda<scalar_t,nd>(edge,X_data,xmin,xmax,gpu_device);
    while (near_field.numel()>0 and ntree_X.avg_nr_points > min_points){
        ntree_X.divide();//needs to be fixed... Should get 451 errors, OK. Memory issue is consistent
        near_field = far_field_run_XX<scalar_t,nd>(
                    ntree_X,
                    near_field,
                    output,
                    b,
                    ls,
                    nr_of_interpolation_points,
                    gpu_device,
                    var_compression,
                    eff_var_limit
                    );
    }

    if (near_field.numel()>0){
        near_field_run<scalar_t,nd>(ntree_X,ntree_X,near_field,output,b,ls,gpu_device);

    }
    return output;
}

template <typename scalar_t, int nd>
torch::Tensor SUPERSMOOTH_FFM_X(
        torch::Tensor &X_data,
        torch::Tensor &b,
        const std::string & gpu_device,
        scalar_t & ls,
        float &min_points,
        int & nr_of_interpolation_points,
        scalar_t  & eff_var_limit
) {

    torch::Tensor output = torch::zeros({X_data.size(0),b.size(1)}).toType(dtype<scalar_t>()).to(gpu_device); //initialize empty output
    torch::Tensor edge,
            xmin,
            xmax,
            interactions,
    interactions_x,interactions_y,interactions_x_parsed,cheb_data_X,
            laplace_combinations,
            chebnodes_1D,
            node_list_cum,
            cheb_w;
            ;//these are needed to figure out which interactions are near/far field
    interactions = torch::zeros({1,2}).toType(torch::kInt32).to(gpu_device);
    std::tie(edge,xmin,xmax) = calculate_edge_X<scalar_t,nd>(X_data,gpu_device); //actually calculate them
    n_tree_cuda<scalar_t,nd> ntree_X = n_tree_cuda<scalar_t,nd>(edge,X_data,xmin,xmax,gpu_device);
    float cur_var = 1e6;

    while (cur_var>=eff_var_limit){
        ntree_X.divide();//needs to be fixed... Should get 451 errors, OK. Memory issue is consistent
        interactions = get_new_interactions(interactions,ntree_X.dim_fac,gpu_device); //Doesn't work for new setup since the division is changed...
        interactions = filter_out_interactions(interactions,ntree_X,ntree_X);
        torch::Tensor x_var = get_low_variance_pairs<scalar_t,nd>(ntree_X,ntree_X.box_indices_sorted);
        cur_var  = x_var.sum(1).max().item<float>();
    };

    std::tie(cheb_data_X,laplace_combinations,node_list_cum,chebnodes_1D,cheb_w) = smolyak_grid<scalar_t,nd>(nr_of_interpolation_points,gpu_device);
    torch::Tensor cheb_data = cheb_data_X*ntree_X.edge/2.+ntree_X.edge/2.; //scaling lagrange nodes to edge scale
    std::tie(interactions_x,interactions_y) = unbind_sort(interactions);
    interactions_x_parsed = process_interactions<nd>(interactions_x,ntree_X.centers.size(0),gpu_device);
    far_field_compute_v2<scalar_t,nd>( //Very many far field interactions quite fast...
            interactions_x_parsed,
            interactions_y,
            ntree_X,
            ntree_X,
            output,
            b,
            gpu_device,
            ls,
            chebnodes_1D,
            laplace_combinations,
            cheb_data,
            node_list_cum,
            cheb_w
    ); //far field comput

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
    bool & smooth_flag;

    FFM_object( //constructor
            torch::Tensor & X_data,
    torch::Tensor & Y_data,
    scalar_t & ls,
    const std::string & gpu_device,
    float & min_points,
    int & nr_of_interpolation_points,
    bool & var_comp,
    scalar_t  & eff_var,
    bool & smooth_flag
    ): X_data(X_data), Y_data(Y_data),ls(ls),gpu_device(gpu_device),min_points(min_points),nr_of_interpolation_points(nr_of_interpolation_points),
    var_compression(var_comp),eff_var_limit(eff_var),smooth_flag(smooth_flag){
    };
    virtual torch::Tensor operator* (torch::Tensor & b){
        if (X_data.data_ptr()==Y_data.data_ptr()){

            if (smooth_flag){
                return SUPERSMOOTH_FFM_X<scalar_t,nd>(
                        X_data,
                        b,
                        gpu_device,
                        ls,
                        min_points,
                        nr_of_interpolation_points,
                        eff_var_limit
                );
            }else{
                return FFM_X<scalar_t,nd>(
                        X_data,
                        b,
                        gpu_device,
                        ls,
                        min_points,
                        nr_of_interpolation_points,
                        var_compression,
                        eff_var_limit
                );
            }


        }else{
            return FFM_XY<scalar_t,nd>(
                    X_data,
                    Y_data,
                    b,
                    gpu_device,
                    ls,
                    min_points,
                    nr_of_interpolation_points
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
                        bool & smooth_flag

                    )
                         : FFM_object<scalar_t,nd>(X_data, Y_data, ls, gpu_device,min_points,nr_of_interpolation_points,var_compression,eff_var_limit,smooth_flag){};
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


