//
// Created by rhu on 2020-03-28.
//

#pragma once
#include "1_d_conv.cu"
#include "utils.h"
#include <vector>
#include "algorithm"
#include "divide_n_conquer.cu"
//template<typename T>

template<typename T>
at::ScalarType dtype() { return at::typeMetaToScalarType(caffe2::TypeMeta::Make<T>()); }

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> calculate_edge(const torch::Tensor &X,const torch::Tensor &Y){
    torch::Tensor Xmin = std::get<0>(X.min(0));
    torch::Tensor Xmax = std::get<0>(X.max(0));
    torch::Tensor Ymin = std::get<0>(Y.min(0));
    torch::Tensor Ymax = std::get<0>(Y.max(0));
    torch::Tensor edge = torch::cat({Xmax - Xmin, Ymax - Ymin}).max();
    return std::make_tuple(edge*1.01,Xmin,Ymin);
};


template <typename scalar_t>
struct n_tree_cuda{
    torch::Tensor &data;
    torch::Tensor edge,xmin,box_indicator,multiply,coord_tensor,sorted_index,unique_counts,box_indices_sorted,centers,tmp,sort_ref;
    std::string device;
    int dim,dim_fac,largest_box_n,depth;
    int *dim_fac_pointer;
    float avg_nr_points;
    n_tree_cuda(torch::Tensor &e, torch::Tensor &d, torch::Tensor &xm, const std::string &cuda_str):data(d){
        device = cuda_str;
        xmin = xm;
        edge = e;
        dim = data.size(1);
        dim_fac = pow(2,dim);
        cudaMalloc((void **)&dim_fac_pointer, sizeof(int));
        cudaMemcpy(dim_fac_pointer, &dim_fac, sizeof(int), cudaMemcpyHostToDevice);
        box_indicator = torch::zeros({data.size(0)}).toType(torch::kInt32).to(device);
        sorted_index = torch::argsort(box_indicator).toType(torch::kInt32);
        box_indices_sorted = torch::tensor(0).to(device);
        unique_counts = torch::tensor(data.size(0)).to(device);
        largest_box_n = unique_counts.max().item<int>();
        avg_nr_points =  unique_counts.toType(torch::kFloat32).mean().item<float>();
        multiply = torch::pow(2,torch::arange(dim).toType(torch::kInt32)).to(device);
        coord_tensor = torch::zeros({dim_fac,dim}).toType(torch::kInt32).to(device);
        get_centers<<<8,192>>>(coord_tensor.packed_accessor32<int,2,torch::RestrictPtrTraits>());
        cudaDeviceSynchronize();
        centers = xmin + 0.5 * edge;
        centers = centers.unsqueeze(0);
        depth = 0;
    }

    void divide(){
        torch::Tensor tmp_o = torch::zeros({box_indicator.size(0)}).toType(torch::kInt32).to(device);
        dim3 blockSize,gridSize;
        int memory;
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(nd, data.size(0));
        box_division<scalar_t><<<gridSize,blockSize>>>(
                data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                centers.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                multiply.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                box_indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                tmp_o.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                dim_fac_pointer
        );
        cudaDeviceSynchronize();
        box_indicator = tmp_o;
        std::tie(sort_ref,sorted_index) = torch::sort(box_indicator);
        std::tie(box_indices_sorted,tmp,unique_counts) = torch::unique_consecutive(sort_ref,false,true);
        unique_counts = unique_counts.toType(torch::kInt32);
        largest_box_n =  unique_counts.max().item<int>();
        avg_nr_points = unique_counts.toType(torch::kFloat32).mean().item<float>();
        sorted_index = sorted_index.toType(torch::kInt32);
        if (depth==0){
            centers = centers.repeat_interleave(dim_fac,0)+ 0.25 * edge * coord_tensor;
        }else{
            int r = pow(dim_fac,depth);
            centers = centers.repeat_interleave(dim_fac,0)+ 0.25 * edge * coord_tensor.repeat({r,1});
        }
        edge = edge*0.5;
        depth += 1;
    };
    std::tuple<torch::Tensor,torch::Tensor> get_box_sorted_data(){
        return std::make_tuple(unique_counts,sorted_index);
    }
};


template <typename scalar_t>
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
        rbf_1d_reduce_shared_torch<scalar_t><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            cuda_Y_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            cuda_b_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            output_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            d_ls
                                                                            );
    }else{
        rbf_1d_reduce_simple_torch<scalar_t><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
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

template <typename scalar_t>
void call_skip_conv(
        torch::Tensor & cuda_X_job,
        torch::Tensor & cuda_Y_job,
        torch::Tensor & cuda_b_job,
        torch::Tensor & output_job,
        scalar_t & ls,
        torch::Tensor & centers_X,
        torch::Tensor & centers_Y,
        torch::Tensor & x_boxes_count,
        torch::Tensor & y_boxes_count,
        torch::Tensor & x_box_idx,
        const std::string & device_gpu,
        torch::Tensor & x_idx_reordering,
        torch::Tensor & y_idx_reordering,
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
        torch::Tensor indicator,box_block_indicator;
        int min_size=x_boxes_count.min().item<int>();
        blkSize = optimal_blocksize(min_size);
        std::tie(blockSize,gridSize,memory,indicator,box_block_indicator) = skip_kernel_launch<scalar_t>(nd,blkSize,x_boxes_count,x_box_idx);
        indicator = indicator.to(device_gpu);
        box_block_indicator = box_block_indicator.to(device_gpu);

        torch::Tensor x_boxes_count_cumulative = torch::cat({torch::zeros({1}).toType(torch::kInt32).to(device_gpu),x_boxes_count.cumsum(0).toType(torch::kInt32)}) ;
        torch::Tensor y_boxes_count_cumulative = torch::cat({torch::zeros({1}).toType(torch::kInt32).to(device_gpu),y_boxes_count.cumsum(0).toType(torch::kInt32)}) ;
        skip_conv_1d_shared<scalar_t><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                     cuda_Y_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                     cuda_b_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                     output_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                     d_ls,
                                                                     centers_X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                     centers_Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                     x_boxes_count_cumulative.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                     y_boxes_count_cumulative.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                     indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                     box_block_indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                     x_idx_reordering.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                     y_idx_reordering.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                     interactions_x_parsed.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                                                     interactions_y.packed_accessor32<int,1,torch::RestrictPtrTraits>()
                                                                     );

    }else{
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(nd, cuda_X_job.size(0));
        x_box_idx = x_box_idx.to(device_gpu);
        torch::Tensor x_boxes_count_cumulative = x_boxes_count.cumsum(0).toType(torch::kInt32).to(device_gpu);
        torch::Tensor y_boxes_count_cumulative = y_boxes_count.cumsum(0).toType(torch::kInt32).to(device_gpu);
        skip_conv_1d<scalar_t><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              cuda_Y_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              cuda_b_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              output_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              d_ls,
                                                              centers_X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              centers_Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              x_boxes_count_cumulative.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                              y_boxes_count_cumulative.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
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

template <typename scalar_t>
void near_field_compute_v2(
                        torch::Tensor & interactions_x_parsed,
                        torch::Tensor & interactions_y,
                        n_tree_cuda<scalar_t> & x_box,
                        n_tree_cuda<scalar_t> & y_box,
                        torch::Tensor & output,
                        torch::Tensor& b,
                        const std::string & device_gpu,
                        scalar_t & ls){

    torch::Tensor b_permuted,
    update,
    x_boxes_ind_count,
    y_boxes_ind_count,
    boolean_interactions,
    x_idx_reordering,
    y_idx_reordering;
    std::tie(x_boxes_ind_count, x_idx_reordering)=x_box.get_box_sorted_data();
    std::tie( y_boxes_ind_count,y_idx_reordering)=y_box.get_box_sorted_data();

    //Fix this mechanic, probably use a different mechanic... ask glaunès!
    torch::Tensor &x_data = x_box.data;
    torch::Tensor &y_data = y_box.data;
    call_skip_conv<scalar_t>(x_data,
                             y_data,
                             b,
                             output,
                             ls,
                             x_box.centers,
                             y_box.centers,
                             x_boxes_ind_count,
                             y_boxes_ind_count,
                                x_box.box_indices_sorted,
                             device_gpu,
                             x_idx_reordering,
                             y_idx_reordering,
                             interactions_x_parsed,
                             interactions_y,
                             true);
};

torch::Tensor chebyshev_nodes_1D(const int & nodes){
    float PI = atan(1.)*4.;
    torch::Tensor chebyshev_nodes = torch::arange(1, nodes+1).toType(torch::kFloat32);
    chebyshev_nodes = cos((chebyshev_nodes*2.-1.)*PI/(2*chebyshev_nodes.size(0)));
    return chebyshev_nodes;
};


template <typename scalar_t>
void apply_laplace_interpolation_v2(
        n_tree_cuda<scalar_t>& n_tree,
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
    memory = memory+laplace_nodes*sizeof(scalar_t);
    torch::Tensor boxes_count_cumulative = torch::cat({torch::zeros({1}).toType(torch::kInt32).to(device_gpu),boxes_count.cumsum(0).toType(torch::kInt32)});

    if (transpose){

        laplace_shared<scalar_t><<<gridSize,blockSize,memory>>>(
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
        laplace_shared_transpose<scalar_t><<<gridSize,blockSize,memory>>>(
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



template <typename scalar_t>
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
    skip_conv_far_boxes_opt<scalar_t><<<gridSize,blockSize,memory>>>(
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
template <typename scalar_t>
void far_field_compute_v2(
                       torch::Tensor & interactions_x_parsed,
                       torch::Tensor & interactions_y,
                       n_tree_cuda<scalar_t> & x_box,
                       n_tree_cuda<scalar_t> & y_box,
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
    apply_laplace_interpolation_v2<scalar_t>(y_box,
                                            b,
                                            device_gpu,
                                            chebnodes_1D,
                                            laplace_combinations,
                                            true,
                                            low_rank_y); //no problems here!
    //Consider limiting number of boxes!!!
    low_rank_y = setup_skip_conv<scalar_t>( //error happens here
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

    apply_laplace_interpolation_v2<scalar_t>(x_box,
                                            low_rank_y,
                                            device_gpu,
                                            chebnodes_1D,
                                            laplace_combinations,
                                            false,
                                            output);
};
torch::Tensor get_new_interactions(torch::Tensor & old_near_interactions, int & p,const std::string & gpu_device){
    int n = old_near_interactions.size(0);
//    std::vector<torch::Tensor> unbound_old = torch::unbind(old_near_interactions.repeat_interleave(p*p,0),1);
    torch::Tensor arr = torch::arange(p).toType(torch::kInt32).to(gpu_device);
//    torch::Tensor test = arr.repeat(p*n);
//    torch::Tensor test_2 = arr.repeat_interleave(p).repeat(n);
//    torch::Tensor interaction_x = p*unbound_old[0] + arr.repeat_interleave(p).repeat(n);
//    torch::Tensor interaction_y = p*unbound_old[1] + arr.repeat(p*n);

    torch::Tensor new_interactions_vec = torch::stack({arr.repeat_interleave(p).repeat(n),arr.repeat(p*n)},1)+p*old_near_interactions.repeat_interleave(p*p,0);
    new_interactions_vec = new_interactions_vec.index({torch::argsort(new_interactions_vec.slice(1,0,1).squeeze()),torch::indexing::Slice()});
//    std::vector<torch::Tensor> unbound = torch::unbind(new_interactions_vec,1);
    return new_interactions_vec;
}

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

template<typename scalar_t>
std::tuple<torch::Tensor,torch::Tensor> parse_cheb_data(
        torch::Tensor & cheb_nodes,
        int d,
        const std::string & gpu_device
        ){
    int n = (int) pow(cheb_nodes.size(0),d);
    torch::Tensor cheb_idx = torch::zeros({n,d}).toType(torch::kInt32).to(gpu_device);
    torch::Tensor cheb_data = torch::zeros({n,d}).toType(dtype<scalar_t>()).to(gpu_device);
    dim3 block,grid;
    int shared;
    std::tie(block,grid,shared) =  get_kernel_launch_params<scalar_t>(d,n);
    get_cheb_idx_data<scalar_t><<<grid,block,shared>>>(
            cheb_nodes.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            cheb_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            cheb_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>()
            );
    cudaDeviceSynchronize();
    return std::make_tuple(cheb_idx,cheb_data);

}

std::tuple<torch::Tensor,torch::Tensor> unbind_nearfield(torch::Tensor & near_field){
    std::vector<torch::Tensor> tmp = near_field.unbind(1);
    return std::make_tuple(tmp[0],tmp[1]);
}

template <typename scalar_t>
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
    boolean_separate_interactions<scalar_t><<<gridSize,blockSize>>>(
            centers_X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            centers_Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            interactions.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            edge.data_ptr<scalar_t>(),
            far_field_mask.packed_accessor32<bool,1,torch::RestrictPtrTraits>()
    );
    cudaDeviceSynchronize();
    return std::make_tuple(interactions.index({far_field_mask}),interactions.index({far_field_mask.logical_not()}));
}


template <typename scalar_t>
torch::Tensor FFM(
        torch::Tensor &X_data,
        torch::Tensor &Y_data,
        torch::Tensor &b,
        const std::string & gpu_device,
        scalar_t & ls
        ) {
    torch::Tensor output = torch::zeros({X_data.size(0),b.size(1)}).to(gpu_device); //initialize empty output
    torch::Tensor edge,xmin,ymin,interactions,far_field,near_field,interactions_x,interactions_y,interactions_x_parsed,cheb_data_X,laplace_combinations; //these are needed to figure out which interactions are near/far field
    std::tie(edge,xmin,ymin) = calculate_edge(X_data,Y_data); //actually calculate them
    n_tree_cuda<scalar_t> ntree_X = n_tree_cuda<scalar_t>(edge,X_data,xmin,gpu_device);
    n_tree_cuda<scalar_t> ntree_Y = n_tree_cuda<scalar_t>(edge,Y_data,ymin,gpu_device);
    near_field = torch::zeros({1,2}).toType(torch::kInt32).to(gpu_device);
    float min_points = 1000;
    torch::Tensor chebnodes_1D = chebyshev_nodes_1D(laplace_nodes).to(gpu_device); //get chebyshev nodes, laplace_nodes is fixed now, should probably be a variable
    std::tie(laplace_combinations,cheb_data_X)=parse_cheb_data<scalar_t>(chebnodes_1D,nd,gpu_device);

    while (near_field.numel()>0 and ntree_X.avg_nr_points > min_points and ntree_Y.avg_nr_points > min_points){
        ntree_X.divide();//needs to be fixed... Should get 451 errors, OK. Memory issue is consistent
        ntree_Y.divide();//divide ALL boxes recursively once
        interactions = get_new_interactions(near_field,ntree_X.dim_fac,gpu_device); //fix this. it does something funny
        std::tie(far_field,near_field) =
                separate_interactions<scalar_t>(
                interactions,
                ntree_X.centers,
                ntree_Y.centers,
                ntree_X.edge,
                gpu_device);
        if (far_field.numel()>0) {
            torch::Tensor cheb_data = cheb_data_X*ntree_X.edge/2.+ntree_X.edge/2.;
            std::tie(interactions_x,interactions_y) = unbind_nearfield(far_field);
            interactions_x_parsed = process_interactions(interactions_x,ntree_X.centers.size(0),gpu_device);
            far_field_compute_v2<scalar_t>( //Very many far field interactions quite fast...
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
        }

    if (near_field.numel()>0){
        std::tie(interactions_x,interactions_y) = unbind_nearfield(near_field);
        interactions_x_parsed = process_interactions(interactions_x,ntree_X.centers.size(0),gpu_device);
        near_field_compute_v2<scalar_t>(interactions_x_parsed,interactions_y,ntree_X, ntree_Y, output, b, gpu_device,ls); //Make sure this thing works first!
    }
    return output;
}
template <typename scalar_t>
struct FFM_object{
    torch::Tensor & X_data;
    torch::Tensor & Y_data;
    scalar_t & ls;
    scalar_t &lambda;
    const std::string & gpu_device;

    FFM_object( //constructor
            torch::Tensor & X_data,
    torch::Tensor & Y_data,
    scalar_t & ls,
    scalar_t &lambda,
    const std::string & gpu_device): X_data(X_data), Y_data(Y_data),ls(ls),lambda(lambda),gpu_device(gpu_device){
    };
    virtual torch::Tensor operator* (torch::Tensor & b){
        return FFM<scalar_t>(
                X_data,
                Y_data,
                b,
                gpu_device,
                ls
        )+b*lambda;
    };
};
template <typename scalar_t>
struct exact_MV : FFM_object<scalar_t>{
    exact_MV(torch::Tensor & X_data,
                         torch::Tensor & Y_data,
                         scalar_t & ls,
                         scalar_t &lambda,
                         const std::string & gpu_device): FFM_object<scalar_t>(X_data, Y_data, ls, lambda, gpu_device){};
    torch::Tensor operator* (torch::Tensor & b){
        torch::Tensor output = torch::zeros({FFM_object<scalar_t>::X_data.size(0), b.size(1)}).to(FFM_object<scalar_t>::gpu_device);
        rbf_call<scalar_t>(
                FFM_object<scalar_t>::X_data,
                FFM_object<scalar_t>::Y_data,
                b,
                output,
                FFM_object<scalar_t>::ls,
                true
        );
        output = output+ b * FFM_object<scalar_t>::lambda;
        return output;
    };
};

template<typename scalar_t>
std::tuple<torch::Tensor,torch::Tensor> CG(FFM_object<scalar_t> & MV, torch::Tensor &b, float & tol, int & max_its, bool tridiag){
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
template <typename scalar_t>
std::tuple<torch::Tensor,torch::Tensor> trace_and_log_det_calc(FFM_object<scalar_t> &MV, FFM_object<scalar_t> &MV_grad, int& T, int &max_its, float &tol){
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

template<typename scalar_t>
torch::Tensor ls_grad_calculate(FFM_object<scalar_t> &MV_grad, torch::Tensor & b_sol, torch::Tensor &trace_est){
    return trace_est + torch::sum(b_sol*(MV_grad*b_sol));
}

torch::Tensor GP_loss(torch::Tensor &log_det,torch::Tensor &b_sol,torch::Tensor &b){
    return log_det - torch::sum(b*b_sol);
}
template<typename scalar_t>
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> calculate_loss_and_grad(FFM_object<scalar_t> &MV,
                                                                              FFM_object<scalar_t> &MV_grad,
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
