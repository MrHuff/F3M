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

template <typename scalar_t>
void update_2d_rows_cpu(
        torch::Tensor &original,
        torch::Tensor &update,
        torch::Tensor &rows){
    int col = original.size(1);
    int n = rows.size(0);
    auto o_accessor = original.accessor<scalar_t,2>();
    auto row_accessor = rows.accessor<long,1>();
    auto u_accessor = update.accessor<scalar_t,2>();
    for (int i = 0;i<n;i++){
        for (int j=0; j<col;j++) {
            o_accessor[row_accessor[i]][j]+=u_accessor[i][j];
        }
    }
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> calculate_edge(const torch::Tensor &X,const torch::Tensor &Y){
    torch::Tensor Xmin = std::get<0>(X.min(0));
    torch::Tensor Xmax = std::get<0>(X.max(0));
    torch::Tensor Ymin = std::get<0>(Y.min(0));
    torch::Tensor Ymax = std::get<0>(Y.max(0));
    torch::Tensor edge = torch::cat({Xmax - Xmin, Ymax - Ymin}).max();
    return std::make_tuple(edge*1.01,Xmin,Ymin);
};

template<typename T>
T pop_front_get(std::vector<T> &v)
{
    if (!v.empty()) {
        T t = v.front();
        v.erase(v.begin());
        return t;
    }
}

std::vector<torch::Tensor> recursive_center(std::vector<torch::Tensor> &input, std::vector<torch::Tensor> &memory){
    if (input.empty()){
        return memory;
    }
    else{
        const torch::Tensor &cut = pop_front_get(input);
        torch::Tensor anti_cut = -1*cut;
        if (memory.empty()){
            memory.push_back(cut);
            memory.push_back(anti_cut);
            return recursive_center(input,memory);
        }else{
            std::vector<torch::Tensor> mem_copy = memory;
            int d = mem_copy.size();
            for (int i = 0; i < d; i++) {
                memory[i] = torch::cat({memory[i],cut});
            }
            for (int i = 0; i < d; i++) {
                memory.push_back(torch::cat({mem_copy[i],anti_cut}));
            }
            return recursive_center(input,memory);
        }
    }
};


std::vector<torch::Tensor> recursive_divide(std::vector<torch::Tensor> &input, std::vector<torch::Tensor> &memory){
    if (input.empty()){
        return memory;
    }
    else{
        const torch::Tensor &cut = pop_front_get(input);
        torch::Tensor anti_cut =torch::logical_not(cut);
        if (memory.empty()){
            memory.push_back(cut);
            memory.push_back(anti_cut);
            return recursive_divide(input,memory);
        }else{
            std::vector<torch::Tensor> mem_copy = memory;
            int d = mem_copy.size();
            for (int i = 0; i < d ; i++) {
                memory[i] = memory[i] * cut;
            }
            for (int i = 0;  i < d; i++) {
                memory.push_back(mem_copy[i] * anti_cut);
            }
            return recursive_divide(input,memory);
        }
    }

};

struct n_tree { //might want not have indexing in these to save memory and just the actual points...
    int index;
    int n_elems;
    torch::Tensor row_indices;
    torch::Tensor center;
};
std::ostream& operator<<(std::ostream& os, const n_tree& v)
{
    os<< "cube: "<<v.index<<" n_lemens: "<<v.n_elems<<" center: "<<'\n';
    os<<v.center<<'\n';
    return os;
}
template <typename scalar_t>
struct n_tree_cuda{
    torch::Tensor &data;
    torch::Tensor edge,xmin,box_indicator,multiply,coord_tensor,sorted_index,unique_counts,box_indices_sorted,centers,tmp;
    std::vector<torch::Tensor> output_coord = {};
    std::vector<torch::Tensor> ones = {};
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
        box_indices_sorted = torch::tensor(0);
        unique_counts = torch::tensor(data.size(0));
        largest_box_n = unique_counts.max().item<int>();
        avg_nr_points =  unique_counts.toType(torch::kFloat32).mean().item<float>();
        multiply = torch::pow(2,torch::arange(dim).toType(torch::kInt32)).to(device);
        for (int i=0;i<dim;i++){
            ones.push_back(-1*torch::ones(1));
        }
        output_coord = recursive_center(ones,output_coord);
        coord_tensor = torch::stack(output_coord,0).to(device);
        centers = xmin + 0.5 * edge;
        centers = centers.unsqueeze(0);
        depth = 0;
    }

    torch::Tensor get_centers_expanded(){
        return centers.repeat_interleave(unique_counts.toType(torch::kLong),0);
    }

    void divide(){
        torch::Tensor tmp_o = torch::zeros_like(box_indicator);
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
        sorted_index = torch::argsort(box_indicator).toType(torch::kInt32);
        std::tie(box_indices_sorted,tmp,unique_counts) = torch::_unique2(box_indicator,true,false,true);
        unique_counts = unique_counts.toType(torch::kInt32);
        largest_box_n =  unique_counts.max().item<int>();
        avg_nr_points = unique_counts.toType(torch::kFloat32).mean().item<float>();
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

    std::tuple<torch::Tensor,torch::Tensor> distance(const n_tree_cuda& other, const torch::Tensor &interactions ) { //give all interactions, i.e. cartesian product of indices
        torch::Tensor l1_distances = other.centers.index(interactions.slice(1,1,2).squeeze())-centers.index(interactions.slice(1,0,1).squeeze());
        return std::make_tuple(l1_distances.pow(2).sum(1).sqrt() ,l1_distances);
    }
    std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> far_and_near_field(
            const torch::Tensor & square_dist,
            const torch::Tensor &interactions,
            const torch::Tensor &L1_dist) const{
        torch::Tensor far_field = square_dist>=(edge*2+1e-6);
        return std::make_tuple(interactions.index({far_field}),
                               interactions.index({torch::logical_not(far_field)}),
                               L1_dist.index({far_field}),L1_dist.index({torch::logical_not(far_field)}));
    }

};


template <typename scalar_t>
void rbf_call(
        torch::Tensor & cuda_X_job,
        torch::Tensor & cuda_Y_job,
        torch::Tensor & cuda_b_job,
        torch::Tensor & output_job,
        scalar_t & ls,
        rbf_pointer<scalar_t> & op,
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
                                                                            d_ls,
                                                                            op
                                                                            );
    }else{
        rbf_1d_reduce_simple_torch<scalar_t><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            cuda_Y_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            cuda_b_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            output_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                            d_ls,
                                                                            op
                                                                            );
    }
    cudaDeviceSynchronize();
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

template <typename scalar_t>
void call_skip_conv(
        torch::Tensor & cuda_X_job,
        torch::Tensor & cuda_Y_job,
        torch::Tensor & cuda_b_job,
        torch::Tensor & output_job,
        scalar_t & ls,
        rbf_pointer<scalar_t> & op,
        torch::Tensor & centers_X,
        torch::Tensor & centers_Y,
        torch::Tensor & edge,
        torch::Tensor & x_boxes_count,
        torch::Tensor & y_boxes_count,
        torch::Tensor & x_box_idx,
        const std::string & device_gpu,
        torch::Tensor & x_idx_reordering,
        torch::Tensor & y_idx_reordering,
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
                                                                     op,
                                                                     centers_X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                     centers_Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                     x_boxes_count_cumulative.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                     y_boxes_count_cumulative.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                     indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                     box_block_indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                     x_idx_reordering.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                     y_idx_reordering.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                                     edge.data_ptr<scalar_t>());

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
                                                              op,
                                                              centers_X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              centers_Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                              x_boxes_count_cumulative.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                              y_boxes_count_cumulative.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                              x_box_idx.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                                              edge.data_ptr<scalar_t>());
    } //fix types...
    cudaDeviceSynchronize();
}

//Could represent indices with [box_id, nr of points], make sure its all concatenated correctly and in order.
//local variable: x_i, box belonging. How to get loading scheme. [nr of points].cumsum().  Iteration schedual...
// calculate box belonging.


torch::Tensor get_boolean_2d_mask(torch::Tensor & near_field_interactions,
                                    int nx_box,
                                    int ny_box){
    torch::Tensor boolean_mask = torch::zeros({nx_box,ny_box}).toType(torch::kBool);
    auto accessor_bool =boolean_mask.accessor<bool,2>();
    auto accessor_interactions =near_field_interactions.accessor<long,2>();
    int n_int = near_field_interactions.size(0);
    for (int i=0;i<n_int;i++){
        accessor_bool[accessor_interactions[i][0]][accessor_interactions[i][1]]=true;
    };
    return boolean_mask;
}

template <typename scalar_t>
void near_field_compute_v2(torch::Tensor & near_field_interactions,
                        n_tree_cuda<scalar_t> & x_box,
                        n_tree_cuda<scalar_t> & y_box,
                        torch::Tensor & output,
                        torch::Tensor& b,
                        const std::string & device_gpu,
                        scalar_t & ls,
                        rbf_pointer<scalar_t> & op){

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
                             op,
                             x_box.centers,
                             y_box.centers,
                             x_box.edge,
                             x_boxes_ind_count,
                             y_boxes_ind_count,
                                x_box.box_indices_sorted,
                             device_gpu,
                             x_idx_reordering,
                             y_idx_reordering,
                             true);
};

torch::Tensor chebyshev_nodes_1D(const int & nodes){
    float PI = atan(1.)*4.;
    torch::Tensor chebyshev_nodes = torch::arange(1, nodes+1).toType(torch::kFloat32);
    chebyshev_nodes = cos((chebyshev_nodes*2.-1.)*PI/(2*chebyshev_nodes.size(0)));
    return chebyshev_nodes;
};


void recursive_indices(
        const int nodes,
        int d,
        std::vector<int> & container_small,
        std::vector< std::vector<int> > & container_big){
    if (d==1){
        for (int i =0;i<nodes;i++){
            std::vector<int> tmp = container_small;
            tmp.push_back(i);
            container_big.push_back(tmp);
        }
        return;
    }else{
        for (int i =0;i<nodes;i++){
            std::vector<int> tmp = container_small;
            tmp.push_back(i);
            recursive_indices(nodes,d-1,tmp,container_big);
        }
    }
}

torch::Tensor get_recursive_indices(const int nodes,
                                    int d){
    std::vector< std::vector<int> > indices = {};
    std::vector<int> init = {};
    std::vector<torch::Tensor> cat = {};
    recursive_indices(nodes,d,init,indices);
    for ( auto &row : indices )
    {
        cat.push_back(torch::from_blob(row.data(),{1,d},torch::kInt32));
    }
    return torch::cat(cat,0);
}
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
//    box_data = (2 / n_tree.edge) * (box_data - n_tree.get_centers_expanded());
    int l_n=laplace_indices.size(0);
    int *d_min_size;
    cudaMalloc((void **)&d_min_size, sizeof(int));
    cudaMemcpy(d_min_size, &l_n, sizeof(int), cudaMemcpyHostToDevice);
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
                d_min_size,
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
                d_min_size,
                centers.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                edge.data_ptr<scalar_t>(),
                idx_reordering.packed_accessor32<int,1,torch::RestrictPtrTraits>()
        );
        cudaDeviceSynchronize();
    }
}



template <typename scalar_t>
torch::Tensor get_cheb_data(
        torch::Tensor & cheb_nodes,
        torch::Tensor & laplace_combinations
        ){
    torch::Tensor cheb_data = torch::zeros_like(laplace_combinations).toType(torch::kFloat32);
    auto cheb_node_accessor = cheb_nodes.accessor<scalar_t,1>();
    auto cheb_data_accessor = cheb_data.accessor<scalar_t,2>();
    auto laplace_combinations_accessor = laplace_combinations.accessor<int,2>();
    for(int i =0; i<laplace_combinations.size(0);i++){
        for(int j =0; j<nd;j++){
            cheb_data_accessor[i][j] = cheb_node_accessor[laplace_combinations_accessor[i][j]];
        }
    }
    return cheb_data;
}


torch::Tensor get_distance_tensor(torch::Tensor & near_field_interactions,
                                  int nx_box,
                                  int ny_box,
                                  torch::Tensor & distances
                                  ){
    torch::Tensor distance_tensor = torch::zeros({nx_box,ny_box,nd}).toType(torch::kFloat32);
    auto accessor_distance_tensor =distance_tensor.accessor<float,3>();
    auto accessor_interactions =near_field_interactions.accessor<long,2>();
    auto accessor_distances = distances.accessor<float,2>();
    int n_int = near_field_interactions.size(0);
    for (int i=0;i<n_int;i++){
        for (int k=0;k<nd;k++){
            accessor_distance_tensor[accessor_interactions[i][0]][accessor_interactions[i][1]][k]=accessor_distances[i][k];
        }
    };
    return distance_tensor;
}

template <typename scalar_t>
torch::Tensor setup_skip_conv(torch::Tensor &cheb_data,
                              torch::Tensor &b_data,
                              torch::Tensor & centers_X,
                              torch::Tensor & centers_Y,
                              torch::Tensor & unique_sorted_boxes_idx,
                              rbf_pointer<scalar_t> & op,
                              scalar_t & ls,
                              const std::string & device_gpu,
                              torch::Tensor & interactions_x_parsed,
                              torch::Tensor & interactions_y
){
    scalar_t *d_ls;
    int *d_min_size;
    cudaMalloc((void **)&d_ls, sizeof(scalar_t));
    cudaMemcpy(d_ls, &ls, sizeof(scalar_t), cudaMemcpyHostToDevice);
    torch::Tensor indicator,box_block,output;
    int min_size=cheb_data.size(0);
    int blkSize = optimal_blocksize(min_size);
//    torch::Tensor boxes_count = min_size*torch::ones({unique_sorted_boxes_idx.size(0)+1}).toType(torch::kInt32);
    torch::Tensor boxes_count = min_size*torch::ones(unique_sorted_boxes_idx.size(0)).toType(torch::kInt32).to(device_gpu);
    dim3 blockSize,gridSize;
    int memory,required_shared_mem;
    unique_sorted_boxes_idx = unique_sorted_boxes_idx.toType(torch::kInt32);
    std::tie(blockSize,gridSize,memory,indicator,box_block) = skip_kernel_launch<scalar_t>(nd,blkSize,boxes_count,unique_sorted_boxes_idx);
    output = torch::zeros_like(b_data);
    required_shared_mem = (min_size*(nd+1)+nd)*sizeof(scalar_t);
    cudaMalloc((void **)&d_min_size, sizeof(int));
    cudaMemcpy(d_min_size, &min_size, sizeof(int), cudaMemcpyHostToDevice);

    if (required_shared_mem<=SHAREDMEMPERBLOCK/8){
        skip_conv_far_boxes_opt<scalar_t><<<gridSize,blockSize,required_shared_mem>>>(
                cheb_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                b_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                d_ls,
                op,
                centers_X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                centers_Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                box_block.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                d_min_size,
                interactions_x_parsed.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                interactions_y.packed_accessor32<int,1,torch::RestrictPtrTraits>()
        );
    }else{
        memory = memory+nd*sizeof(scalar_t);
        skip_conv_far_cookie<scalar_t><<<gridSize,blockSize,memory>>>(
                cheb_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                b_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                d_ls,
                op,
                centers_X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                centers_Y.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                indicator.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                box_block.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                d_min_size
        );
    }
    return output;
}

//Always pass interactions..., if there aint any then ok etc...
template <typename scalar_t>
torch::Tensor far_field_compute_v2(
                       torch::Tensor & interactions_x_parsed,
                       torch::Tensor & interactions_y,
                       n_tree_cuda<scalar_t> & x_box,
                       n_tree_cuda<scalar_t> & y_box,
                       torch::Tensor & output,
                       torch::Tensor &b,
                       const std::string & device_gpu,
                       torch::Tensor &dist,
                       scalar_t & ls,
                       rbf_pointer<scalar_t> & op,
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
    low_rank_y = setup_skip_conv<scalar_t>(
            cheb_data_X,
            low_rank_y,
            x_box.centers,
            y_box.centers,
            x_box.box_indices_sorted,
            op,
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
std::tuple<torch::Tensor,torch::Tensor> get_new_interactions(torch::Tensor & old_near_interactions, int & p,const std::string & gpu_device){
    int n = old_near_interactions.size(0);
    old_near_interactions = old_near_interactions.repeat_interleave(p*p,0);
    torch::Tensor arr = torch::arange(p).to(gpu_device);
//    torch::Tensor test = arr.repeat(p*n);
//    torch::Tensor test_2 = arr.repeat_interleave(p).repeat(n);
    torch::Tensor interaction_x = arr.repeat_interleave(p).repeat(n)+p*old_near_interactions;
    torch::Tensor interaction_y = arr.repeat(p*n)+p*old_near_interactions;

//    torch::Tensor new_interactions_vec = torch::stack({arr.repeat_interleave(p).repeat(n),arr.repeat(p*n)},1).to(gpu_device)+p*old_near_interactions;
    return std::make_tuple(interaction_x,interaction_y);
}

torch::Tensor process_interactions(torch::Tensor & interactions,int x_boxes,const std::string & gpu_device){
    torch::Tensor box_indices,tmp,counts,count_cumsum,results;
    std::tie(box_indices,tmp,counts) = torch::_unique2(interactions,true,false,true);
    count_cumsum = torch::cat({box_indices,counts.cumsum(0).toType(torch::kInt32)},1);  //64+1 vec
    results = -torch::ones({x_boxes,2}).toType(torch::kInt32).to(gpu_device);
    dim3 blockSize,gridSize;
    int memory;
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<int>(nd, count_cumsum.size(0));
    parse_x_boxes<<<gridSize,blockSize>>>(
            count_cumsum.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
            results.packed_accessor32<int,2,torch::RestrictPtrTraits>()
            );
    return results;

}

template<typename scalar_t>
std::tuple<torch::Tensor,torch::Tensor> get_cheb_data(
        torch::Tensor & cheb_nodes,
        int &d,
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

    return std::make_tuple(cheb_idx,cheb_data);

}

template <typename scalar_t>
torch::Tensor FFM(
        torch::Tensor &X_data,
        torch::Tensor &Y_data,
        torch::Tensor &b,
        const std::string & gpu_device,
        scalar_t & ls,
        rbf_pointer<scalar_t> & op
        ) {
    torch::Tensor output = torch::zeros({X_data.size(0),b.size(1)}).to(gpu_device); //initialize empty output
    torch::Tensor edge,xmin,ymin,x_near_unique,tmp,y_near_unique,square_dist,dist,interactions,far_field,near_field,dist_far_field,dist_near_field; //these are needed to figure out which interactions are near/far field
    std::tie(edge,xmin,ymin) = calculate_edge(X_data,Y_data); //actually calculate them
    n_tree_cuda<scalar_t> ntree_X = n_tree_cuda<scalar_t>(edge,X_data,xmin,gpu_device);
    n_tree_cuda<scalar_t> ntree_Y = n_tree_cuda<scalar_t>(edge,Y_data,ymin,gpu_device);
    near_field = torch::zeros({1,2}).toType(torch::kInt32).to(gpu_device);
    float min_points = pow((float) laplace_nodes,nd);
    torch::Tensor chebnodes_1D = chebyshev_nodes_1D(laplace_nodes).to(gpu_device); //get chebyshev nodes, laplace_nodes is fixed now, should probably be a variable
    torch::Tensor cheb_data_X,laplace_combinations;
    std::tie(laplace_combinations,cheb_data_X)=get_cheb_data<scalar_t>(chebnodes_1D,nd,gpu_device);

    //Could probably rewrite this...

    torch::Tensor interactions_x,interactions_y,interactions_x_parsed;
    while (near_field.numel()>0 and ntree_X.avg_nr_points > min_points and ntree_Y.avg_nr_points > min_points){
        ntree_X.divide();//divide ALL boxes recursively once
        ntree_Y.divide();//divide ALL boxes recursively once
        std::tie(interactions_x,interactions_y) = get_new_interactions(near_field,ntree_X.dim_fac,gpu_device);
        interactions_x_parsed = process_interactions(interactions_y,ntree_X.centers.size(0),gpu_device);

        //near_field = get_new_interactions(near_field,ntree_X.dim_fac);
        //Remove old far_field interactions...
        //release mode for timings
        //Where it takes time...
        //fix memory stuff
//        std::tie(square_dist, dist) = ntree_X.distance(ntree_Y, interactions); //get distances for all interactions
//        std::tie(far_field, near_field, dist_far_field,dist_near_field) = ntree_X.far_and_near_field(square_dist, interactions,dist); //classify into near and far field
//        if(far_field.numel()>0){
        torch::Tensor cheb_data = cheb_data_X*ntree_X.edge/2.+ntree_X.edge/2.;
        //pass interactions and get near fields...
        near_field = far_field_compute_v2<scalar_t>(
                interactions_x_parsed,
                interactions_y,
                ntree_X,
                ntree_Y,
                output,
                b,
                gpu_device,
                dist_far_field,
                ls,
                op,
                chebnodes_1D,
                laplace_combinations,
                cheb_data
                ); //far field compute

//        }
    }
//    if (near_field.numel()>0){
//        near_field_compute_v2<scalar_t>(near_field,ntree_X, ntree_Y, output, b, gpu_device,ls,op); //Make sure this thing works first!
//    }
    return output;
}
template <typename scalar_t>
struct FFM_object{
    torch::Tensor & X_data;
    torch::Tensor & Y_data;
    scalar_t & ls;
    rbf_pointer<scalar_t> & op;
    scalar_t &lambda;
    const std::string & gpu_device;

    FFM_object( //constructor
            torch::Tensor & X_data,
    torch::Tensor & Y_data,
    scalar_t & ls,
    rbf_pointer<scalar_t> & op,
    scalar_t &lambda,
    const std::string & gpu_device): X_data(X_data), Y_data(Y_data),ls(ls),op(op),lambda(lambda),gpu_device(gpu_device){
    };
    virtual torch::Tensor operator* (torch::Tensor & b){
        return FFM<scalar_t>(
                X_data,
                Y_data,
                b,
                gpu_device,
                ls,
                op
        )+b*lambda;
    };
};
template <typename scalar_t>
struct exact_MV : FFM_object<scalar_t>{
    exact_MV(torch::Tensor & X_data,
                         torch::Tensor & Y_data,
                         scalar_t & ls,
                         rbf_pointer<scalar_t> & op,
                         scalar_t &lambda,
                         const std::string & gpu_device): FFM_object<scalar_t>(X_data, Y_data, ls, op, lambda, gpu_device){};
    torch::Tensor operator* (torch::Tensor & b){
        torch::Tensor output = torch::zeros({FFM_object<scalar_t>::X_data.size(0), b.size(1)}).to(FFM_object<scalar_t>::gpu_device);
        rbf_call<scalar_t>(
                FFM_object<scalar_t>::X_data,
                FFM_object<scalar_t>::Y_data,
                b,
                output,
                FFM_object<scalar_t>::ls,
                FFM_object<scalar_t>::op,
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
