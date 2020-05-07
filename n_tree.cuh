//
// Created by rhu on 2020-03-28.
//

#pragma once
#include "1_d_conv.cu"
#include "utils.h"
#include <vector>
//template<typename T>

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
struct n_tree_big {
    torch::Tensor edge;
    torch::Tensor data;
    torch::Tensor xmin;
    std::vector<n_tree> n_roons;
    int current_nr_boxes;
    float avg_nr_points;
    int dim;
    int number_of_divisions;
    int largest_box_n;
    std::vector<torch::Tensor> output_coord = {};
    std::vector<torch::Tensor> ones = {};
    n_tree_big(torch::Tensor &e, torch::Tensor &d, torch::Tensor &xm){
        edge = e;
        data = d;
        xmin = xm;
        dim = data.size(1);
        for (int i=0;i<dim;i++){
            ones.push_back(-1*torch::ones(1));
        }
        output_coord = recursive_center(ones,output_coord);
        n_roons.push_back(n_tree{0, (int) data.size(0), torch::arange(data.size(0)), xmin + 0.5 * edge});
        current_nr_boxes= n_roons.size();
        avg_nr_points = data.size(0);
        number_of_divisions = (int) pow(2,dim);
        largest_box_n = data.size(0);
    };
    void divide(){
        int _new_nr_of_boxes = 0;
        int n_elem;
        int index;
        float sum_points = 0;
        std::vector<n_tree> _new_n_roons;
        for (int i=0;i<current_nr_boxes;i++){
            torch::Tensor tmp_points = data.index({n_roons[i].row_indices}); //have to use a copy?
            std::vector<torch::Tensor> bool_vector = (tmp_points<=n_roons[i].center).unbind(dim=1);
            std::vector<torch::Tensor> n_divisors = {};
            n_divisors = recursive_divide(bool_vector,n_divisors);
            for (int j=0;j<number_of_divisions;j++){
                n_elem = n_divisors[j].sum().item().toInt();
                sum_points+=(float) n_elem;
                index = (int) i*number_of_divisions + j;
                if (n_elem>0){
                    _new_n_roons.push_back(n_tree{index, n_elem, n_roons[i].row_indices.index({n_divisors[j]}), n_roons[i].center + 0.25 * edge * output_coord[j]} );
                _new_nr_of_boxes++;
                }
            }
        };
        current_nr_boxes = _new_nr_of_boxes;
        n_roons = _new_n_roons;
        avg_nr_points = sum_points/(float) current_nr_boxes;
        edge = edge*0.5;
        largest_box_n = get_largest_box();
    };
    int get_largest_box(){
        std::vector<int> box_sizes = {};
        for (auto el:n_roons){
            box_sizes.push_back(el.n_elems);
        }
        return *std::max_element(box_sizes.begin(),box_sizes.end());
    };
    torch::Tensor operator*(const n_tree_big& other){ //give all interactions, i.e. cartesian product of boxes
        std::vector<torch::Tensor> tl = {};
        for (int i=0; i<current_nr_boxes;i++){
            for (int j=0; j<other.current_nr_boxes;j++){
                tl.push_back(torch::tensor({n_roons[i].index,other.n_roons[j].index}).unsqueeze(0));
            }
        };
        return torch::cat(tl,0);
    }

    std::tuple<torch::Tensor,torch::Tensor> distance(const n_tree_big& other, const torch::Tensor &interactions ) { //give all interactions, i.e. cartesian product of indices
        std::vector<torch::Tensor> dist = {};
        auto interaction_accessor = interactions.accessor<long,2>();
        for (int i = 0; i < interactions.size(0); i++) {
            dist.push_back((n_roons[interaction_accessor[i][1]].center - n_roons[interaction_accessor[i][0]].center).unsqueeze(0));
        }
        torch::Tensor l1_distances = torch::cat(dist);
        return std::make_tuple(l1_distances.pow(2).sum(1).sqrt() ,l1_distances);
    }
    std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> far_and_near_field(
            const torch::Tensor & square_dist,
            const torch::Tensor &interactions,
            const torch::Tensor &L1_dist) const{
        torch::Tensor far_field = square_dist>=(edge*2+1e-6);
        return std::make_tuple(interactions.index({far_field}),
                interactions.index({torch::logical_not(far_field)}),
               L1_dist.index({far_field}));
    }
};

std::ostream& operator<<(std::ostream& os, const n_tree_big& v)
{
    for(int i=0; i<v.current_nr_boxes;i++){
        os<<v.n_roons[i]<<"\n";
    }
    return os;
}

torch::Tensor concat_X_box_indices(n_tree_big &x_box, torch::Tensor & box_indices){
    std::vector<torch::Tensor> cat = {};
    for (int i=0; i<box_indices.numel();i++){
        cat.push_back(x_box.n_roons[box_indices.accessor<long,1>()[i]].row_indices);
    }
    return torch::cat(cat,0);
}

void replace_box_index_with_data_index_X(std::vector<torch::Tensor> &job_vector, n_tree_big & x_box) {
    for (auto &el : job_vector) {
        el = concat_X_box_indices(x_box, el);
    }
}

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

template <typename scalar_t>
void near_field_compute(torch::Tensor & near_field_interactions,
                        n_tree_big & x_box,
                        n_tree_big & y_box,
                        torch::Tensor & output,
                        torch::Tensor &b,
                        const std::string & device_gpu,
                        scalar_t & ls,
                        rbf_pointer<scalar_t> & op){
    torch::Tensor unique_box_indices_Y,_inverse_indices_Y,_counts_Y;
    std::tie(unique_box_indices_Y,_inverse_indices_Y,_counts_Y) = torch::_unique2(near_field_interactions.slice(1,1,2),true,true);
    std::vector<torch::Tensor> job_vector;
    auto unique_box_indices_Y_accessor = unique_box_indices_Y.accessor<long,1>();
    for (int i=0; i<unique_box_indices_Y.numel();i++){
        job_vector.push_back(near_field_interactions.slice(1,0,1).index({_inverse_indices_Y==unique_box_indices_Y_accessor[i]}));
    }
    replace_box_index_with_data_index_X(job_vector,x_box);
//    cudaStream_t streams[MAX_STREAMS];
    torch::Tensor & X_data = x_box.data;
    torch::Tensor & Y_data = y_box.data;

    torch::Tensor Y_inds_job,cuda_Y_job,cuda_b_job,cuda_X_job,X_inds_job,output_job;
    std::vector<torch::Tensor> results = {};

    for (int i=0; i<unique_box_indices_Y.numel();i++){ //for each Y-box, find all the "common" X-boxes, stack the X-boxes and compute exactly.
        Y_inds_job = y_box.n_roons[unique_box_indices_Y_accessor[i]].row_indices;
        cuda_Y_job = Y_data.index({Y_inds_job}).to(device_gpu); //breaks on seccond iteration...
        cuda_b_job = b.index({Y_inds_job}).to(device_gpu);
        X_inds_job = job_vector[unique_box_indices_Y_accessor[i]];
        cuda_X_job = X_data.index(X_inds_job).to(device_gpu);
        output_job = torch::zeros({X_inds_job.size(0),output.size(1)}).to(device_gpu);
        rbf_call<scalar_t>(cuda_X_job, cuda_Y_job, cuda_b_job, output_job,ls,op);
        results.push_back(output_job.to("cpu"));
    }
    torch::Tensor update = torch::cat({results});
    torch::Tensor rows = torch::cat({job_vector});
    update_2d_rows_cpu<scalar_t>(output,update,rows);
};
//Make smarter implementation for tmrw using pointers to X and b rather than accessing the entire thing

int get_RFF_dim(n_tree_big & x_box, n_tree_big & y_box){
    auto biggest = (float) max(x_box.largest_box_n,y_box.largest_box_n);
    return (int) round(sqrt(biggest)*log(biggest));
}

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
torch::Tensor apply_laplace_interpolation(
        n_tree_big& BOX,
        torch::Tensor &b,
        long i ,
        const std::string & device_gpu,
        torch::Tensor & nodes,
        torch::Tensor & laplace_indices,
        const bool tranpose_mode=false){
    torch::Tensor box_data = BOX.data.index({BOX.n_roons[i].row_indices});
    box_data = ((2 / BOX.edge) * (box_data - BOX.n_roons[i].center)).to(device_gpu);
    dim3 blockSize,gridSize;
    int memory;
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(nd, box_data.size(0));

    if (not tranpose_mode){
        torch::Tensor laplace_low_rank = torch::zeros({b.size(1),laplace_indices.size(0)}).to(device_gpu);
        torch::Tensor b_data = b.index({BOX.n_roons[i].row_indices}).to(device_gpu);
        laplace_interpolation<scalar_t><<<gridSize,blockSize>>>(
                box_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                b_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                nodes.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                laplace_indices.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                laplace_low_rank.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
        cudaDeviceSynchronize();
        return laplace_low_rank.t_();
    }else{
        torch::Tensor high_rank_res = torch::zeros({box_data.size(0),b.size(1)}).to(device_gpu);
        torch::Tensor& b_data =b;
        laplace_interpolation_transpose<scalar_t><<<gridSize,blockSize>>>(
                box_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                b_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                nodes.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                laplace_indices.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                high_rank_res.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
        cudaDeviceSynchronize();
        return high_rank_res;
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

template <typename scalar_t>
torch::Tensor low_rank_exact(torch::Tensor & cuda_X_job,
                    torch::Tensor & cuda_b_job,
                    torch::Tensor & distance,
                     const std::string & device_gpu,
                     scalar_t & ls,
                     rbf_pointer<scalar_t> & op
                     ){
    distance = distance.to(device_gpu);
    torch::Tensor output_job = torch::zeros_like(cuda_b_job).toType(torch::kFloat32);
    torch::Tensor cuda_Y_job = cuda_X_job + distance;
    rbf_call<scalar_t>(cuda_X_job, cuda_Y_job, cuda_b_job, output_job,ls,op);
    return output_job;
}

template <typename scalar_t>
void far_field_compute(torch::Tensor & far_field_interactions,
                       n_tree_big & x_box,
                       n_tree_big & y_box,
                       torch::Tensor & output,
                       torch::Tensor &b,
                       const std::string & device_gpu,
                       torch::Tensor &dist,
                       scalar_t & ls,
                       rbf_pointer<scalar_t> & op){
    torch::Tensor chebnodes_1D = chebyshev_nodes_1D(laplace_nodes); //get chebyshev nodes, laplace_nodes is fixed now, should probably be a variable
    torch::Tensor laplace_combinations = get_recursive_indices(laplace_nodes,nd); // get all possible combinations of chebyshev nodes. Total combintations should have dimension [laplace_nodes^nd,nd]
    torch::Tensor cheb_data_X = (get_cheb_data<scalar_t>(chebnodes_1D,laplace_combinations)*x_box.edge/2+x_box.edge/2).to(device_gpu); //get the actual data given the indices
    chebnodes_1D=chebnodes_1D.to(device_gpu);
    laplace_combinations=laplace_combinations.to(device_gpu);
    torch::Tensor unique_box_indices_Y,_inverse_indices_Y,_counts_Y,cheb_data_Y,unique_box_indices_X,_1,_2;
    std::tie(unique_box_indices_Y,_inverse_indices_Y,_counts_Y) = torch::_unique2(far_field_interactions.slice(1, 1, 2), true, false);
    std::tie(unique_box_indices_X,_1,_2) = torch::_unique2(far_field_interactions.slice(1, 0, 1), true, false);

    /*
     * for all unique indices do a transformation, i.e. L_y*u.
     */
    std::map<long, torch::Tensor> Y_box_transforms;
    torch::Tensor laplace_low_rank,y_subset,b_subset;
    auto unique_box_indices_Y_accessor = unique_box_indices_Y.accessor<long,1>();
    auto unique_box_indices_X_accessor = unique_box_indices_X.accessor<long,1>();

    for (int i=0; i<unique_box_indices_Y.numel();i++){//parallelize!
        Y_box_transforms[unique_box_indices_Y_accessor[i]] = apply_laplace_interpolation<scalar_t>(
                y_box,
                b,
                unique_box_indices_Y_accessor[i],
                device_gpu,
                chebnodes_1D,
                laplace_combinations,
                false);
    }

    /*
     * Do all low rank interactions,i.e. T*(L_y*u) we apply T here.
     */
    std::map<long, torch::Tensor> results; //Store/accumulate results according to unique X-boxes
    for (int i=0; i<unique_box_indices_X.numel();i++){
        results[unique_box_indices_X_accessor[i]] = torch::zeros({cheb_data_X.size(0), b.size(1)}).toType(torch::kFloat32).to(device_gpu);
    }
    torch::Tensor xy; //distance offset
    long m,n;
    auto far_field_accessor = far_field_interactions.accessor<long,2>();
    for (int i =0; i<dist.size(0);i++){ //parallelize
        xy = dist.slice(0,i,i+1);
        m = far_field_accessor[i][1];
        n = far_field_accessor[i][0];
        results[unique_box_indices_X_accessor[n]]+=low_rank_exact<scalar_t>(
                cheb_data_X,
                Y_box_transforms[unique_box_indices_Y_accessor[m]],
                xy,
                device_gpu,
                ls,
                op
                );
    }

    /*
     * Finally transform back to regular dimensionality i.e. Apply L_x *(T*(L_y*u))
     */
    std::vector<torch::Tensor> final_res = {}; // We cache all the results...
    std::vector<torch::Tensor> final_indices = {}; // ... and remember the index we applied it to!

    for (int i=0; i<unique_box_indices_X.numel();i++){//parallelize!
        final_res.push_back(apply_laplace_interpolation<scalar_t>(
                x_box,
                results[unique_box_indices_X_accessor[i]],
                unique_box_indices_X_accessor[i],
                device_gpu,
                chebnodes_1D,
                laplace_combinations,
                true).to("cpu"));
        final_indices.push_back(x_box.n_roons[unique_box_indices_X_accessor[i]].row_indices);
        }
    torch::Tensor update = torch::cat({final_res});
    torch::Tensor rows = torch::cat({final_indices});
    update_2d_rows_cpu<scalar_t>(output,update,rows); //since we have not done accumulation ,we do accumulation here...
};
template <typename scalar_t>
torch::Tensor FFM(
        torch::Tensor &X_data,
        torch::Tensor &Y_data,
        torch::Tensor &b,
        const std::string & gpu_device,
        scalar_t & ls,
        rbf_pointer<scalar_t> & op
        ) {
    torch::Tensor output = torch::zeros({X_data.size(0),b.size(1)}); //initialize empty output
    torch::Tensor edge,xmin,ymin; //these are needed for computing centers of boxes
    std::tie(edge,xmin,ymin) = calculate_edge(X_data,Y_data); //actually calculate them
    n_tree_big ntree_X = n_tree_big{edge, X_data, xmin}; //Intialize tree for X-data
    n_tree_big ntree_Y = n_tree_big{edge, Y_data, ymin};//Intialize tree for Y-data
    torch::Tensor square_dist,dist,interactions,far_field,near_field,dist_far_field; //these are needed to figure out which interactions are near/far field
    near_field = torch::zeros({1,2}).toType(torch::kLong);
    while (near_field.numel()>0 and ntree_X.avg_nr_points > 100. and ntree_Y.avg_nr_points > 100.){
        ntree_X.divide();//divide ALL boxes recursively once
        ntree_Y.divide();//divide ALL boxes recursively once
        interactions =  ntree_X*ntree_Y; //find ALL interactions
        std::tie(square_dist, dist) = ntree_X.distance(ntree_Y, interactions); //get distances for all interactions
//        std::cout<<square_dist<<std::endl;
//        std::cout<<dist<<std::endl;
        std::tie(far_field, near_field, dist_far_field) = ntree_X.far_and_near_field(square_dist, interactions, dist); //classify into near and far field
        if(far_field.numel()>0){
            far_field_compute<scalar_t>(far_field, ntree_X, ntree_Y, output, b, gpu_device, dist_far_field,ls,op); //far field compute
        }
    }
    if (near_field.numel()>0){
        near_field_compute<scalar_t>(near_field,ntree_X, ntree_Y, output, b, gpu_device,ls,op);
    }
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
    const std::string & gpu_device): X_data(X_data), Y_data(Y_data),ls(ls),op(op),lambda(lambda),gpu_device(gpu_device){};
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
        b = b.to(FFM_object<scalar_t>::gpu_device);
        torch::Tensor gpu_X = FFM_object<scalar_t>::X_data.to(FFM_object<scalar_t>::gpu_device);
        torch::Tensor gpu_Y = FFM_object<scalar_t>::Y_data.to(FFM_object<scalar_t>::gpu_device);

        rbf_call<scalar_t>(
                gpu_X,
                gpu_Y,
                b,
                output,
                FFM_object<scalar_t>::ls,
                FFM_object<scalar_t>::op,
                true
        );
        output = output+ b * FFM_object<scalar_t>::lambda;
        b = b.to("cpu");
        output = output.to("cpu");
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
