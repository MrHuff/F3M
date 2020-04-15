//
// Created by rhu on 2020-03-28.
//

#pragma once
#include "1_d_conv.cu"
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

struct n_roon { //might want not have indexing in these to save memory and just the actual points...
    int index;
    int n_elems;
    torch::Tensor row_indices;
    torch::Tensor center;
};
std::ostream& operator<<(std::ostream& os, const n_roon& v)
{
    os<< "cube: "<<v.index<<" n_lemens: "<<v.n_elems<<" center: "<<'\n';
    os<<v.center<<'\n';
    return os;
}
struct n_roon_big {
    torch::Tensor edge;
    torch::Tensor data;
    torch::Tensor xmin;
    std::vector<n_roon> n_roons;
    int current_nr_boxes;
    float avg_nr_points;
    int dim;
    int number_of_divisions;
    int largest_box_n;
    std::vector<torch::Tensor> output_coord = {};
    std::vector<torch::Tensor> ones = {};
    n_roon_big(torch::Tensor &e, torch::Tensor &d, torch::Tensor &xm){
        edge = e;
        data = d;
        xmin = xm;
        dim = data.size(1);
        for (int i=0;i<dim;i++){
            ones.push_back(-1*torch::ones(1));
        }
        output_coord = recursive_center(ones,output_coord);
        n_roons.push_back(n_roon{0,(int) data.size(0),torch::arange(data.size(0)),xmin+0.5*edge});
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
        std::vector<n_roon> _new_n_roons;
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
                    _new_n_roons.push_back( n_roon{index,n_elem,n_roons[i].row_indices.index({n_divisors[j]}),n_roons[i].center+0.25*edge*output_coord[j]} );
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
    torch::Tensor operator*(const n_roon_big& other){ //give all interactions, i.e. cartesian product of indices
        std::vector<torch::Tensor> tl = {};
        for (int i=0; i<current_nr_boxes;i++){
            for (int j=0; j<other.current_nr_boxes;j++){
                tl.push_back(torch::tensor({n_roons[i].index,other.n_roons[j].index}).unsqueeze(0));
            }
        };
        return torch::cat(tl,0);
    }

    std::tuple<torch::Tensor,torch::Tensor> distance(const n_roon_big& other, const torch::Tensor &interactions ) { //give all interactions, i.e. cartesian product of indices
        std::vector<torch::Tensor> d_l_1 = {};
        auto interaction_accessor = interactions.accessor<long,2>();
        for (int i = 0; i < interactions.size(0); i++) {
            d_l_1.push_back((n_roons[interaction_accessor[i][0]].center-n_roons[interaction_accessor[i][1]].center).unsqueeze(0));
        }
        torch::Tensor l1_distances = torch::cat(d_l_1);
        return std::make_tuple(l1_distances.pow_(2).sum(1).sqrt() ,l1_distances);
    }
    std::tuple<torch::Tensor,torch::Tensor> far_and_near_field(const torch::Tensor & square_dist, const torch::Tensor &interactions) const{
        torch::Tensor far_field = square_dist>=(edge*2+1e-6);
        return std::make_tuple(interactions.index({far_field}) ,interactions.index({torch::logical_not(far_field)}));
    }
};

std::ostream& operator<<(std::ostream& os, const n_roon_big& v)
{
    for(int i=0; i<v.current_nr_boxes;i++){
        os<<v.n_roons[i]<<"\n";
    }
    return os;
}

torch::Tensor concat_X_box_indices(n_roon_big &x_box,torch::Tensor & box_indices){
    std::vector<torch::Tensor> cat = {};
    for (int i=0; i<box_indices.numel();i++){
        cat.push_back(x_box.n_roons[box_indices.accessor<long,1>()[i]].row_indices);
    }
    return torch::cat(cat,0);
}

void replace_box_index_with_data_index_X(std::vector<torch::Tensor> &job_vector,n_roon_big & x_box) {
    for (auto &el : job_vector) {
        el = concat_X_box_indices(x_box, el);
    }
}

template <typename scalar_t>
void near_field_compute(torch::Tensor & near_field_interactions,
        n_roon_big & x_box,
        n_roon_big & y_box,
        torch::Tensor & output,
        torch::Tensor &b,
        const std::string & device_gpu){
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
    dim3 blockSize,gridSize;
    int memory;
    torch::Tensor Y_inds_job;
    torch::Tensor cuda_Y_job;
    torch::Tensor cuda_b_job;
    torch::Tensor cuda_X_job;
    torch::Tensor X_inds_job;
    torch::Tensor output_job;
    std::vector<torch::Tensor> results = {};
    for (int i=0; i<unique_box_indices_Y.numel();i++){
        Y_inds_job = y_box.n_roons[unique_box_indices_Y_accessor[i]].row_indices;
        cuda_Y_job = Y_data.index({Y_inds_job}).to(device_gpu); //breaks on seccond iteration...
        cuda_b_job = b.index({Y_inds_job}).to(device_gpu);
        X_inds_job = job_vector[unique_box_indices_Y_accessor[i]];
        cuda_X_job = X_data.index(X_inds_job).to(device_gpu);
        output_job = torch::zeros({X_inds_job.size(0),output.size(1)}).to(device_gpu);
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(nd, job_vector[0].size(0));
        rbf_1d_reduce_shared_torch<scalar_t><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                     cuda_Y_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                     cuda_b_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                     output_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());

        cudaDeviceSynchronize();
        results.push_back(output_job.to("cpu"));
    }
    torch::Tensor update = torch::cat({results});
    torch::Tensor rows = torch::cat({job_vector});
    update_2d_rows_cpu<float>(output,update,rows);
};
//Make smarter implementation for tmrw using pointers to X and b rather than accessing the entire thing

int get_RFF_dim(n_roon_big & x_box,n_roon_big & y_box){
    auto biggest = (float) max(x_box.largest_box_n,y_box.largest_box_n);
    return (int) round(sqrt(biggest)*log(biggest));
}

torch::Tensor chebyshev_nodes_1D(const int & nodes){
    float PI = atan(1.)*4.;
    torch::Tensor chebyshev_nodes = torch::arange(1, nodes).toType(torch::kFloat32);
    chebyshev_nodes = cos((chebyshev_nodes*2.-1.)*PI/chebyshev_nodes.size(0));
    return chebyshev_nodes;
};


void recursive_indices(
        const int nodes,
        int d, std::vector<int> & container_small,
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
        cat.push_back(torch::from_blob(row.data(),{1,3},torch::kInt32));
    }
    return torch::cat(cat,0);
}

torch::Tensor apply_laplace_interpolation(
        n_roon_big& y_box,
        torch::Tensor &b,
        int i ,
        const std::string & device_gpu,
        torch::Tensor & nodes,
        torch::Tensor & laplace_indices){
    torch::Tensor y_data = y_box.data.index({y_box.n_roons[i].row_indices}).to(device_gpu);
    torch::Tensor b_data = b.index({y_box.n_roons[i].row_indices}).to(device_gpu);
    torch::Tensor nodes_gpu = nodes.to(device_gpu);
    torch::Tensor laplace_indices_gpu = laplace_indices.to(device_gpu);



}

template <typename scalar_t>
void far_field_compute(torch::Tensor & near_field_interactions,
                        n_roon_big & x_box,
                        n_roon_big & y_box,
                        torch::Tensor & output,
                        torch::Tensor &b,
                        const std::string & device_gpu,
                        const int & nodes){
    torch::Tensor unique_box_indices_Y,_inverse_indices_Y,_counts_Y;
    std::tie(unique_box_indices_Y,_inverse_indices_Y,_counts_Y) = torch::_unique2(near_field_interactions.slice(1,1,2),true,true);
    int dimensions = x_box.dim;
    torch::Tensor chebnodes_1D = chebyshev_nodes_1D(nodes);
    torch::Tensor laplace_combinations = get_recursive_indices(nodes,dimensions);
//    auto unique_box_indices_Y_accessor = unique_box_indices_Y.accessor<long,1>(); //Get centers instead, think about n-dim chebyshev polynomials...



    //    std::vector<torch::Tensor> job_vector;
//    for (int i=0; i<unique_box_indices_Y.numel();i++){
//        job_vector.push_back(near_field_interactions.slice(1,0,1).index({_inverse_indices_Y==unique_box_indices_Y_accessor[i]}));
//    }
//    replace_box_index_with_data_index_X(job_vector,x_box);
////    cudaStream_t streams[MAX_STREAMS] ;
//    torch::Tensor & X_data = x_box.data;
//    torch::Tensor & Y_data = y_box.data;
//    dim3 blockSize,gridSize;
//    int memory;
//    torch::Tensor Y_inds_job;
//    torch::Tensor cuda_Y_job;
//    torch::Tensor cuda_b_job;
//    torch::Tensor cuda_X_job;
//    torch::Tensor X_inds_job;
//    torch::Tensor output_job;
//    std::vector<torch::Tensor> results = {};
//    for (int i=0; i<unique_box_indices_Y.numel();i++){
//        Y_inds_job = y_box.n_roons[unique_box_indices_Y_accessor[i]].row_indices;
//        cuda_Y_job = Y_data.index({Y_inds_job}).to(device_gpu); //breaks on seccond iteration...
//        cuda_b_job = b.index({Y_inds_job}).to(device_gpu);
//        X_inds_job = job_vector[unique_box_indices_Y_accessor[i]];
//        cuda_X_job = X_data.index(X_inds_job).to(device_gpu);
//        output_job = torch::zeros({X_inds_job.size(0),output.size(1)}).to(device_gpu);
//        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(nd, job_vector[0].size(0));
//        rbf_1d_reduce_shared_torch<scalar_t><<<gridSize,blockSize,memory>>>(cuda_X_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
//                                                                            cuda_Y_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
//                                                                            cuda_b_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
//                                                                            output_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
//
//        cudaDeviceSynchronize();
//        results.push_back(output_job.to("cpu"));
//    }
//    torch::Tensor update = torch::cat({results});
//    torch::Tensor rows = torch::cat({job_vector});
//    update_2d_rows_cpu<float>(output,update,rows);
};