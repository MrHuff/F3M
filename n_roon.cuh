//
// Created by rhu on 2020-03-28.
//

#pragma once
#include "1_d_conv.cu"
#include <vector>
//template<typename T>




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
        cat.push_back(x_box.n_roons[box_indices.accessor<long,1>()[i]].row_indices.to(torch::kInt32));
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
    torch::Tensor X_data = x_box.data;
    torch::Tensor & Y_data = y_box.data;
    X_data = X_data.to(device_gpu);
    output = output.to(device_gpu);
    auto X_data_accessor = X_data.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>();
    auto output_accessor = output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>();
    dim3 blockSize,gridSize;
    int memory;
    torch::Tensor Y_inds_job;
    torch::Tensor cuda_Y_job;
    torch::Tensor cuda_b_job;
    torch::Tensor X_indices_torch; //use same strat for output and X?
    for (int i=0; i<unique_box_indices_Y.numel();i++){
        Y_inds_job = y_box.n_roons[unique_box_indices_Y_accessor[i]].row_indices;
        cuda_Y_job = Y_data.index({Y_inds_job}).to(device_gpu); //breaks on seccond iteration...
        auto cuda_Y_job_accessor = cuda_Y_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>();
        cuda_b_job = b.index({Y_inds_job}).to(device_gpu);
        auto cuda_b_job_accessor = cuda_b_job.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>();
        X_indices_torch = job_vector[unique_box_indices_Y_accessor[i]].to(device_gpu);
        auto * X_indices_ptr = (int *) X_indices_torch.data_ptr();
        std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<scalar_t>(nd, job_vector[0].size(0));
        near_field_rbf<scalar_t><<<gridSize,blockSize>>>(X_data_accessor,cuda_Y_job_accessor,cuda_b_job_accessor,X_indices_ptr,output_accessor,X_indices_torch.size(0));
        cudaDeviceSynchronize();
        std::cout<<output<<std::endl;
    }
};
//Make smarter implementation for tmrw using pointers to X and b rather than accessing the entire thing
