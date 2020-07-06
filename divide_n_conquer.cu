//
// Created by rhu on 2020-07-05.
//

#pragma once
#include <torch/torch.h> //for n00bs like me, direct translation to python rofl
#include "1_d_conv.cu"
template <typename scalar_t>
__global__ void box_division(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> X_data,
                             const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> centers,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> b,
                             const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> old_indices,
                             torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> output,
                             int divide_num


){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current thread
    if (i>X_data.size(0)-1){return;}
    int old_ind = old_indices[i];
    int add = old_ind*divide_num;
    for (int k=0;k<nd;k++){
        output[i]+= (centers[old_ind][k]<=X_data[i][k])*b[k]+add;
    }
}



