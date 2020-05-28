//
// Created by rhu on 2020-05-24.
//
// partial box interaction -> no shared memory, unless doing parallel processes/ non fully allocated gpu cores. Could try different implementations
#pragma once
#include "n_tree.cuh"


//struct recursive_n_tree{
//    torch::Tensor edge;
//    torch::Tensor data;
//    torch::Tensor xmin;
//    std::vector<recursive_n_tree> tree_obj;
//    int parent_nr_boxes,child_nr_boxes,parent_largest_box_size,child_largest_box_size,dim,level;
//    float parent_avg_nr_points,child_avg_nr_points;
//    std::vector<torch::Tensor> output_coord = {};
//    recursive_n_tree(torch::Tensor &e, torch::Tensor &d, torch::Tensor &xm){
//        edge = e;
//        data = d;
//        xmin = xm;
//        dim = data.size(1);
//        parent_nr_boxes;
//    };
//};
//


