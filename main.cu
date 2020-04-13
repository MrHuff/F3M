/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *
 */
//#include <stdio.h>
#include "1_d_conv.cu"
#include "n_roon.cuh"

/*
*********************************************************************
function name: main
description: test and compare
parameters:
            none
return: none
*********************************************************************
*/


int main(int argc, char const *argv[]){

    const std::string device_cuda = "cuda:0"; //officially retarded
    const std::string device_cpu = "cpu";
//    std::cout<<X<<std::endl;
//    X = X.to(device_cuda); // needs to do the double op...
//    std::cout<<X<<std::endl;
//    X = X.to(device_cpu); // needs to do the double op...
//    std::cout<<X<<std::endl;
//    X = X.to(device_cuda); // moves contigously to cuda automagically, behaves cute. great!
//    std::cout<<X<<std::endl;
//    auto* ptr = (float*) X.data_ptr();
//    print_test<float><<<1,1000>>>(ptr);
//    for (int i=0; i<X.size(0);i++){
//        std::cout<<ptr[i]<<std::endl;
//    }
//    X = X.to(device_cuda); // needs to do the double op...

    // Python: tensor[:,i] <-> tensor.slice(1,i,i+1)
    torch::Tensor X = torch::rand({nx, nd});
    torch::Tensor b = torch::randn({X.size(0),2});
    torch::Tensor output = torch::zeros_like(b);

    torch::Tensor edge,xmin,ymin;
    std::tie(edge,xmin,ymin) = calculate_edge(X,X);
    n_roon_big octaroon_X = n_roon_big{edge,X,xmin};
    octaroon_X.divide();
    octaroon_X.divide();
    n_roon_big octaroon_Y = octaroon_X;
    torch::Tensor interactions =  octaroon_X*octaroon_Y;
//    torch::Tensor square_dist,l_1_dist;
//    std::tie(square_dist,l_1_dist) = octaroon_X.distance(octaroon_Y,interactions);
//    torch::Tensor far_field,near_field;
//    std::tie(far_field,near_field) = octaroon_X.far_and_near_field(square_dist,interactions);
//    //Idea is to find all X boxes and understand which are needed for interactions. Then remember each x-axis box's y-interactions. almost like matrix...
    near_field_compute<float>(interactions,octaroon_X,octaroon_Y,output,b,device_cuda);
    std::cout<<output<<std::endl;

    printf("--------------------------------------\n");
    torch::Tensor output_2 = torch::zeros_like(b).to(device_cuda);
    X = X.to(device_cuda);
    torch::Tensor Y = X;
    b = b.to(device_cuda);
    dim3 blockSize,gridSize;
    int memory;
    std::tie(blockSize,gridSize,memory) = get_kernel_launch_params<float>(nd, X.size(0));
    rbf_1d_reduce_simple_torch<float><<<gridSize,blockSize>>>(
            X.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            Y.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            b.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            output_2.packed_accessor32<float,2,torch::RestrictPtrTraits>());
    cudaDeviceSynchronize();
    std::cout<< output_2<<std::endl;
    return 0;
}
