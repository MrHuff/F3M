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
#include "n_tree.cuh"
#include <cuda_profiler_api.h>
//#include "utils.h"
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

    /*
     * Implement "skip heuristic" for full interactions..., figure out how to calculate everything on cuda at the end, little interactions as possible. for each level of granularity...
     * */

    /*Pass array [i*n+j] just flattened matrix.... maybe not so much faster... inline stuff...
     * time building of laplace nodes and call to cuda kernel, time this using nvprof...
     * Wednesday? 4pm 16th of September.
     * debug the nans...why do I get them???
     *
     *
     * */

    /*
     * Evaluate performance and accuracy
     */
//    auto warmup_1 = std::chrono::high_resolution_clock::now();
//    res_ref = exact_ref * b_train;
//    auto end_warmup_1 = std::chrono::high_resolution_clock::now();
//    res = ffm_obj * b_train; //Horrendus complexity, needs to fixed now
//    auto end_warmup_2 = std::chrono::high_resolution_clock::now();

//    auto warmup_duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_warmup_1-warmup_1);
//    auto warmup_duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_warmup_2-end_warmup_1);
//    std::cout<<warmup_duration_1.count()<<std::endl;
//    std::cout<<warmup_duration_2.count()<<std::endl;
    int l_n = std::stoi(argv[1]);
    int n = std::stoi(argv[2]);
    float min_points = std::stof(argv[3]);
    int threshold = std::stoi(argv[4]);
    float a = std::stof(argv[5]);
    float b = std::stof(argv[6]);
    float ls = std::stof(argv[7]);
//    benchmark_1<3>(l_n,n,min_points,threshold,a,b,ls);
    benchmark_2<3>(l_n,n,min_points,threshold,a,b,ls);
    cudaProfilerStop();
    cudaDeviceReset();
    //chart out nodes, n, speed etc...
    return 0;
}
