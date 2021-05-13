/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *
 */
#include <cuda_profiler_api.h>
#include "utils.h"
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
    int n = std::stoi(argv[1]);
    float min_points = std::stof(argv[2]);
    int threshold = std::stoi(argv[3]);
    float a = std::stof(argv[4]);
    float b = std::stof(argv[5]);
    float ls = std::stof(argv[6]);
    int nr_of_interpolation_nodes = std::stoi(argv[7]);
    bool var_comp = (bool) std::stoi(argv[8]);
    float eff_var =  std::stof(argv[9]);
    int small_field_limit = std::stoi(argv[10]);
    bool small_x = (bool)std::stoi(argv[11]);
    int job = std::stoi(argv[12]);
    char * fname = const_cast<char *>(argv[13]);

    if (job==1){
        benchmark_1<float,3>(n,min_points,threshold,a,b,ls,nr_of_interpolation_nodes,var_comp,eff_var,small_field_limit,small_x,fname); //Can't do a billion points for 3 dim...
    }
    if(job==2){
        benchmark_2<float,5>(n,min_points,threshold,a,b,ls,nr_of_interpolation_nodes,var_comp,eff_var,small_field_limit,small_x,fname);//Weird curse of dimensionality... Should scale linearly in diemnsions...
    }
    if(job==3){
        benchmark_3<float,3>(n,min_points,threshold,a,b,ls,nr_of_interpolation_nodes,var_comp,eff_var,small_field_limit,small_x,fname);//Weird curse of dimensionality... Should scale linearly in diemnsions...
    }
    if(job==4){
        benchmark_4<float,3>(n,min_points,threshold,a,b,ls,nr_of_interpolation_nodes,var_comp,eff_var,small_field_limit,small_x,fname);//Weird curse of dimensionality... Should scale linearly in diemnsions...
    }
    if(job==5){
        benchmark_5<double,3>(n,min_points,threshold,a,b,ls,nr_of_interpolation_nodes,var_comp,eff_var,small_field_limit,small_x,fname);//Weird curse of dimensionality... Should scale linearly in diemnsions...
    }
    if(job==6){
        benchmark_6<float,3>(n,min_points,threshold,a,b,ls,nr_of_interpolation_nodes,var_comp,eff_var,small_field_limit,small_x,fname);//Weird curse of dimensionality... Should scale linearly in diemnsions...
    }
    if(job==7){
        benchmark_7<float,3>(n,min_points,threshold,a,b,ls,nr_of_interpolation_nodes,var_comp,eff_var,small_field_limit,small_x,fname);//Weird curse of dimensionality... Should scale linearly in diemnsions...
    }


    cudaProfilerStop();
    cudaDeviceReset();
    //chart out nodes, n, speed etc...
    return 0;
}
