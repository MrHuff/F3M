//
// Created by rhu on 2020-04-17.
//

#pragma once
#include "n_tree.cuh"
#include "GP_utils.cuh"
#include <fstream>


int writeOnfile_exp_1(char * filename,float a, float b, int l_p, int n, int d, float min_points, int time,float error) {
    std::fstream job_results;
    job_results.open(filename, std::fstream::in | std::fstream::out | std::fstream::app);
    job_results.seekg(0, std::ios::end);
    if (job_results.tellg() == 0) {
        std::cout << "file empty appending columns" << std::endl;
        job_results << "uniform_a,uniform_b,l_p,n,d,min_points,FFM_time,relative_error"<< std::endl;
    }
    job_results<<a<<","<<b<<","<<l_p<<","<<n<<","<<d<<","<<min_points<<","<<time<<","<<error<<std::endl;
    job_results.close();
    return 0;
}

int writeOnfile_exp_2(char* filename,float a, float b, int l_p, int n, int d, float min_points, int time,float error) {
    std::fstream job_results;
    job_results.open(filename, std::fstream::in | std::fstream::out | std::fstream::app);
    job_results.seekg(0, std::ios::end);
    if (job_results.tellg() == 0) {
        std::cout << "file empty appending columns" << std::endl;
        job_results << "normal mean,normal std,l_p,n,d,min_points,FFM_time,relative_error"<< std::endl;
    }
    job_results<<a<<","<<b<<","<<l_p<<","<<n<<","<<d<<","<<min_points<<","<<time<<","<<error<<std::endl;
    job_results.close();
    return 0;
}

template <typename type>
torch::Tensor read_csv(const std::string filename,const int rows,const int cols){
    std::ifstream file(filename);
    type r[rows][cols];
    std::vector<std::vector<type>> vec;
    if (file.is_open()) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                file >> r[i][j];
                file.get(); // Throw away the comma
            }
        }
        for (int i = 0; i < rows; ++i) {
            std::vector<type> tmp = {};
            for (int j = 0; j < cols; ++j) {
                tmp.push_back(r[i][j]);
            }
            vec.push_back(tmp);
        }
    }
    std::vector<torch::Tensor> blob = {};

    for ( auto &row : vec )
    {
        blob.push_back(torch::from_blob(row.data(),{1,cols},torch::kFloat32));
    }

    return torch::cat(blob);
}

template <int nd>
void benchmark_1(int laplace_n,int n,float min_points, int threshold,float a,float b,float ls,char* fname){
    const std::string device_cuda = "cuda:0"; //officially retarded
    const std::string device_cpu = "cpu";
//    torch::manual_seed(0);
//    torch::Tensor X_train = read_csv<float>("X_train_PCA.csv",11619,3); something wrong with data probably...
//    torch::Tensor b_train = read_csv<float>("Y_train.csv",11619,1); something wrong with data probably...
    torch::Tensor X_train = torch::empty({n,nd}).uniform_(a, b).to(device_cuda); //Something fishy going on here, probably the boxes stuff... //Try other distributions for pathological distributions!
    torch::Tensor b_train = torch::randn({n,1}).to(device_cuda);
    float lambda = 1e-1; // ridge parameter
    torch::Tensor res,res_ref;
    FFM_object<float,nd> ffm_obj = FFM_object<float,nd>(X_train, X_train, ls, lambda, device_cuda,laplace_n,min_points); //FMM object
//    FFM_object<float> ffm_obj_grad = FFM_object<float>(X,X,ls,op_grad,lambda,device_cuda);
//    exact_MV<float> ffm_obj_grad_exact = exact_MV<float>(X,X,ls,op_grad,lambda,device_cuda);
    std::cout<<"------------- "<<"Uniform distribution : "<< "a "<<a<<" b "<<b<<" laplace nodes: "<<laplace_n<<" n: "<<n<<" min_points: "<< min_points <<" -------------"<<std::endl;
    torch::Tensor subsampled_X = X_train.slice(0,0,threshold);
    exact_MV<float,nd> exact_ref = exact_MV<float,nd>(subsampled_X, X_train, ls,  lambda, device_cuda,laplace_n,min_points); //Exact method reference
    auto start = std::chrono::high_resolution_clock::now();
    res_ref = exact_ref *b_train;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout<<"Full matmul time (ms): "<<duration_1.count()<<std::endl;
    res = ffm_obj * b_train; //Fast math creates problems... fast math does a lot!!!
    auto end_2 = std::chrono::high_resolution_clock::now();
    auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2-end);
    torch::Tensor res_compare = res.slice(0,0,threshold);
    torch::Tensor rel_error  = ((res_ref-res_compare)/res_ref).abs_().mean();
    float rel_error_float = rel_error.item<float>();
    std::cout<<"FFM time (ms): "<<duration_2.count()<<std::endl;
    std::cout<<"Relative error: "<<rel_error_float<<std::endl;
    writeOnfile_exp_1(fname,a,b,laplace_n,n,nd,min_points,duration_2.count(),rel_error_float);

}

template <int nd>
void benchmark_2(int laplace_n,int n,float min_points, int threshold,float mean,float var,float ls,char* fname){
    const std::string device_cuda = "cuda:0"; //officially retarded
    const std::string device_cpu = "cpu";
//    torch::manual_seed(0);

//    torch::Tensor X_train = read_csv<float>("X_train_PCA.csv",11619,3); something wrong with data probably...
//    torch::Tensor b_train = read_csv<float>("Y_train.csv",11619,1); something wrong with data probably...
    torch::Tensor X_train = torch::empty({n,nd}).normal_(mean, var).to(device_cuda); //Something fishy going on here, probably the boxes stuff... //Try other distributions for pathological distributions!
    torch::Tensor b_train = torch::randn({n,1}).to(device_cuda);
    float lambda = 1e-1; // ridge parameter
    torch::Tensor res,res_ref;
    FFM_object<float,nd> ffm_obj = FFM_object<float,nd>(X_train, X_train, ls, lambda, device_cuda,laplace_n,min_points); //FMM object
//    FFM_object<float> ffm_obj_grad = FFM_object<float>(X,X,ls,op_grad,lambda,device_cuda);
//    exact_MV<float> ffm_obj_grad_exact = exact_MV<float>(X,X,ls,op_grad,lambda,device_cuda);
    std::cout<<"------------- "<<"Normal distribution: "<< "mean "<<mean<<" variance "<<var<<" laplace nodes: "<<laplace_n<<" n: "<<n<<" min_points: "<< min_points <<" -------------"<<std::endl;
    torch::Tensor subsampled_X = X_train.slice(0,0,threshold);
    exact_MV<float,nd> exact_ref = exact_MV<float,nd>(subsampled_X, X_train, ls,  lambda, device_cuda,laplace_n,min_points); //Exact method reference
    auto start = std::chrono::high_resolution_clock::now();
    res_ref = exact_ref *b_train;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout<<"Full matmul time (ms): "<<duration_1.count()<<std::endl;
    res = ffm_obj * b_train; //Fast math creates problems... fast math does a lot!!!
    auto end_2 = std::chrono::high_resolution_clock::now();
    auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2-end);
    torch::Tensor res_compare = res.slice(0,0,threshold);
    torch::Tensor rel_error  = ((res_ref-res_compare)/res_ref).abs_().mean();
    float rel_error_float = rel_error.item<float>();
    std::cout<<"FFM time (ms): "<<duration_2.count()<<std::endl;
    std::cout<<"Relative error: "<<rel_error_float<<std::endl;
    writeOnfile_exp_2(fname,mean,var,laplace_n,n,nd,min_points,duration_2.count(),rel_error_float);

}