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


int main(int argc, char const *argv[])
{
// Python: tensor[:,i] <-> tensor.slice(1,i,i+1)
    torch::Tensor X = torch::rand({nx, nd});
    torch::Tensor b = torch::randn({nx,1});
    torch::Tensor output = torch::zeros_like(b);

    torch::Tensor edge,xmin,ymin;
    std::tie(edge,xmin,ymin) = calculate_edge(X,X);
    n_roon_big octaroon_X = n_roon_big{edge,X,xmin};
//    std::cout<<octaroon_X;
    octaroon_X.divide();
//    std::cout<<octaroon_X;
    octaroon_X.divide();
//    std::cout<<octaroon_X;
    n_roon_big octaroon_Y = octaroon_X;
    torch::Tensor interactions =  octaroon_X*octaroon_Y;
//    std::cout<<interactions<<std::endl;
    torch::Tensor square_dist,l_1_dist;
    std::tie(square_dist,l_1_dist) = octaroon_X.distance(octaroon_Y,interactions);
    torch::Tensor far_field,near_field;
    std::tie(far_field,near_field) = octaroon_X.far_and_near_field(square_dist,interactions);
    //Idea is to find all X boxes and understand which are needed for interactions. Then remember each x-axis box's y-interactions. almost like matrix...

    near_field_compute(near_field,octaroon_X,octaroon_Y,output,b);
//    std::cout<<far_field<<std::endl;
//    std::cout<<near_field<<std::endl;


//    std::cout<< X.index({(X<0).slice(1,0,1)})<<std::endl;



//    std::vector<torch::Tensor> output_coord = {};
//    std::vector<torch::Tensor> ones = {};
//    for (int i=0;i<nd;i++){
//        ones.push_back(-1*torch::ones(1));
//    }
//    output_coord = recursive_center(ones,output_coord);
//    std::cout<<output_coord<<std::endl;

//    std::vector<int> bool_vec = {1,0,0,1};
//    int size = bool_vec.size();
//    torch::Tensor bool_tensor = torch::from_blob(bool_vec.data(),{size},torch::kBool);
//    std::cout<<bool_tensor<<std::endl;
//    std::cout<<torch::logical_not(bool_tensor)<<std::endl;

//    signed long data[] ={0,0,0,2,1,1,2};
//    std::vector<long> data{0,10,10,2,1,1,2,3,3,3};
//    int test= data.front();
//    std::cout<<test<<std::endl;
//    std::cout<<data<<std::endl;
//    pop_front(data);
//    std::cout<<test<<std::endl;
//    std::cout<<data<<std::endl;

//    int size = data.size();
//    auto ids = torch::from_blob(data.data(),{size},torch::kLong);
//    auto tmp = X.index({ids});
//    std::vector<torch::Tensor> bool_list = (tmp<=xmin+0.5*edge).unbind(1);
//    std::cout<<bool_list<<std::endl;
//    std::vector<torch::Tensor> bool_container;
//    std::vector<torch::Tensor> test = X.unbind(1);
//    std::cout<<X.slice(1,0,1)<<std::endl;
//    std::cout<<X<<std::endl;

//    for (int i=0; i<nd;i++){
//        auto rip = tmp.slice(1,i,i+1)<=(xmin+0.5*edge)[i];
//        bool_container.push_back(rip);
//        std::cout<< bool_container[i]<<std::endl; //get row slices...
//    }
//

    return 0;
}
//    torch::Tensor rip;



//    std::cout<< X.index({ids})<<std::endl; //get row slices...

//    std::cout<<edge<<std::endl;
//    std::cout<<xmin<<std::endl;
//    std::cout<<ymin<<std::endl;

    //    print_torch_cuda_1D<float><<<gridSize,blockSize,blockSize.x * (nd) * sizeof(float)>>>(tensor_cuda_a);
    /*
    int output_dim = 2;
    torch::Tensor tensor = torch::randn({nx, nd}).to(torch::Device("cuda:0"));
    torch::Tensor b = torch::randn({nx, output_dim}).to(torch::Device("cuda:0"));
    torch::Tensor output_tensor = torch::zeros_like(b).to(torch::Device("cuda:0"));
    torch::Tensor output_tensor_ref = torch::zeros_like(b).to(torch::Device("cuda:0"));

    auto tensor_cuda_a = tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
    auto tensor_cuda_output = output_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
    auto tensor_cuda_output_ref = output_tensor_ref.packed_accessor32<float,2,torch::RestrictPtrTraits>();
    auto tensor_cuda_b = b.packed_accessor32<float,2,torch::RestrictPtrTraits>();

    dim3 blockSize = get_blocksize();
    dim3 gridSize = get_gridsize(blockSize);
    conv_1d_torch_rbf<float><<<gridSize,blockSize,blockSize.x * (nd) * sizeof(float)>>>(tensor_cuda_a,tensor_cuda_a,tensor_cuda_b,tensor_cuda_output);
    cudaDeviceSynchronize();
    rbf_1d_reduce_simple_torch<float><<<gridSize,blockSize,blockSize.x * (nd) * sizeof(float)>>>(tensor_cuda_a,tensor_cuda_a,tensor_cuda_b,tensor_cuda_output_ref);
    cudaDeviceSynchronize();
    std::cout<< output_tensor<<std::endl;
    printf("-------------------------------------------\n");
    std::cout<< output_tensor_ref<<std::endl;
    */
//    auto tensor_a = tensor.accessor<float,2>();
//    printf("%f",tensor_a[0][0]);


    /** 1-d conv chunk of code

    print_row_mat(mat,nx,nd); //figure out how to move pointer of pointer to device!
    print_mat(std::cout,mat,N,M);
    auto cuda_mat_res = allocate_cuda_mat<float>(nx,ny);

    auto mat = generate_row_random_matrix(nx, nd); //matrix but contigous in the first element!
    print_row_mat(mat,nx,nd);
    auto cuda_mat_x = allocate_cuda_mat<float>(nx,nd);
    host_to_cuda(cuda_mat_x,mat[0],nx,nd); //moves row major matrix to cuda memory

    auto b = generate_row_random_matrix(nx,1);
    auto cuda_b = allocate_cuda_mat<float>(nx,1);
    host_to_cuda(cuda_b,b[0],nx,1); //moves row major matrix to cuda memory
    auto cuda_b_res = allocate_cuda_mat<float>(nx,1);
    auto cuda_kernel_mat = allocate_cuda_mat<float>(nx,ny);
    auto cuda_kernel_mat_2 = allocate_cuda_mat<float>(nx,ny);

    auto cuda_b_res_2 = allocate_cuda_mat<float>(nx,1);


//    print_row_mat(b,nx,1); //figure out how to move pointer of pointer to device!

//    print_mat_cuda<float><<<grid_X,BLOCK_SIZE>>>(cuda_mat_x);
//    cudaDeviceSynchronize();
//    rbf_1d_kernel<<<grid_X,BLOCK_SIZE>>>(cuda_mat_x,cuda_mat_x,cuda_mat_res);
//    int grid_X = max((int)ceil((float)nx/(float)BLOCK_SIZE),1);
    dim3 blockSize;
    int denonminator = std::max(1,(int) (nd*sizeof(float)));
    blockSize.x = min(BLOCK_SIZE,min(MAXTHREADSPERBLOCK,(int) ( (float)SHAREDMEMPERBLOCK / float(denonminator))));
    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);
    printf("%i \n",gridSize.x );

    rbf_1d_reduce_shared<<<gridSize, blockSize,blockSize.x * (nd) * sizeof(float)>>>(cuda_mat_x, cuda_mat_x, cuda_b, cuda_b_res);
    cudaDeviceSynchronize();
    auto cpu_res = cuda_to_host_and_pointer(cuda_b_res,nx,1);
    print_mat(std::cout,cpu_res,nx,1); //ok does something but very incorrectly
    printf("--------------------------------------------------------------------------------------------\n");
    rbf_1d_reduce_simple<<<gridSize, blockSize,blockSize.x * (nd) * sizeof(float)>>>(cuda_mat_x, cuda_mat_x, cuda_b, cuda_b_res_2);
    cudaDeviceSynchronize();
    auto cpu_res_2 = cuda_to_host_and_pointer(cuda_b_res_2,nx,1);
    print_mat(std::cout,cpu_res_2,nx,1); //ok does something but very incorrectly
**/

//    rbf_1d_kernel_shared<<<gridSize, blockSize,blockSize.x * (nd) * sizeof(float)>>>(cuda_mat_x, cuda_mat_x, cuda_kernel_mat);
//    cudaDeviceSynchronize();
//    auto cpu_res = cuda_to_host_and_pointer(cuda_kernel_mat,nx,ny);
//    print_mat(std::cout,cpu_res,nx,nx); //ok does something but very incorrectly
//    printf("\n");
//    rbf_1d_kernel<<<gridSize, blockSize,blockSize.x * (nd) * sizeof(float)>>>(cuda_mat_x, cuda_mat_x, cuda_kernel_mat_2);
//    cudaDeviceSynchronize();
//    auto cpu_res_2 = cuda_to_host_and_pointer(cuda_kernel_mat_2,nx,ny);
//    print_mat(std::cout,cpu_res_2,nx,nx); //ok does something but very incorrectly

//


//    auto cuda_kernel_mat = allocate_cuda_mat<float>(N,N);
//
//    std::cout<< grid_Y<<std::endl;
//    std::cout<< grid_X<<std::endl;
//
//    dim3 grid(grid_Y,grid_X);
//    dim3 block_dim(BLOCK_SIZE,BLOCK_SIZE);
//    host_to_cuda(cuda_mat,mat,N,M);
//    print_mat_cuda<float><<<grid,block_dim>>>(cuda_mat,M,N);
//    cudaDeviceSynchronize();


//    int n = 10;
//    int m = 10;
//    cumatrix<float> a(n, m), b(n, m),f(1,1); //this calls the constructors
//    cumatrix<float> c = a * b; //Do matmul, constructor, we initialize a new cumatrix<float> which is c in the function.
//
////    f = c; //this calls the copy ass
////    cumatrix<float> bub(a); //this calls the copy constructor
////    cumatrix<float> foo = test_move_constructor(cumatrix<float>(1,1)); //this calls the move constructor
////    c =  test_move_constructor(cumatrix<float>(1,1)); //this calls the move assignment op
