
#pragma once
#include "n_tree.cu"
#include <torch/extension.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    py::class_<FFM_object<float,3>>(m, "FFM_3D_FLOAT_CLASS").def(py::init<torch::Tensor &,torch::Tensor & ,
//                    float & ,
//    const std::string & ,
//    float & ,
//    int & ,
//    bool & ,
//    float  & ,
//    bool & >()).def("__mul__",&FFM_object<float,3>::operator*,py::is_operator());

m.def("FFM_XY_FLOAT_10", &FFM_XY<float,10>, "xy 10");

m.def("FFM_XY_FLOAT_9", &FFM_XY<float,9>, "xy 9");

m.def("FFM_XY_FLOAT_8", &FFM_XY<float,8>, "xy 8");

m.def("FFM_XY_FLOAT_7", &FFM_XY<float,7>, "xy 5");

m.def("FFM_XY_FLOAT_6", &FFM_XY<float,6>, "xy 6");

m.def("FFM_XY_FLOAT_5", &FFM_XY<float,5>, "xy 5");

m.def("FFM_XY_FLOAT_4", &FFM_XY<float,4>, "xy 4");

m.def("FFM_XY_FLOAT_3", &FFM_XY<float,3>, "xy 3");

m.def("FFM_XY_FLOAT_2", &FFM_XY<float,2>, "xy 2");

//m.def("SUPERSMOOTH_FFM_FLOAT_1", &SUPERSMOOTH_FFM<float,1>, "smooth xx 1");
m.def("FFM_XY_FLOAT_1", &FFM_XY<float,1>, "xy 1");
/*
 * RBF benchmark
 * */
m.def("rbf_float_10", &rbf_call<float,10>, "rbf xy 10 dim float");
m.def("rbf_float_9", &rbf_call<float,9>, "rbf xy 9 dim float");
m.def("rbf_float_8", &rbf_call<float,8>, "rbf xy 8 dim float");
m.def("rbf_float_7", &rbf_call<float,7>, "rbf xy 7 dim float");
m.def("rbf_float_6", &rbf_call<float,6>, "rbf xy 6 dim float");
m.def("rbf_float_5", &rbf_call<float,5>, "rbf xy 5 dim float");
m.def("rbf_float_4", &rbf_call<float,4>, "rbf xy 4 dim float");
m.def("rbf_float_3", &rbf_call<float,3>, "rbf xy 3 dim float");
m.def("rbf_float_2", &rbf_call<float,2>, "rbf xy 2 dim float");
m.def("rbf_float_1", &rbf_call<float,1>, "rbf xy 1 dim float");

m.def("rbf_double_10", &rbf_call<double,10>, "rbf xy 10 dim double");
m.def("rbf_double_9", &rbf_call<double,9>, "rbf xy 9 dim double");
m.def("rbf_double_8", &rbf_call<double,8>, "rbf xy 8 dim double");
m.def("rbf_double_7", &rbf_call<double,7>, "rbf xy 7 dim double");
m.def("rbf_double_6", &rbf_call<double,6>, "rbf xy 6 dim double");
m.def("rbf_double_5", &rbf_call<double,5>, "rbf xy 5 dim double");
m.def("rbf_double_4", &rbf_call<double,4>, "rbf xy 4 dim double");
m.def("rbf_double_3", &rbf_call<double,3>, "rbf xy 3 dim double");
m.def("rbf_double_2", &rbf_call<double,2>, "rbf xy 2 dim double");
m.def("rbf_double_1", &rbf_call<double,1>, "rbf xy 1 dim double");


}
