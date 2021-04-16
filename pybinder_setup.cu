
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
m.def("FFM_X_FLOAT_5", &FFM_X<float,5>, "xx 5");
m.def("SUPERSMOOTH_FFM_X_FLOAT_5", &SUPERSMOOTH_FFM_X<float,5>, "smooth xx 5");
m.def("FFM_XY_FLOAT_5", &FFM_XY<float,5>, "xy 5");

m.def("FFM_X_FLOAT_4", &FFM_X<float,4>, "xx 4");
m.def("SUPERSMOOTH_FFM_X_FLOAT_4", &SUPERSMOOTH_FFM_X<float,4>, "smooth xx 4");
m.def("FFM_XY_FLOAT_4", &FFM_XY<float,4>, "xy 4");


m.def("FFM_X_FLOAT_3", &FFM_X<float,3>, "xx 3");
m.def("SUPERSMOOTH_FFM_X_FLOAT_3", &SUPERSMOOTH_FFM_X<float,3>, "smooth xx 3");
m.def("FFM_XY_FLOAT_3", &FFM_XY<float,3>, "xy 3");

m.def("FFM_X_FLOAT_2", &FFM_X<float,2>, "xx 2");
m.def("SUPERSMOOTH_FFM_X_FLOAT_2", &SUPERSMOOTH_FFM_X<float,2>, "smooth xx 2");
m.def("FFM_XY_FLOAT_2", &FFM_XY<float,2>, "xy 2");

m.def("FFM_X_FLOAT_1", &FFM_X<float,1>, "xx 1");
m.def("SUPERSMOOTH_FFM_X_FLOAT_1", &SUPERSMOOTH_FFM_X<float,1>, "smooth xx 1");
m.def("FFM_XY_FLOAT_1", &FFM_XY<float,1>, "xy 1");
/*
 * RBF benchmark
 * */

m.def("rbf_float_5", &rbf_call<float,5>, "rbf xy 5 dim float");
m.def("rbf_float_4", &rbf_call<float,4>, "rbf xy 4 dim float");
m.def("rbf_float_3", &rbf_call<float,3>, "rbf xy 3 dim float");
m.def("rbf_float_2", &rbf_call<float,2>, "rbf xy 2 dim float");
m.def("rbf_float_1", &rbf_call<float,1>, "rbf xy 1 dim float");


}
