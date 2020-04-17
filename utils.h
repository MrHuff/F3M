//
// Created by rhu on 2020-04-17.
//

#pragma once

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
