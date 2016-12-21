
#include <flann/flann.hpp>

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <ctime>


using namespace flann;

int main(int argc, char** argv)
{
    std::chrono::time_point<std::chrono::system_clock> start, end;

    int nn = 3;
    int n_dp = 35000000;
    //int n_dp = 3500;
    int n_dim = 40;

    std::vector<float>  data(n_dim*n_dp,0);
    std::vector<int>    indices_mem(n_dp*nn,0);
    std::vector<float>  dists_mem(n_dp*nn,0);

    for(int i = 0; i < n_dp*n_dim; ++i){
        data[i] = (std::rand()%1000)/1000.;
    }

    Matrix<float> dataset(data.data(),n_dp,n_dim);
    Matrix<float> query(data.data(),n_dp,n_dim);


    Matrix<int> indices(indices_mem.data(), query.rows, nn);
    Matrix<float> dists(dists_mem.data(), query.rows, nn);

    // construct an randomized kd-tree index using 4 kd-trees
    Index<L2<float> > index(dataset, flann::KDTreeIndexParams(4));
    start = std::chrono::system_clock::now();
    index.buildIndex();
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "memory: " << index.usedMemory() << "\n";

    // do a knn search, using 128 checks
    //index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

    
    return 0;
}
