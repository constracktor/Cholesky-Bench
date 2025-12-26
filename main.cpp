#include <random>
#include <iostream>
#include <chrono>
#include <vector>

#include "mkl_adapter.hpp"
#include "tiled_cholesky.hpp"

#define CALC_TYPE double

int main(int argc, char* argv[])
{
  // loop size for averaging
  std::size_t n_loop = 1;
  // define exponents of 2
  std::size_t exp_start = 10;
  std::size_t exp_stop = 16;
  // runtime data holder
  std::size_t total_potrf;
  std::vector<CALC_TYPE> M_pos_1(pow(2,2*exp_stop));
  // timer
  auto t = std::chrono::steady_clock();
  // create logscale n vector
  std::vector<std::size_t> n_vector;
  for (size_t i = exp_start; i <= exp_stop; i++)
  {
      n_vector.push_back(pow(2, i));
  }
  // genereate header
  std::cout << "N;POTRF;TRSM;SYRK;GEMM;loop;" << n_loop << "\n";
  // short warm up
  std::size_t warmup = 0;
  for (size_t k = 0; k < 100000; k++)
  {
    warmup = warmup * warmup + 1;
  }
  // loop over logscale vector
  for (size_t k = 0; k < n_vector.size(); k++)
  {
    std::size_t n_dim = n_vector[k];
    std::size_t m_size = n_dim * n_dim;
    // reset data holders
    total_potrf = 0;
    // loop for averaging
    for (size_t loop = 0; loop < n_loop; loop++)
    {
        // //////////////////////////////////////////////////////////////////////////
        // // create random matrices
        // // setup number generator
        // size_t seed = (k + 1) * loop;
        // std::mt19937 generator ( seed );
        // std::uniform_real_distribution< CALC_TYPE > distribute( 0, 1 );
        // #pragma omp parallel for schedule(static)
        // for (size_t i = 0; i < n_dim; i++)
        // {
        //     // then create symmetric matrix
        //     for (size_t j = 0; j <= i; j++)
        //     {
        //         double avg = distribute( generator ); //0.5 * (M_pos_1[i * n_dim + j] + M_pos_1[j * n_dim + i]);
        //         M_pos_1[i * n_dim + j] = avg;
        //         M_pos_1[j * n_dim + i] = avg;
        //     }
        //     // add n_dim on diagonal
        //     M_pos_1[i * n_dim + i] = M_pos_1[i * n_dim + i] + 10* n_dim;
        // }
        // ////////////////////////////////////////////////////////////////////////////
        // // benchmark
        // // time cholesky decomposition
        // auto start_potrf = t.now();
        // mkl_potrf(M_pos_1, n_dim);
        // auto stop_potrf = t.now();
        const int n_tiles = 32;
        const int tile_size = n_dim / n_tiles;
        auto test = gen_tiled_matrix(tile_size, n_tiles);

        auto start_potrf = t.now();
        right_looking_cholesky_tiled(test, tile_size, n_tiles);
        auto stop_potrf = t.now();
        ////////////////////////////////////////////////////////////////////////////
        // add time difference to total time
        total_potrf += std::chrono::duration_cast<std::chrono::microseconds>(stop_potrf - start_potrf).count();
    }
    std::cout <<  n_dim << ";"
              <<  total_potrf / 1000000.0 / n_loop << ";\n";
  }


}
