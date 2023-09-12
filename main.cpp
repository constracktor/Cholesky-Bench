#include <cmath>
#include <random>
#include <iostream>
#include <chrono>
#include <vector>

#include "mkl_adapter.hpp"

#define CALC_TYPE double

int main(int argc, char* argv[])
{
  // loop size for averaging
  std::size_t n_loop = 100;
  // define exponents of 10
  std::size_t exp_start = 1;
  std::size_t exp_stop = 3;
  // define early stop;
  std::size_t early_stop = 10 * pow(10, exp_stop - 1);
  // runtime data holder
  std::size_t total_potrf;
  std::size_t total_trsm;
  std::size_t total_syrk
  std::size_t total_gemm;
  // timer
  auto t = std::chrono::high_resolution_timer();
  // create logscale n vector
  std::vector<std::size_t> n_vector;
  n_vector.resize(9 * (exp_stop - exp_start) + 1);
  for (size_t i = exp_start; i < exp_stop; i++)
  {
    for (size_t j = 1; j < 10; j++)
    {
      n_vector[(i - exp_start) * 9 + j - 1] = j * pow(10, i);
    }
  }
  n_vector[9 * (exp_stop - exp_start)] = pow(10, exp_stop);
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
    // early stopping
    if (early_stop < n_dim)
    {
      break;
    }
    // reset data holders
    total_potrf = 0;
    total_trsm = 0;
    total_syrk = 0;
    total_gemm = 0;
    // loop for averaging
    for (size_t loop = 0; loop < n_loop; loop++)
    {
        //////////////////////////////////////////////////////////////////////////
        // create random matrices
        // setup number generator
        size_t seed = (k + 1) * loop;
        std::mt19937 generator ( seed );
        std::uniform_real_distribution< CALC_TYPE > distribute( 0, 1 );
        // create two positive definite matrices
        // first create random matrices
        std::vector<CALC_TYPE> M_pos_1(m_size);
        std::vector<CALC_TYPE> M_pos_2(m_size);
        std::vector<CALC_TYPE> M_1(m_size);
        std::vector<CALC_TYPE> M_2(m_size);
        std::vector<CALC_TYPE> M_2(m_size);
        // initialize matrices with random values
        for (size_t i = 0; i < m_size; i++)
        {
            M_pos_1[i] = distribute( generator );
            M_1[i] = distribute( generator );
        }
        // then create symmetric matrix
        for (size_t i = 0; i < n_dim; i++)
        {
        for (size_t j = 0; j <= i; j++)
        {
            M_pos_1[i * n_dim + j] = 0.5 * (M_pos_1[i * n_dim + j] + M_pos_1[j * n_dim + i]);
            M_pos_1[j * n_dim + i] = M_pos_1[i * n_dim + j];
        }
        }
        // then add n_dim on diagonal
        for (size_t i = 0; i < n_dim; i++)
        {
            M_pos_1[i * n_dim + i] = M_pos_1[i * n_dim + i] + n_dim;
        }
        //
        M_pos_2 = std::copy(M_pos_1);
        M_2 = std::copy(M_1);
        M_3 = std::copy(M_1);
        ////////////////////////////////////////////////////////////////////////////
        // benchmark
        // time cholesky decomposition
        auto start_potrf = t.now();
        potrf(M_pos_1, n_dim);
        auto stop_potrf = t.now();
        // time triangular solve
        auto start_trsm = t.now();
        trsm(M_1, M_pos_1, n_dim);
        auto stop_trsm = t.now();
        trsm(M_2, M_pos_1, n_dim);
        // time symmetrik k rank update solve
        auto start_syrk = t.now();
        syrk(M_pos_2, M_1, n_dim);
        auto stop_syrk = t.now();
        // time matrix multiplication
        auto start_gemm = t.now();
        gemm(M_3, M_1, M_2, n_dim);
        auto stop_gemm = t.now();
        ////////////////////////////////////////////////////////////////////////////
        // add time difference to total time
        total_potrf += stop_potrf - start_potrf;
        total_trsm += stop_trsm - start_trsm;
        total_syrk += stop_syrk - start_syrk;
        total_gemm += stop_gemm - start_gemm;
    }
    std::cout <<  n_dim << ";"
              <<  total_potrf / n_loop << ";"
              <<  total_trsm / n_loop << ";"
              <<  total_gemm / n_loop << ";\n";
  }
}