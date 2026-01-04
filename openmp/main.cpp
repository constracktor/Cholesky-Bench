#include "functions.hpp"
#include "tile_generation.hpp"
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

// bool are_identical(const std::vector<std::vector<double>> &A,
//                    const std::vector<std::vector<double>> &B,
//                    double tol = 1e-14)
// {
//     if (A.size() != B.size())
//     {
//         std::cout << "Size mismatch: rows " << A.size() << " vs " << B.size() << std::endl;
//         return false;
//     }
//
//     for (std::size_t i = 0; i < A.size(); ++i)
//     {
//         if (A[i].size() != B[i].size())
//         {
//             std::cout << "Size mismatch at row " << i << ": cols " << A[i].size() << " vs " << B[i].size() <<
//             std::endl; return false;
//         }
//
//         for (std::size_t j = 0; j < A[i].size(); ++j)
//         {
//             double diff = std::abs(A[i][j] - B[i][j]);
//             if (diff > tol)
//             {
//                 std::cout << "Mismatch at (" << i << "," << j << ")  " << "cpu=" << B[i][j] << " gpu=" << A[i][j]
//                           << " diff=" << diff << std::endl;
//                 return false;
//             }
//         }
//     }
//
//     return true;
// }

int main(int argc, char *argv[])
{
    ///////////////////////////////////////////////////////////////////////////
    // cmdline arguments
    std::size_t loop = 1;
    std::size_t size_start = 32, size_stop = 128;
    std::size_t tiles_start = 16, tiles_stop = 32;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--loop" && i + 1 < argc)
        {
            loop = std::stoul(argv[++i]);
        }
        else if (arg == "--size_start" && i + 1 < argc)
        {
            size_start = std::stoul(argv[++i]);
        }
        else if (arg == "--size_stop" && i + 1 < argc)
        {
            size_stop = std::stoul(argv[++i]);
        }
        else if (arg == "--tiles_start" && i + 1 < argc)
        {
            tiles_start = std::stoul(argv[++i]);
        }
        else if (arg == "--tiles_stop" && i + 1 < argc)
        {
            tiles_stop = std::stoul(argv[++i]);
        }
    }
    ///////////////////////////////////////////////////////////////////////////
    // configuration
    const std::size_t LOOP = loop;

    const std::size_t START_SIZE = size_start;
    const std::size_t STOP_SIZE = size_stop;
    const std::size_t STEP_SIZE = 2;

    const std::size_t START_TILES = tiles_start;
    const std::size_t STOP_TILES = tiles_stop;
    const std::size_t STEP_TILES = 2;

    // print and write results
    bool HEADER_FLAG = true;
    std::string runtime_file_path = "runtimes_openmp_cholesky_";
    if (START_TILES != STOP_TILES)
    {
        runtime_file_path += std::string("tile_");
    }
    if (START_SIZE != STOP_SIZE)
    {
        runtime_file_path += std::string("size_");
    }
    runtime_file_path += std::to_string(LOOP) + std::string(".txt");
    std::ofstream runtime_file;
    runtime_file.open(runtime_file_path, std::ios_base::app);

    for (std::size_t n_tiles = START_TILES; n_tiles <= STOP_TILES; n_tiles = n_tiles * STEP_TILES)
    {
        for (std::size_t size = START_SIZE; size <= STOP_SIZE; size = size * STEP_SIZE)
        {
            for (std::size_t l = 0; l < LOOP; l++)
            {
                std::size_t tile_size = size / n_tiles;
                // header for output file
                std::string header = "threads;problem_size;tile_size;n_tiles";
                // runtime config and values
                std::string values = std::to_string(omp_get_max_threads());
                values += std::string(";") + std::to_string(size);
                values += std::string(";") + std::to_string(size / n_tiles);
                values += std::string(";") + std::to_string(n_tiles);
                ///////////////////////////////////////////////////////////////////////////
                std::vector<std::string> modes = { "for_collapse",
                                                   "for_naive",

                                                   "task_naive",
                                                   "task_depend" };

                for (const auto &mode : modes)
                {
                    auto tiled_matrix = gen_tiled_matrix(size, n_tiles);
                    auto cholesky_cpu = cpu::cholesky(tiled_matrix, mode);

                    header += ";" + mode;
                    values += ";" + std::to_string(cholesky_cpu);
                }
                ///////////////////////////////////////////////////////////////////////////
                // print/write header only once
                if (HEADER_FLAG)
                {
                    HEADER_FLAG = false;
                    std::cout << header << std::endl;
                    runtime_file << header << std::endl;
                }
                // print/write runtimes
                std::cout << values << std::endl;
                runtime_file << values << std::endl;
            }
        }
    }

    runtime_file.close();
    return 0;
}
