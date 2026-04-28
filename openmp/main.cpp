#include "functions.hpp"
#include "tile_generation.hpp"
#ifdef ENABLE_VALIDATION
#include "validate.hpp"
#endif
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

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
                std::vector<std::string> modes = { "for_collapse", "for_naive", "task_naive", "task_depend" };
                //,"task_prio" };

                for (const auto &mode : modes)
                {
                    auto tiled_matrix = gen_tiled_matrix(size, n_tiles);
                    auto cholesky_cpu = cpu::cholesky(tiled_matrix, mode);

                    header += ";" + mode;
                    values += ";" + std::to_string(cholesky_cpu);

#ifdef ENABLE_VALIDATION
                    // Validate by computing relative residual ||A - L L^T||_F / ||A||_F
                    constexpr double residual_tol = 1e-10;
                    const double residual = cpu::cholesky_residual(size, n_tiles, tiled_matrix);
                    std::cout << "[validate] mode=" << mode << " size=" << size << " n_tiles=" << n_tiles
                              << " residual=" << residual << std::endl;
                    if (!(residual <= residual_tol))  // catches NaN too
                    {
                        std::cerr << "Validation warning: variant '" << mode << "' residual " << residual
                                  << " exceeds tolerance " << residual_tol << " (size=" << size
                                  << ", n_tiles=" << n_tiles << ")" << std::endl;
                    }
#endif
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
