#include "functions.hpp"
#include "tile_generation.hpp"
#ifdef ENABLE_VALIDATION
#include "validate.hpp"
#endif
#include <fstream>
#include <hpx/hpx_main.hpp>
#include <iostream>
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
    using namespace hpx::program_options;
    options_description opts;
    opts.add_options()("loop", value<std::size_t>()->default_value(1), "Number of repititions")(
        "size_start", value<std::size_t>()->default_value(32), "Start problem size")(
        "size_stop", value<std::size_t>()->default_value(128), "Stop problem size")(
        "tiles_start", value<std::size_t>()->default_value(16), "Start tiles per dimension")(
        "tiles_stop", value<std::size_t>()->default_value(32), "Stop tiles per dimension");
    variables_map vm;
    store(parse_command_line(argc, argv, opts), vm);
    notify(vm);
    ///////////////////////////////////////////////////////////////////////////
    // configuration
    const std::size_t LOOP = vm["loop"].as<std::size_t>();

    const std::size_t START_SIZE = vm["size_start"].as<std::size_t>();
    const std::size_t STOP_SIZE = vm["size_stop"].as<std::size_t>();
    const std::size_t STEP_SIZE = 2;

    const std::size_t START_TILES = vm["tiles_start"].as<std::size_t>();
    const std::size_t STOP_TILES = vm["tiles_stop"].as<std::size_t>();
    const std::size_t STEP_TILES = 2;

    // print and write results
    bool HEADER_FLAG = true;
    std::string runtime_file_path = "runtimes_hpx_cholesky_";
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
                std::string values = std::to_string(hpx::get_num_worker_threads());
                values += std::string(";") + std::to_string(size);
                values += std::string(";") + std::to_string(size / n_tiles);
                values += std::string(";") + std::to_string(n_tiles);
#ifdef ENABLE_VALIDATION
                // Relative residual ||A - L L^T||_F / ||A||_F. 1e-10
                // is a loose-but-safe bound for an FP64 tiled
                // Cholesky on the problem sizes this benchmark
                // exercises. Compiled in only when the CMake option
                // ENABLE_VALIDATION is set; not written to the CSV
                // output file - purely console.
                constexpr double residual_tol = 1e-10;
                auto report_residual = [&](const std::string &mode, double residual) {
                    std::cout << "[validate] mode=" << mode << " size=" << size << " n_tiles=" << n_tiles
                              << " residual=" << residual << std::endl;
                    if (!(residual <= residual_tol))  // catches NaN too
                    {
                        std::cerr << "Validation warning: variant '" << mode << "' residual " << residual
                                  << " exceeds tolerance " << residual_tol << " (size=" << size
                                  << ", n_tiles=" << n_tiles << ")" << std::endl;
                    }
                };
#endif

                ///////////////////////////////////////////////////////////////////////////
                // futurized
                std::vector<std::string> f_modes = { "async_future", "sync_future" };
                for (const auto &mode : f_modes)
                {
                    auto f_tiled_matrix = gen_futurized_tiled_matrix(size, n_tiles);
                    auto cholesky_cpu = cpu::cholesky_future(f_tiled_matrix, mode);

                    header += ";" + mode;
                    values += ";" + std::to_string(cholesky_cpu);

#ifdef ENABLE_VALIDATION
                    // All futures are ready by now (wait_all inside cholesky_future);
                    // dereference each into a plain matrix for the residual check.
                    Tiled_vector_matrix L(n_tiles * n_tiles);
                    for (std::size_t i = 0; i < n_tiles; ++i)
                    {
                        for (std::size_t j = 0; j <= i; ++j)
                        {
                            L[i * n_tiles + j] = f_tiled_matrix[i * n_tiles + j].get();
                        }
                    }
                    double residual = cpu::cholesky_residual(size, n_tiles, L);
                    report_residual(mode, residual);
#endif
                }
                ///////////////////////////////////////////////////////////////////////////
                // loop
                std::vector<std::string> loop_modes = { "loop_one", "loop_two" };
                for (const auto &mode : loop_modes)
                {
                    auto tiled_matrix = gen_tiled_matrix(size, n_tiles);
                    auto cholesky_cpu = cpu::cholesky_loop(tiled_matrix, mode);

                    header += ";" + mode;
                    values += ";" + std::to_string(cholesky_cpu);

#ifdef ENABLE_VALIDATION
                    double residual = cpu::cholesky_residual(size, n_tiles, tiled_matrix);
                    report_residual(mode, residual);
#endif
                }
                ///////////////////////////////////////////////////////////////////////////
                // void-future variant (no vector copies in BLAS operations)
                {
                    Tiled_vector_matrix tiles;
                    Tiled_void_matrix dep_tiles;
                    gen_void_tiled_matrix(tiles, dep_tiles, size, n_tiles);
                    auto cholesky_cpu = cpu::cholesky_void(tiles, dep_tiles, n_tiles);

                    header += ";async_void";
                    values += ";" + std::to_string(cholesky_cpu);

#ifdef ENABLE_VALIDATION
                    double residual = cpu::cholesky_residual(size, n_tiles, tiles);
                    report_residual("async_void", residual);
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
