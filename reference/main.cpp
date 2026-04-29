#include "functions.hpp"
#include "matrix_generation.hpp"
#ifdef ENABLE_VALIDATION
#include "validate.hpp"
#endif
#ifdef ENABLE_PLASMA
#include <plasma.h>
#endif
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char *argv[])
{
    ///////////////////////////////////////////////////////////////////////////
    // cmdline arguments
    //
    // The reference benchmark calls a single threaded LAPACKE_dpotrf on the
    // full matrix, so there is no tiling axis. We still accept --tiles_start
    // / --tiles_stop for CLI compatibility with the openmp/ and hpx/ binaries
    // (they are silently ignored), which keeps any shared driver script
    // unchanged.
    std::size_t loop = 1;
    std::size_t size_start = 32, size_stop = 128;

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
        else if ((arg == "--tiles_start" || arg == "--tiles_stop") && i + 1 < argc)
        {
            // Accept-and-ignore for CLI parity with the tiled variants.
            ++i;
        }
    }
    ///////////////////////////////////////////////////////////////////////////
    // configuration
    const std::size_t LOOP = loop;

    const std::size_t START_SIZE = size_start;
    const std::size_t STOP_SIZE = size_stop;
    const std::size_t STEP_SIZE = 2;

    // print and write results
    bool HEADER_FLAG = true;
    std::string runtime_file_path = "runtimes_reference_cholesky_";
    if (START_SIZE != STOP_SIZE)
    {
        runtime_file_path += std::string("size_");
    }
    runtime_file_path += std::to_string(LOOP) + std::string(".txt");
    std::ofstream runtime_file;
    runtime_file.open(runtime_file_path, std::ios_base::app);

#ifdef ENABLE_PLASMA
    // PLASMA spins up its own context and worker pool; do this once so the
    // cost is not folded into any timed factorisation.
    if (plasma_init() != 0)
    {
        throw std::runtime_error("plasma_init() failed");
    }
#endif

    for (std::size_t size = START_SIZE; size <= STOP_SIZE; size = size * STEP_SIZE)
    {
        for (std::size_t l = 0; l < LOOP; l++)
        {
            // header for output file -- columns mirror the openmp/hpx output so
            // results from all three benchmarks can be merged on (problem_size).
            // The reference has no tiling, so tile_size == problem_size and
            // n_tiles == 1.
            std::string header = "threads;problem_size;tile_size;n_tiles";
            std::string values = std::to_string(omp_get_max_threads());
            values += std::string(";") + std::to_string(size);
            values += std::string(";") + std::to_string(size);
            values += std::string(";") + std::to_string(1);
            ///////////////////////////////////////////////////////////////////
            // Reference modes:
            //   reference   -> single threaded LAPACKE_dpotrf2 on the full
            //                  matrix (currently disabled; uncomment the
            //                  initializer below to re-enable)
            //   plasma      -> single plasma_dpotrf (high-level synchronous
            //                  PLASMA API; added when ENABLE_PLASMA=ON)
            //   plasma_tile -> plasma_omp_dpotrf over a manually-built
            //                  plasma_desc_t (PLASMA's asynchronous tile
            //                  interface; added when ENABLE_PLASMA=ON)
            std::vector<std::string> modes = {
                // "reference",
            };
#ifdef ENABLE_PLASMA
            modes.push_back("plasma");
            modes.push_back("plasma_tile");
#endif

            for (const auto &mode : modes)
            {
                header += ";" + mode;

                // We let one mode fail (e.g. PLASMA running out of memory at
                // very large N -- its high-level wrapper allocates an extra
                // tiled triangular copy on top of the input buffer) without
                // killing the whole sweep. The failed cell is recorded as NaN
                // and we continue with the next mode and size.
                std::vector<double> A;
                try
                {
                    A = gen_matrix(size);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error: gen_matrix(size=" << size << ") threw '" << e.what()
                              << "'. Recording NaN for variant '" << mode << "'." << std::endl;
                    values += ";nan";
                    continue;
                }

                double cholesky_cpu = std::numeric_limits<double>::quiet_NaN();
                try
                {
                    cholesky_cpu = cpu::cholesky(A, size, mode);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error: variant '" << mode << "' failed at size=" << size << ": " << e.what()
                              << ". Recording NaN and continuing." << std::endl;
                    values += ";nan";
                    continue;
                }

                values += ";" + std::to_string(cholesky_cpu);

#ifdef ENABLE_VALIDATION
                // Validate by computing relative residual ||A - L L^T||_F / ||A||_F
                constexpr double residual_tol = 1e-10;
                const double residual = cpu::cholesky_residual(size, A);
                std::cout << "[validate] mode=" << mode << " size=" << size << " residual=" << residual << std::endl;
                if (!(residual <= residual_tol))  // catches NaN too
                {
                    std::cerr << "Validation warning: variant '" << mode << "' residual " << residual
                              << " exceeds tolerance " << residual_tol << " (size=" << size << ")" << std::endl;
                }
#endif
            }
            ///////////////////////////////////////////////////////////////////
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

    runtime_file.close();

#ifdef ENABLE_PLASMA
    plasma_finalize();
#endif

    return 0;
}
