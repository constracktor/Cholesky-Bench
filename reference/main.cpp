#include "functions.hpp"
#include "matrix_generation.hpp"
#ifdef ENABLE_VALIDATION
#include "validate.hpp"
#endif
#ifdef ENABLE_PLASMA
#include <plasma.h>
#endif
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
    if (plasma_init() != 0)
    {
        throw std::runtime_error("plasma_init() failed");
    }
#endif

    for (std::size_t size = START_SIZE; size <= STOP_SIZE; size = size * STEP_SIZE)
    {
        for (std::size_t l = 0; l < LOOP; l++)
        {
            std::string header = "threads;problem_size;tile_size;n_tiles";
            std::string values = std::to_string(omp_get_max_threads());
            values += std::string(";") + std::to_string(size);
            values += std::string(";") + std::to_string(size);
            values += std::string(";") + std::to_string(1);
            ///////////////////////////////////////////////////////////////////
            // Reference modes:
            std::vector<std::string> modes = {};
#ifdef ENABLE_LAPACKE
            modes.push_back("lapacke");
#endif
#ifdef ENABLE_PLASMA
            modes.push_back("plasma");
#endif

            for (const auto &mode : modes)
            {
                header += ";" + mode;
                std::size_t mode_size = size;

                // PLASMA's triangular descriptor allocation
                // overflows int32 for N>65280 with the default nb=256. For
                // input sizes in (65280, 65536] we silently clamp PLASMA's
                // working size down to 65280;                std::size_t mode_size = size;
                if (mode == "plasma" && mode_size > 65'280 && mode_size <= 65'536)
                {
                    mode_size = 65'280;
                }

                std::vector<double> matrix = gen_matrix(mode_size);
                // NaN guard
                double cholesky_cpu = std::numeric_limits<double>::quiet_NaN();
                try
                {
                    cholesky_cpu = cpu::cholesky(matrix, mode_size, mode);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error: variant '" << mode << "' failed at size=" << mode_size << ": " << e.what()
                              << ". Recording NaN and continuing." << std::endl;
                    values += ";nan";
                    continue;
                }

                values += ";" + std::to_string(cholesky_cpu);

#ifdef ENABLE_VALIDATION
                // Validate by computing relative residual ||A - L L^T||_F / ||A||_F
                constexpr double residual_tol = 1e-10;
                const double residual = cpu::cholesky_residual(mode_size, matrix);
                std::cout << "[validate] mode=" << mode << " size=" << mode_size << " residual=" << residual
                          << std::endl;
                if (!(residual <= residual_tol))  // catches NaN too
                {
                    std::cerr << "Validation warning: variant '" << mode << "' residual " << residual
                              << " exceeds tolerance " << residual_tol << " (size=" << mode_size << ")" << std::endl;
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
