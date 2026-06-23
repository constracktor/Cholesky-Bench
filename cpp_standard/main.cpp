#include "functions.hpp"
#include "task_backend.hpp"
#include "tile_generation.hpp"
#ifdef ENABLE_VALIDATION
#include "validate.hpp"
#endif
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

// Minimal command-line parser replacing hpx::program_options. Accepts both
// "--key value" and "--key=value" spellings for the options below.
struct Options
{
    std::size_t loop = 1;
    std::size_t size_start = 32;
    std::size_t size_stop = 128;
    std::size_t tiles_start = 16;
    std::size_t tiles_stop = 32;
};

[[noreturn]] void usage_and_exit(const char *prog, int code)
{
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --loop=N          Number of repetitions (default 1)\n"
              << "  --size_start=N    Start problem size (default 32)\n"
              << "  --size_stop=N     Stop problem size (default 128)\n"
              << "  --tiles_start=N   Start tiles per dimension (default 16)\n"
              << "  --tiles_stop=N    Stop tiles per dimension (default 32)\n";
    std::exit(code);
}

std::size_t parse_size(const std::string &key, const std::string &val, const char *prog)
{
    try
    {
        return static_cast<std::size_t>(std::stoull(val));
    }
    catch (const std::exception &)
    {
        std::cerr << "Invalid value for " << key << ": '" << val << "'\n";
        usage_and_exit(prog, 1);
    }
}

Options parse_args(int argc, char *argv[])
{
    Options opt;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help")
        {
            usage_and_exit(argv[0], 0);
        }
        if (arg.rfind("--", 0) != 0)
        {
            std::cerr << "Unexpected argument: '" << arg << "'\n";
            usage_and_exit(argv[0], 1);
        }
        // Split "--key=value"; otherwise consume the next argv as the value.
        std::string key, val;
        auto eq = arg.find('=');
        if (eq != std::string::npos)
        {
            key = arg.substr(2, eq - 2);
            val = arg.substr(eq + 1);
        }
        else
        {
            key = arg.substr(2);
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for --" << key << "\n";
                usage_and_exit(argv[0], 1);
            }
            val = argv[++i];
        }

        if (key == "loop")
        {
            opt.loop = parse_size(key, val, argv[0]);
        }
        else if (key == "size_start")
        {
            opt.size_start = parse_size(key, val, argv[0]);
        }
        else if (key == "size_stop")
        {
            opt.size_stop = parse_size(key, val, argv[0]);
        }
        else if (key == "tiles_start")
        {
            opt.tiles_start = parse_size(key, val, argv[0]);
        }
        else if (key == "tiles_stop")
        {
            opt.tiles_stop = parse_size(key, val, argv[0]);
        }
        else
        {
            std::cerr << "Unknown option: --" << key << "\n";
            usage_and_exit(argv[0], 1);
        }
    }
    return opt;
}

}  // namespace

int main(int argc, char *argv[])
{
    ///////////////////////////////////////////////////////////////////////////
    // cmdline arguments
    const Options opt = parse_args(argc, argv);
    ///////////////////////////////////////////////////////////////////////////
    // configuration
    const std::size_t LOOP = opt.loop;

    const std::size_t START_SIZE = opt.size_start;
    const std::size_t STOP_SIZE = opt.size_stop;
    const std::size_t STEP_SIZE = 2;

    const std::size_t START_TILES = opt.tiles_start;
    const std::size_t STOP_TILES = opt.tiles_stop;
    const std::size_t STEP_TILES = 2;

    // print and write results
    bool HEADER_FLAG = true;
    std::string runtime_file_path = "runtimes_std_cholesky_";
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
                // header for output file
                std::string header = "problem_size;tile_size;n_tiles";
                // runtime config and values
                std::string values = std::to_string(size);
                values += std::string(";") + std::to_string(size / n_tiles);
                values += std::string(";") + std::to_string(n_tiles);
#ifdef ENABLE_VALIDATION
                // Relative residual ||A - L L^T||_F / ||A||_F
                constexpr double residual_tol = 1e-10;
                auto report_residual = [&](const std::string &mode, double residual)
                {
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
                // void-futures: tile data comes from gen_tiled_matrix (same as the
                // loop variants); only the dependency-future matrix is variant-specific.
                {
                    auto tiles = gen_tiled_matrix(size, n_tiles);
                    auto dep_tiles = gen_void_tiled_matrix(n_tiles);
                    auto cholesky_cpu = cpu::cholesky_void(tiles, dep_tiles);

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
