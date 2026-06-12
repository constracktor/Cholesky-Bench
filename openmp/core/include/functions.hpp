#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

#pragma once

#include <string>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;

namespace cpu
{

double cholesky(Tiled_vector_matrix &tiled_matrix, const std::string &variant);

}  // namespace cpu
#endif  // end of CPU_FUNCTIONS_H
