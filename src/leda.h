#ifndef LEDA_H
#define LEDA_H

#include <ap_int.h>
#include <tapa.h>

#define VALUE_TYPE float
#define INDEX_TYPE int

constexpr INDEX_TYPE HBM_CHANNEL_A_NUM = 8;
constexpr INDEX_TYPE HBM_CHANNEL_B_NUM = 4;
constexpr INDEX_TYPE HBM_CHANNEL_C_NUM = 8;

constexpr INDEX_TYPE UNIT_NUM = 2;

constexpr INDEX_TYPE PE_NUM = 8; 

constexpr INDEX_TYPE FIFO_DEPTH = 2;

constexpr INDEX_TYPE Tile_SIZE = 16;

constexpr INDEX_TYPE BATCH_SIZE = 4096 / Tile_SIZE;

const INDEX_TYPE Tile_WIDTH = BATCH_SIZE * Tile_SIZE;

const INDEX_TYPE B_PARTITION_FACTOR = 4;

const INDEX_TYPE URAM_DEPTH = 8192;

const INDEX_TYPE WINDOWS = 10;

using VALUE_TYPE_v16 = tapa::vec_t<VALUE_TYPE, 16>;
using VALUE_TYPE_v8  = tapa::vec_t<VALUE_TYPE, 8>;

void Leda(tapa::mmap<INDEX_TYPE> SpElement_list_ptr,
          tapa::mmaps<ap_uint<512>, HBM_CHANNEL_A_NUM> Matrix_A_data,
          tapa::mmaps<VALUE_TYPE_v16, HBM_CHANNEL_B_NUM> Matrix_B_data,
          tapa::mmaps<VALUE_TYPE_v16, HBM_CHANNEL_C_NUM> Matrix_C_data,

          const INDEX_TYPE Batch_num, 
          const INDEX_TYPE Sparse_Matrix_len, 
          const INDEX_TYPE M, 
          const INDEX_TYPE K,
          const INDEX_TYPE N,
          const INDEX_TYPE Iteration_num
         );

#endif
