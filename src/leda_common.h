#ifndef LEDA_COMMON_H
#define LEDA_COMMON_H

#include <vector>
#include <iostream>
#include <bitset>
#include <omp.h>
#include "mmio_highlevel.h"
#include "leda_common.h"

using std::cout;
using std::endl;
using std::vector;
using std::min;
using std::max;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T> >;

struct SpElement{
    INDEX_TYPE colIdx;
    INDEX_TYPE rowIdx;
    VALUE_TYPE val;
    
    SpElement(INDEX_TYPE colidx = -1, INDEX_TYPE rowidx = -1, VALUE_TYPE value = 0.0): colIdx(colidx), rowIdx(rowidx), val(value) {}
    
    SpElement& operator=(const SpElement& sp) {
        colIdx = sp.colIdx;
        rowIdx = sp.rowIdx;
        val    = sp.val;
        return *this;
    }
};

struct Matrix_COO {
    INDEX_TYPE         M;
    INDEX_TYPE         K;
    INDEX_TYPE         nnzR;

    vector<INDEX_TYPE> ColIdx;
    vector<INDEX_TYPE> RowIdx;
    vector<INDEX_TYPE> RowIdx_copy;
    vector<VALUE_TYPE> Val;

    vector<unsigned short> mask;
    vector<vector<INDEX_TYPE> > map;

    Matrix_COO() : M(0), K(0), nnzR(0), ColIdx() , RowIdx(), Val(), mask(), map() {}
};

struct SparseTile {
    INDEX_TYPE         TileSize;
    INDEX_TYPE         numColTiles;
    INDEX_TYPE         numRowTiles;
    INDEX_TYPE         numTiles;

    vector<INDEX_TYPE> TileColPtr;
    vector<INDEX_TYPE> TileRowIdx;
    vector<Matrix_COO> TileVal;

    SparseTile() : TileSize(0), numColTiles(0), numRowTiles(0), TileColPtr(), TileRowIdx(), TileVal() {}
};

void Read_matrix_size(char       *filename,
                      INDEX_TYPE *M, 
                      INDEX_TYPE *K, 
                      INDEX_TYPE *nnzR,
                      INDEX_TYPE *isSymmetric
                     ) {

    mmio_info(M, K, nnzR, isSymmetric, filename);
}

void Read_matrix_2_CSR(char       *filename, 
                       const INDEX_TYPE M, 
                       const INDEX_TYPE K, 
                       const INDEX_TYPE nnzR,

                       vector<INDEX_TYPE> &RowPtr, 
                       vector<INDEX_TYPE> &ColIdx, 
                       vector<VALUE_TYPE> &Val
                      ) {

    INDEX_TYPE *RowPtr_d = (INDEX_TYPE *)malloc(sizeof(INDEX_TYPE) * (M + 1));
    INDEX_TYPE *ColIdx_d = (INDEX_TYPE *)malloc(sizeof(INDEX_TYPE) * nnzR);
    VALUE_TYPE *Val_d    = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * nnzR);

    mmio_data_csr(RowPtr_d, ColIdx_d, Val_d, filename);

    for(INDEX_TYPE i = 0; i < M + 1; ++i)
        RowPtr[i] = RowPtr_d[i];
    
    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        ColIdx[i] = ColIdx_d[i];
        Val[i]    = Val_d[i];
    }

    free(Val_d);
    free(ColIdx_d);
    free(RowPtr_d);
}

void Read_matrix_2_CSC(char       *filename, 
                       const INDEX_TYPE M, 
                       const INDEX_TYPE K, 
                       const INDEX_TYPE nnzR,

                       vector<INDEX_TYPE> &ColPtr, 
                       vector<INDEX_TYPE> &RowIdx, 
                       vector<VALUE_TYPE> &Val
                      ) {

    INDEX_TYPE *ColPtr_d = (INDEX_TYPE *)malloc(sizeof(INDEX_TYPE) * (K + 1));
    INDEX_TYPE *RowIdx_d = (INDEX_TYPE *)malloc(sizeof(INDEX_TYPE) * nnzR);
    VALUE_TYPE *Val_d    = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * nnzR);

    mmio_data_csc(ColPtr_d, RowIdx_d, Val_d, filename);

    for(INDEX_TYPE i = 0; i < K + 1; ++i)
        ColPtr[i] = ColPtr_d[i];
    
    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        RowIdx[i] = RowIdx_d[i];
        Val[i]    = Val_d[i];
    }

    free(Val_d);
    free(RowIdx_d);
    free(ColPtr_d);
}

void CSC_2_CSR(const INDEX_TYPE M,
               const INDEX_TYPE K,
               const INDEX_TYPE nnzR,

               const vector<INDEX_TYPE> &ColPtr_CSC,
               const vector<INDEX_TYPE> &RowIdx_CSC,
               const vector<VALUE_TYPE> &Val_CSC,
               
               vector<INDEX_TYPE> &RowPtr_CSR,
               vector<INDEX_TYPE> &ColIdx_CSR,
               vector<VALUE_TYPE> &Val_CSR
              ) {

    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        RowPtr_CSR[RowIdx_CSC[i] + 1]++;
    }
    
    for(INDEX_TYPE i = 0; i < M; ++i) {
        RowPtr_CSR[i + 1] += RowPtr_CSR[i];
    }
    
    vector<INDEX_TYPE> row_nnzR(M, 0);
    for(INDEX_TYPE i = 0; i < K; ++i) {
        for(INDEX_TYPE j = ColPtr_CSC[i]; j < ColPtr_CSC[i + 1]; ++j) {
            INDEX_TYPE row = RowIdx_CSC[j];
            INDEX_TYPE col = i;
            VALUE_TYPE val = Val_CSC[j];
            
            INDEX_TYPE pos = RowPtr_CSR[row] + row_nnzR[row];
            Val_CSR[pos] = val;
            ColIdx_CSR[pos] = col;
            row_nnzR[row]++;
        }
    }
}

void CSR_2_CSC(const INDEX_TYPE M, 
               const INDEX_TYPE K, 
               const INDEX_TYPE nnzR,

               const vector<INDEX_TYPE> &RowPtr_CSR, 
               const vector<INDEX_TYPE> &ColIdx_CSR, 
               const vector<VALUE_TYPE> &Val_CSR,

               vector<INDEX_TYPE> &ColPtr_CSC,
               vector<INDEX_TYPE> &RowIdx_CSC,
               vector<VALUE_TYPE> &Val_CSC
              ) {

    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        ColPtr_CSC[ColIdx_CSR[i] + 1]++;
    }

    for(INDEX_TYPE i = 0; i < K; ++i) {
        ColPtr_CSC[i + 1] += ColPtr_CSC[i];
    }

    vector<INDEX_TYPE> col_nnzR(K, 0);
    for(INDEX_TYPE i = 0; i < M; ++i) {
        for(INDEX_TYPE j = RowPtr_CSR[i]; j < RowPtr_CSR[i + 1]; ++j) {
            INDEX_TYPE row = i;
            INDEX_TYPE col = ColIdx_CSR[j];
            VALUE_TYPE val = Val_CSR[j];
            
            INDEX_TYPE pos = ColPtr_CSC[col] + col_nnzR[col];
            Val_CSC[pos] = val;
            RowIdx_CSC[pos] = row;
            col_nnzR[col]++;
        }
    }
}

void CSR_2_COO(const INDEX_TYPE M, 
               const INDEX_TYPE K, 
               const INDEX_TYPE nnzR,

               const vector<INDEX_TYPE> &RowPtr_CSR, 
               const vector<INDEX_TYPE> &ColIdx_CSR, 
               const vector<VALUE_TYPE> &Val_CSR,

               vector<INDEX_TYPE> &RowIdx_COO,
               vector<INDEX_TYPE> &ColIdx_COO,
               vector<VALUE_TYPE> &Val_COO
             ) {

    INDEX_TYPE row = 0;
    for(INDEX_TYPE i = 0; i < M; ++i) {
        for(INDEX_TYPE j = RowPtr_CSR[i]; j < RowPtr_CSR[i + 1]; ++j) {
            RowIdx_COO[j] = row;
            ColIdx_COO[j] = ColIdx_CSR[j];
            Val_COO[j]    = Val_CSR[j];
        }

        row++;
    }
}

void CSC_2_COO(const INDEX_TYPE M, 
               const INDEX_TYPE K, 
               const INDEX_TYPE nnzR,

               const vector<INDEX_TYPE> &ColPtr_CSC, 
               const vector<INDEX_TYPE> &RowIdx_CSC, 
               const vector<VALUE_TYPE> &Val_CSC,

               vector<INDEX_TYPE> &RowIdx_COO,
               vector<INDEX_TYPE> &ColIdx_COO,
               vector<VALUE_TYPE> &Val_COO
             ) {

    INDEX_TYPE col = 0;
    for(INDEX_TYPE i = 0; i < K; ++i) {
        for(INDEX_TYPE j = ColPtr_CSC[i]; j < ColPtr_CSC[i + 1]; ++j) {
            RowIdx_COO[j] = RowIdx_CSC[j];
            ColIdx_COO[j] = col;
            Val_COO[j]    = Val_CSC[j];
        }
        col++;
    }
}

void Generate_Dense_Matrix(const INDEX_TYPE M, 
                           const INDEX_TYPE K,
                           const VALUE_TYPE Val,
                           vector<VALUE_TYPE> &Matrix_Dense,
                           bool val_n,
                           bool is_row_major = true
                          ) {
    if(is_row_major) {
        for(INDEX_TYPE mm = 0; mm < M; ++mm) {
            for(INDEX_TYPE kk = 0; kk < K; ++kk) {
                if(val_n) {
                    Matrix_Dense[mm * K + kk] = mm;
                }
                else {
                    Matrix_Dense[mm * K + kk] = Val;
                }
            }
        }
    }
    else {
        for(INDEX_TYPE kk = 0; kk < K; ++kk) {
            for(INDEX_TYPE mm = 0; mm < M; ++mm) {
                if(val_n) {
                    Matrix_Dense[mm + M * kk] = kk;
                }
                else {
                    Matrix_Dense[mm + M * kk] = (1.0 + kk) + 0.1 * (1.0 + mm);
                }
            }
        }
    }
}

INDEX_TYPE CountOnes(const unsigned short num) {
    INDEX_TYPE count = 0;
    unsigned short num_tmp = num;
    while (num_tmp) {
        count += num_tmp & 1;
        num_tmp >>= 1;
    }
    return count;
}

void SpMM_CPU_CSR(const INDEX_TYPE M,
                  const INDEX_TYPE N,
                  const INDEX_TYPE K,
                  const INDEX_TYPE nnzR,
                  const vector<INDEX_TYPE> &RowPtr_CSR,
                  const vector<INDEX_TYPE> &ColIdx_CSR,
                  const vector<VALUE_TYPE> &Val_CSR,
                  const vector<VALUE_TYPE> &Matrix_B_Dense,
                  vector<VALUE_TYPE>       &Matrix_C_Dense
                 ) {
  for(INDEX_TYPE i = 0; i < M; ++i) {
    for(INDEX_TYPE j = RowPtr_CSR[i]; j < RowPtr_CSR[i+1]; ++j) {
      for(INDEX_TYPE l = 0; l < N; ++l) {
        Matrix_C_Dense[l * M + i] += Val_CSR[j] * Matrix_B_Dense[l * K + ColIdx_CSR[j]];
      }
    }
  }
}

void SpMM_CPU_CSC(const INDEX_TYPE M,
                  const INDEX_TYPE N,
                  const INDEX_TYPE K,
                  const INDEX_TYPE nnzR,
                  const vector<INDEX_TYPE> &ColPtr_CSC,
                  const vector<INDEX_TYPE> &RowIdx_CSC,
                  const vector<VALUE_TYPE> &Val_CSC,
                  const vector<VALUE_TYPE> &Matrix_B_Dense,
                  vector<VALUE_TYPE>       &Matrix_C_Dense
                 ) {
  for(INDEX_TYPE i = 0; i < K; ++i) {
    for(INDEX_TYPE j = ColPtr_CSC[i]; j < ColPtr_CSC[i+1]; ++j) {
      for(INDEX_TYPE l = 0; l < N; ++l) {
        Matrix_C_Dense[l * M + RowIdx_CSC[j]] += Val_CSC[j] * Matrix_B_Dense[l * K + i];
      }
    }
  }
}

void SpMM_CPU_Tile(const INDEX_TYPE M, 
                    const INDEX_TYPE N, 
                    const INDEX_TYPE K,
                    const vector<SparseTile> &Matrix_SparseTile,
                    const vector<VALUE_TYPE>  &Matrix_B_Dense,
                    vector<VALUE_TYPE>        &Matrix_C_Dense
                   ) {

    for(INDEX_TYPE p = 0; p < Matrix_SparseTile.size(); p++) {
        for(INDEX_TYPE j = 0; j < Matrix_SparseTile[p].numColTiles; ++j) {
                for(INDEX_TYPE i = Matrix_SparseTile[p].TileColPtr[j]; i < Matrix_SparseTile[p].TileColPtr[j + 1]; ++i) {
                    INDEX_TYPE TilennzR = Matrix_SparseTile[p].TileVal[i].nnzR;

                    for(INDEX_TYPE k = 0; k < TilennzR; ++k) {

                        INDEX_TYPE r = Matrix_SparseTile[p].TileVal[i].RowIdx_copy[k];
                        INDEX_TYPE c = Matrix_SparseTile[p].TileVal[i].ColIdx[k];
                        VALUE_TYPE v = Matrix_SparseTile[p].TileVal[i].Val[k];
                        
                        for(INDEX_TYPE l = 0; l < N; ++l) {
                                Matrix_C_Dense[l * M + r] += v * Matrix_B_Dense[l * K + c];
                            }
                    }

                }
            }
    }
}

void Matrix_Scatter(const INDEX_TYPE M, 
                    const INDEX_TYPE K, 
                    const INDEX_TYPE nnzR,
                    
                    const vector<INDEX_TYPE> &RowIdx_COO,
                    const vector<INDEX_TYPE> &ColIdx_COO,
                    const vector<VALUE_TYPE> &Val_COO,

                    const INDEX_TYPE NUM_PE,

                    vector<Matrix_COO> &Matrix_Band_COO
                    ) {
                    
    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        INDEX_TYPE p = (RowIdx_COO[i]) % NUM_PE;
        INDEX_TYPE pos = Matrix_Band_COO[p].RowIdx.size();
        Matrix_Band_COO[p].RowIdx.resize(pos + 1);
        Matrix_Band_COO[p].RowIdx_copy.resize(pos + 1);
        Matrix_Band_COO[p].ColIdx.resize(pos + 1);
        Matrix_Band_COO[p].Val.resize(pos + 1);
        Matrix_Band_COO[p].RowIdx[pos] = RowIdx_COO[i] / NUM_PE;
        Matrix_Band_COO[p].RowIdx_copy[pos] = RowIdx_COO[i];
        Matrix_Band_COO[p].ColIdx[pos] = ColIdx_COO[i];
        Matrix_Band_COO[p].Val[pos] = Val_COO[i];
        Matrix_Band_COO[p].nnzR++;  
    }


#pragma omp parallel for
    for(INDEX_TYPE i = 0; i < Matrix_Band_COO.size(); ++i) {
        INDEX_TYPE max_rownum = -1;
        INDEX_TYPE max_colnum = -1;
        for(INDEX_TYPE j = 0; j < Matrix_Band_COO[i].nnzR; ++j) {
            max_rownum = max(max_rownum, Matrix_Band_COO[i].RowIdx[j]);
            max_colnum = max(max_colnum, Matrix_Band_COO[i].ColIdx[j]);
        }
        Matrix_Band_COO[i].M = max_rownum + 1;
        Matrix_Band_COO[i].K = max_colnum + 1;
    }
}

void Create_SparseTile(const INDEX_TYPE M, 
                        const INDEX_TYPE K, 
                        const INDEX_TYPE nnzR,

                        const INDEX_TYPE TileSize,

                        const vector<INDEX_TYPE> &RowIdx_COO,
                        const vector<INDEX_TYPE> &ColIdx_COO,
                        const vector<VALUE_TYPE> &Val_COO,

                        SparseTile &TileMatrix
                        ) {
    
    INDEX_TYPE numColTiles = (K + TileSize - 1) / TileSize;
    INDEX_TYPE numRowTiles = (M + TileSize - 1) / TileSize;

    INDEX_TYPE newnumCols  = numColTiles * TileSize;
    INDEX_TYPE newnumRows  = numRowTiles * TileSize;
    INDEX_TYPE newnnzR     = nnzR;

    if(newnumCols != K || newnumRows != M) {
        newnnzR += (newnumCols - K);
    }

    SparseTile TileMatrix_temp;

    TileMatrix_temp.numColTiles = numColTiles;
    TileMatrix_temp.numRowTiles = numRowTiles;
    TileMatrix_temp.TileSize    = TileSize;

    INDEX_TYPE numTiles         = numColTiles * numRowTiles;

    TileMatrix_temp.numTiles    = numTiles;

    vector<INDEX_TYPE> TileCounts(numTiles, 0);
    for (INDEX_TYPE i = 0; i < nnzR; ++i) {
        INDEX_TYPE row       = RowIdx_COO[i];
        INDEX_TYPE col       = ColIdx_COO[i];
        INDEX_TYPE TileRow   = row / TileSize;
        INDEX_TYPE TileCol   = col / TileSize;
        INDEX_TYPE TileIndex = TileCol * numRowTiles + TileRow;
        TileCounts[TileIndex]++;
    }

    INDEX_TYPE numTiles_nnzR = 0;
    for(INDEX_TYPE i = 0; i < numTiles; ++i) {
        if(TileCounts[i] != 0) numTiles_nnzR++;
    }

    TileMatrix_temp.TileColPtr.resize(numColTiles + 1, 0);
    TileMatrix_temp.TileRowIdx.resize(numTiles_nnzR, 0);

    for(INDEX_TYPE j = 0; j < numColTiles; ++j) {
        for(INDEX_TYPE i = 0; i < numRowTiles; ++i) {
            INDEX_TYPE TileIndex = j * numRowTiles + i;
            if(TileCounts[TileIndex] != 0) {
                TileMatrix_temp.TileColPtr[j + 1] += 1;
                Matrix_COO cooElem_temp;
                cooElem_temp.M    = TileSize;
                cooElem_temp.K    = TileSize;
                cooElem_temp.nnzR = TileCounts[TileIndex];

                TileMatrix_temp.TileVal.push_back(cooElem_temp);
            } 
        }
    }

    for(INDEX_TYPE j = 0; j < numColTiles; ++j) {
        TileMatrix_temp.TileColPtr[j + 1] += TileMatrix_temp.TileColPtr[j];
    }
    
    vector<INDEX_TYPE> TileOffsets(numTiles, 0);
    INDEX_TYPE offset = 0;
    for(INDEX_TYPE i = 0; i < numTiles; ++i) {
        if(TileCounts[i] != 0) {
            TileOffsets[i] = offset;
            offset++;
        }
    }

    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        INDEX_TYPE row        = RowIdx_COO[i];
        INDEX_TYPE col        = ColIdx_COO[i];
        VALUE_TYPE value      = Val_COO[i];
        INDEX_TYPE TileRow    = row / TileSize;
        INDEX_TYPE TileCol    = col / TileSize;
        INDEX_TYPE TileIndex  = TileCol * numRowTiles + TileRow;
        INDEX_TYPE TileOffset = TileOffsets[TileIndex];

        TileMatrix_temp.TileRowIdx[TileOffset] = TileRow;
        TileMatrix_temp.TileVal[TileOffset].ColIdx.push_back(col);
        TileMatrix_temp.TileVal[TileOffset].RowIdx.push_back(row);
        TileMatrix_temp.TileVal[TileOffset].Val.push_back(value);
        
    }

    TileMatrix_temp.numTiles  = TileMatrix_temp.TileColPtr[numColTiles];
    TileMatrix = TileMatrix_temp;
}

void Create_Matrix_Band_SparseTile(const INDEX_TYPE TileSize,
                                    const Matrix_COO &Matrix_Band_COO,
                                    SparseTile      &Matrix_Band_Tile
                                   ) {
    INDEX_TYPE M = Matrix_Band_COO.M; 
    INDEX_TYPE K = Matrix_Band_COO.K;
    INDEX_TYPE nnzR = Matrix_Band_COO.nnzR;

    vector<INDEX_TYPE> RowIdx_COO = Matrix_Band_COO.RowIdx;
    vector<INDEX_TYPE> RowIdx_COO_copy = Matrix_Band_COO.RowIdx_copy;
    vector<INDEX_TYPE> ColIdx_COO = Matrix_Band_COO.ColIdx;
    vector<VALUE_TYPE> Val_COO = Matrix_Band_COO.Val;

    INDEX_TYPE numColTiles = (K + TileSize - 1) / TileSize;
    INDEX_TYPE numRowTiles = (M + TileSize - 1) / TileSize;

    INDEX_TYPE newnumCols  = numColTiles * TileSize;
    INDEX_TYPE newnumRows  = numRowTiles * TileSize;
    INDEX_TYPE newnnzR     = nnzR;

    if(newnumCols != K || newnumRows != M) {
        newnnzR += (newnumCols - K);
    }

    SparseTile Matrix_Band_Tile_temp;

    Matrix_Band_Tile_temp.numColTiles = numColTiles;
    Matrix_Band_Tile_temp.numRowTiles = numRowTiles;
    Matrix_Band_Tile_temp.TileSize    = TileSize;

    INDEX_TYPE numTiles         = numColTiles * numRowTiles;

    Matrix_Band_Tile_temp.numTiles    = numTiles;
    
    vector<INDEX_TYPE> TileCounts(numTiles, 0);

    for (INDEX_TYPE i = 0; i < nnzR; ++i) {
        INDEX_TYPE row       = RowIdx_COO[i];
        INDEX_TYPE col       = ColIdx_COO[i];
        INDEX_TYPE TileRow   = row / TileSize;
        INDEX_TYPE TileCol   = col / TileSize;
        INDEX_TYPE TileIndex = TileCol * numRowTiles + TileRow;
        TileCounts[TileIndex]++;
    }

    INDEX_TYPE numTiles_nnzR = 0;
    for(INDEX_TYPE i = 0; i < numTiles; ++i) {
        if(TileCounts[i] != 0) numTiles_nnzR++;
    }

    Matrix_Band_Tile_temp.TileColPtr.resize(numColTiles + 1, 0);
    Matrix_Band_Tile_temp.TileRowIdx.resize(numTiles_nnzR, 0);

    for(INDEX_TYPE j = 0; j < numColTiles; ++j) {
        for(INDEX_TYPE i = 0; i < numRowTiles; ++i) {
            INDEX_TYPE TileIndex = j * numRowTiles + i;
            if(TileCounts[TileIndex] != 0) {
                Matrix_Band_Tile_temp.TileColPtr[j + 1] += 1;

                Matrix_COO cooElem_temp;
                cooElem_temp.M    = TileSize;
                cooElem_temp.K    = TileSize;
                cooElem_temp.nnzR = TileCounts[TileIndex];
                cooElem_temp.mask.resize(TileSize, 0);
                cooElem_temp.map.resize(TileSize);

                Matrix_Band_Tile_temp.TileVal.push_back(cooElem_temp);
            } 
        }
    }

    for(INDEX_TYPE j = 0; j < numColTiles; ++j) {
        Matrix_Band_Tile_temp.TileColPtr[j + 1] += Matrix_Band_Tile_temp.TileColPtr[j];
    }
    
    vector<INDEX_TYPE> TileOffsets(numTiles, 0);
    INDEX_TYPE offset = 0;
    for(INDEX_TYPE i = 0; i < numTiles; ++i) {
        if(TileCounts[i] != 0) {
            TileOffsets[i] = offset;
            offset++;
        }
    }

    vector<INDEX_TYPE> Tile_push_num(numTiles_nnzR, 0);

    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        INDEX_TYPE row         = RowIdx_COO[i];
        INDEX_TYPE row_copy    = RowIdx_COO_copy[i];
        INDEX_TYPE col         = ColIdx_COO[i];
        VALUE_TYPE value       = Val_COO[i];
        INDEX_TYPE TileRow    = row / TileSize;
        INDEX_TYPE TileCol    = col / TileSize;
        INDEX_TYPE TileIndex  = TileCol * numRowTiles + TileRow;
        INDEX_TYPE TileOffset = TileOffsets[TileIndex];

        Matrix_Band_Tile_temp.TileRowIdx[TileOffset] = TileRow;
        Matrix_Band_Tile_temp.TileVal[TileOffset].ColIdx.push_back(col);
        Matrix_Band_Tile_temp.TileVal[TileOffset].RowIdx.push_back(row);
        
        Matrix_Band_Tile_temp.TileVal[TileOffset].RowIdx_copy.push_back(row_copy);
        Matrix_Band_Tile_temp.TileVal[TileOffset].Val.push_back(value);

        Matrix_Band_Tile_temp.TileVal[TileOffset].mask[col - TileCol * Tile_SIZE] |=  (0x1 << (row - TileRow * Tile_SIZE));
        Matrix_Band_Tile_temp.TileVal[TileOffset].map[col - TileCol * Tile_SIZE].push_back(Tile_push_num[TileOffset]);
        Tile_push_num[TileOffset]++;
    }

    Matrix_Band_Tile_temp.numTiles  = Matrix_Band_Tile_temp.TileColPtr[numColTiles];
    Matrix_Band_Tile = Matrix_Band_Tile_temp;
}


void Create_Matrix_Band_SparseTile_ex(const vector<Matrix_COO> &Matrix_Band_COO,
                                       vector<SparseTile> &Matrix_Band_Tile) {
#pragma omp parallel for
    for(INDEX_TYPE i = 0; i < Matrix_Band_Tile.size(); ++i) {
        Create_Matrix_Band_SparseTile(Tile_SIZE, Matrix_Band_COO[i], Matrix_Band_Tile[i]);
    }

}

void Tile_MiniSimilar_Column_reorder(Matrix_COO &TileVal) {

    vector<INDEX_TYPE> RowIdx_tmp;
    vector<INDEX_TYPE> RowIdx_copy_tmp;
    vector<INDEX_TYPE> ColIdx_tmp;
    vector<VALUE_TYPE> Val_tmp;
    
    vector<unsigned short> mask_tmp = TileVal.mask;

    vector<INDEX_TYPE> list;

    INDEX_TYPE mask_num = 0;

    for(INDEX_TYPE maskcol = 0; maskcol < Tile_SIZE; ++maskcol) {
        if(CountOnes(mask_tmp[maskcol]) != 0) {
            mask_num++;
        }
    }

    for(INDEX_TYPE maskcol = 0; maskcol < Tile_SIZE; ++maskcol) {
        if(CountOnes(mask_tmp[maskcol]) != 0) {
            list.push_back(maskcol);
            break;
        }
    }

    INDEX_TYPE num = 0;
    while(num < mask_num - 1) {
        INDEX_TYPE pos = list.size();
        INDEX_TYPE this_col = list[pos - 1];
        unsigned short this_mask = mask_tmp[this_col];

        INDEX_TYPE min_val = 10000;
        INDEX_TYPE min_colidx;

        for(INDEX_TYPE maskcol_next = 0; maskcol_next < Tile_SIZE; ++maskcol_next) {
            if(this_col != maskcol_next) {
                if(CountOnes(mask_tmp[maskcol_next]) != 0) {
                    INDEX_TYPE countone = CountOnes(this_mask & mask_tmp[maskcol_next]);
                    if(countone < min_val) {
                        min_val = countone;
                        min_colidx = maskcol_next;
                    }
                }
            }
        }

        list.resize(pos + 1);
        list[pos] = min_colidx;
        mask_tmp[this_col] &= 0;
        num++;
    }

    for(INDEX_TYPE i = 0; i < list.size(); ++i) {
        for(INDEX_TYPE j = 0; j < TileVal.map[list[i]].size(); ++j) {
            RowIdx_tmp.push_back(TileVal.RowIdx[TileVal.map[list[i]][j]]);
            RowIdx_copy_tmp.push_back(TileVal.RowIdx_copy[TileVal.map[list[i]][j]]);
            ColIdx_tmp.push_back(TileVal.ColIdx[TileVal.map[list[i]][j]]);
            Val_tmp.push_back(TileVal.Val[TileVal.map[list[i]][j]]);
        }
    }

    TileVal.RowIdx = RowIdx_tmp;
    TileVal.RowIdx_copy = RowIdx_copy_tmp;
    TileVal.ColIdx = ColIdx_tmp;
    TileVal.Val = Val_tmp;
}


void Get_tile_nnzr(const SparseTile &Matrix_SparseTile, vector<INDEX_TYPE> &tile_nnzr, INDEX_TYPE &tile_num) {
    for(INDEX_TYPE j = 0; j < Matrix_SparseTile.numColTiles; ++j) {
        for(INDEX_TYPE i = Matrix_SparseTile.TileColPtr[j]; i < Matrix_SparseTile.TileColPtr[j + 1]; ++i) {
            INDEX_TYPE nnzr = Matrix_SparseTile.TileVal[i].nnzR;
            if(nnzr != 0) {
                INDEX_TYPE pos = tile_nnzr.size();
                tile_nnzr.resize(pos + 1);
                tile_nnzr[pos] = nnzr;
                tile_num++;
            }
        }
    }
}

void Reordering(const vector<SpElement> &temp_SpElement_list,
                vector<SpElement> &SpEelment_list,
                const INDEX_TYPE base_col_index,
                const INDEX_TYPE i_start,
                const INDEX_TYPE NUM_Row,
                const INDEX_TYPE NUM_PE,
                const INDEX_TYPE WIDTH
                ) {

    SpElement sp_empty = {-1, -1, (VALUE_TYPE)0};

    vector<SpElement> scheduled_SpElement;
    
    vector<INDEX_TYPE> sliding_window(NUM_Row, -WIDTH);
    INDEX_TYPE org_row_idx;

    for(INDEX_TYPE p = 0; p < temp_SpElement_list.size(); ++p) {
        org_row_idx = temp_SpElement_list[p].rowIdx;
        INDEX_TYPE win_row_idx = sliding_window[org_row_idx] + WIDTH;
        INDEX_TYPE insert_flag = 1;
        while(insert_flag){
            if(win_row_idx >= ((INDEX_TYPE)scheduled_SpElement.size())) {
                scheduled_SpElement.resize(win_row_idx + 1);
                scheduled_SpElement[win_row_idx] = sp_empty;
            }
            SpElement sp = scheduled_SpElement[win_row_idx];
            if(sp.rowIdx == -1 && sp.colIdx == -1 && sp.val == 0.0) {
                insert_flag = 0;
            }
            else {
                win_row_idx++;
            }
        }

        scheduled_SpElement[win_row_idx].colIdx = temp_SpElement_list[p].colIdx - base_col_index;
        scheduled_SpElement[win_row_idx].rowIdx = org_row_idx;
        scheduled_SpElement[win_row_idx].val = temp_SpElement_list[p].val;
        sliding_window[org_row_idx] = win_row_idx;
    }

    INDEX_TYPE scheduled_SpElement_size = scheduled_SpElement.size();

    if(scheduled_SpElement_size > 0) {
        SpEelment_list.resize(i_start + scheduled_SpElement_size, sp_empty);
        for(INDEX_TYPE i = 0; i < scheduled_SpElement_size; ++i) {
            SpEelment_list[i + i_start] = scheduled_SpElement[i];
        }
    }
}

void Push_SpEelment_list(const vector<SpElement> &temp_SpElement_list,
                         vector<SpElement> &SpEelment_list,
                         const INDEX_TYPE base_col_index,
                         const INDEX_TYPE i_start
                        ) {

    SpElement sp_empty = {-1, -1, (VALUE_TYPE)0};

    vector<SpElement> scheduled_SpElement;

    for(INDEX_TYPE p = 0; p < temp_SpElement_list.size(); ++p) {
        INDEX_TYPE pos = scheduled_SpElement.size();
        scheduled_SpElement.resize(pos + 1);
        scheduled_SpElement[pos].rowIdx = temp_SpElement_list[p].rowIdx;
        scheduled_SpElement[pos].colIdx = temp_SpElement_list[p].colIdx - base_col_index;
        scheduled_SpElement[pos].val = temp_SpElement_list[p].val;
    }

    INDEX_TYPE scheduled_SpElement_size = scheduled_SpElement.size();

    if(scheduled_SpElement_size > 0) {
        SpEelment_list.resize(i_start + scheduled_SpElement_size, sp_empty);
        for(INDEX_TYPE i = 0; i < scheduled_SpElement_size; ++i) {
            SpEelment_list[i + i_start] = scheduled_SpElement[i];
        }
    }
}

void Create_SpElement_list_for_all_PEs(const INDEX_TYPE NUM_PE,
                                       const INDEX_TYPE NUM_ROW,
                                       const INDEX_TYPE NUM_COLUMN,
                                       const INDEX_TYPE Tile_SIZE,
                                       const INDEX_TYPE BATCH_SIZE,

                                       vector<SparseTile> &Matrix_Band_Tile,
                                       vector<vector<SpElement> > &SpElement_list_pes,
                                       vector<INDEX_TYPE> &SpElement_list_ptr,
                                       const INDEX_TYPE WINDOWS = 10
                                      ) {
    SpElement_list_pes.resize(NUM_PE); 

    INDEX_TYPE numColTiles_max = -1;                                  
    for(INDEX_TYPE p = 0; p < NUM_PE; ++p) {
        numColTiles_max = max(Matrix_Band_Tile[p].numColTiles, numColTiles_max);
    }

    SpElement_list_ptr.resize((numColTiles_max + BATCH_SIZE - 1) / BATCH_SIZE + 1, 0);

    vector<vector<SpElement> > temp_SpElement_list_pes(NUM_PE);
    for(INDEX_TYPE i = 0; i < (numColTiles_max + BATCH_SIZE - 1) / BATCH_SIZE; ++i) {

#pragma omp parallel for     
        for(INDEX_TYPE p = 0; p < NUM_PE; p++) {
            temp_SpElement_list_pes[p].resize(0);
            for(INDEX_TYPE Tilecolidx =  BATCH_SIZE * i; Tilecolidx < min(BATCH_SIZE * (i + 1), Matrix_Band_Tile[p].numColTiles); ++Tilecolidx) {
                for(INDEX_TYPE j = Matrix_Band_Tile[p].TileColPtr[Tilecolidx]; j < Matrix_Band_Tile[p].TileColPtr[Tilecolidx + 1]; ++j) {
                    INDEX_TYPE TilennzR = Matrix_Band_Tile[p].TileVal[j].nnzR;
                    Tile_MiniSimilar_Column_reorder(Matrix_Band_Tile[p].TileVal[j]);

                    for(INDEX_TYPE k = 0; k < TilennzR; ++k) {
                        INDEX_TYPE pos = temp_SpElement_list_pes[p].size();
                        temp_SpElement_list_pes[p].resize(pos + 1);
                        temp_SpElement_list_pes[p][pos] = SpElement(Matrix_Band_Tile[p].TileVal[j].ColIdx[k], Matrix_Band_Tile[p].TileVal[j].RowIdx[k], Matrix_Band_Tile[p].TileVal[j].Val[k]);
                    }
                }
            } 

            INDEX_TYPE i_start = SpElement_list_pes[p].size();
            INDEX_TYPE base_col_index = i * BATCH_SIZE * Tile_SIZE;

            Reordering(temp_SpElement_list_pes[p],
                       SpElement_list_pes[p],
                       base_col_index,
                       i_start,
                       NUM_ROW,
                       NUM_PE,
                       WINDOWS
                      );
        }

        INDEX_TYPE max_len = 0;
        for(INDEX_TYPE p = 0; p < NUM_PE; ++p) {
            max_len = max((INDEX_TYPE) SpElement_list_pes[p].size(), max_len);
        }
        
        for(INDEX_TYPE p = 0; p < NUM_PE; ++p) {
            SpElement_list_pes[p].resize(max_len, SpElement(-1, -1, 0.0));
        }
        
        SpElement_list_ptr[i + 1] = max_len;
    } 
}


void Create_SpElement_list_for_all_channels(const vector<vector<SpElement> > &SpElement_list_pes,
                                            const vector<INDEX_TYPE>         &SpElement_list_ptr,
                                            vector<vector<unsigned long, tapa::aligned_allocator<unsigned long> > > &Matrix_A_fpga_data,
                                            const INDEX_TYPE HBM_CHANNEL_A_NUM = 8
                                           ) {
    INDEX_TYPE Matrix_fpga_data_column_size = 8 * SpElement_list_ptr[SpElement_list_ptr.size() - 1] * 4 / 4;
    INDEX_TYPE Matrix_fpga_data_channel_size  = ((Matrix_fpga_data_column_size + 512 - 1) / 512) * 512;

    for(INDEX_TYPE c = 0; c < HBM_CHANNEL_A_NUM; ++c) {
        Matrix_A_fpga_data[c].resize(Matrix_fpga_data_channel_size, 0);
    }
    
    for(INDEX_TYPE i = 0; i < SpElement_list_ptr[SpElement_list_ptr.size() - 1]; ++i) {
        for(INDEX_TYPE c = 0; c < HBM_CHANNEL_A_NUM; ++c) {
            for(INDEX_TYPE j = 0; j < 8; ++j) {
                SpElement sp = SpElement_list_pes[j + c * 8][i];

                unsigned long x = 0;
                if(sp.rowIdx == -1) {
                    x = 0x3FFFF;
                    x = x << 32;
                } 
                else {
                    unsigned long x_col = sp.colIdx;
                    x_col = (x_col & 0x3FFF) << (32 + 18); 
                    unsigned long x_row = sp.rowIdx;
                    x_row = (x_row & 0x3FFFF) << 32;
                    VALUE_TYPE x_float = sp.val;
                    
                    unsigned int x_float_in_int = *((unsigned int*)(&x_float));
                    unsigned long x_float_val_64 = ((unsigned long) x_float_in_int);
                    x_float_val_64 = x_float_val_64 & 0xFFFFFFFF;

                    x = x_col | x_row | x_float_val_64;
                }
                if(HBM_CHANNEL_A_NUM * 8 <= 16) {
                    Matrix_A_fpga_data[c][j + i * 8] = x;
                } 
                else if(HBM_CHANNEL_A_NUM == 8) {


                    INDEX_TYPE seg = 16 / HBM_CHANNEL_A_NUM;
                    INDEX_TYPE seg_idx = (c * 8 + j) / seg;
                    INDEX_TYPE c_new = seg_idx % HBM_CHANNEL_A_NUM;
                    INDEX_TYPE j_new = seg * (seg_idx / HBM_CHANNEL_A_NUM) + j % seg;
                    Matrix_A_fpga_data[c_new][j_new + i * 8] = x;
                } 
                else if(HBM_CHANNEL_A_NUM == 4) {
                    INDEX_TYPE pe_idx = j + c * 8;
                    Matrix_A_fpga_data[(pe_idx / 4) % 4][pe_idx % 4 + (pe_idx / 16) * 4 + i * 8] = x;
                }
            }
        }
    }
}

void Create_SpElement_list_data_FPGA(const vector<INDEX_TYPE> &SpElement_list_ptr,
                                     aligned_vector<INDEX_TYPE> &SpElement_list_ptr_fpga
                                    ) {
    INDEX_TYPE SpElement_list_ptr_fpga_size = ((SpElement_list_ptr.size() + 15) / 16) * 16;
    INDEX_TYPE SpElement_list_ptr_fpga_chunk_size = ((SpElement_list_ptr_fpga_size + 1023) / 1024) * 1024;
    SpElement_list_ptr_fpga.resize(SpElement_list_ptr_fpga_chunk_size, 0);
    for(INDEX_TYPE i = 0; i < SpElement_list_ptr.size(); ++i) {
        SpElement_list_ptr_fpga[i] = SpElement_list_ptr[i];
    }
}

void Create_Matrix_B_data_FPGA(const INDEX_TYPE K,
                               const INDEX_TYPE N,
                               const INDEX_TYPE HBM_CHANNEL_B_NUM,
                               const vector<VALUE_TYPE> &Matrix_B_CPU_Dense,
                               vector<aligned_vector<VALUE_TYPE> > &Matrix_B_fpga_data
                              ) {
    INDEX_TYPE mat_B_fpga_column_size;

    if(HBM_CHANNEL_B_NUM == 8) {
        mat_B_fpga_column_size = ((K + 16 - 1) / 16) * 16;
    }
    else if(HBM_CHANNEL_B_NUM == 4) {
        mat_B_fpga_column_size = ((K + 8 - 1) / 8) * 8 * 2;
    }

    INDEX_TYPE mat_B_fpga_chunk_size = ((mat_B_fpga_column_size * (N / 8) + 1023)/1024) * 1024;

    for(INDEX_TYPE c = 0; c < HBM_CHANNEL_B_NUM; ++c) {
        Matrix_B_fpga_data[c].resize(mat_B_fpga_chunk_size, 0.0);
    }
    for(INDEX_TYPE nn = 0; nn < N; ++nn) {
        for(INDEX_TYPE kk = 0; kk < K; ++kk) {
            INDEX_TYPE pos = (kk / 8) * 16 + (nn % 2) * 8 + kk % 8 + mat_B_fpga_column_size * (nn / 8);     
            Matrix_B_fpga_data[(nn/2) % 4][pos] = Matrix_B_CPU_Dense[kk + K * nn];
        }
    }
}


void Create_Matrix_C_data_FPGA(const INDEX_TYPE M,
                               const INDEX_TYPE N,
                               const INDEX_TYPE HBM_CHANNEL_C_NUM,
                               const vector<VALUE_TYPE> &Matrix_C_CPU_Dense,
                               vector<aligned_vector<VALUE_TYPE> > &Matrix_C_fpga_data
                              ) {
    INDEX_TYPE mat_C_fpga_column_size = ((M + 16 - 1) / 16) * 16;
    INDEX_TYPE mat_C_fpga_chunk_size = ((mat_C_fpga_column_size * (N / 8) + 1023)/1024) * 1024;
    for(INDEX_TYPE c = 0; c < HBM_CHANNEL_C_NUM; ++c) {
        Matrix_C_fpga_data[c].resize(mat_C_fpga_chunk_size, 0.0);
    }
                              
}

void Verify_correctness(INDEX_TYPE &error_num,
                        const VALUE_TYPE &CPU_val,
                        const VALUE_TYPE &FPGA_val,
                        const double     threshold = 1e-4
                       ) {
    double difference = fabs(CPU_val - FPGA_val);
    double x = min(fabs(CPU_val), fabs(FPGA_val)) + threshold;
    if(difference / x > threshold) {
        error_num++;
    }
}

#endif
