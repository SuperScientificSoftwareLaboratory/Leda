#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iostream>

#include <ap_int.h>
#include <tapa.h>

#include "mmio.h"
#include "leda.h"
#include "leda_common.h"

using namespace std;


int main(int argc, char **argv) {
    cout << "Run SpMM... " << endl;
    
    INDEX_TYPE ITERATION_NUM = 1;

    if(argc == 4) {
        ITERATION_NUM = atoi(argv[3]);
    }
    else if(argc != 3) {
        cout << "Message: " << argv[0] << " [Sparse Matrix Path] [N] [ITERATION_NUM] " << std::endl;
        return EXIT_FAILURE;
    }

    char *filename = argv[1];

    INDEX_TYPE N = tapa::round_up<8>(atoi(argv[2]));

    std::string bitstream;
    if(const auto bitstream_ptr = getenv("BITFILE")) {
        bitstream = bitstream_ptr;
    }

    cout << "\nConfiguration : \n";
    cout << "Iter_num = " << ITERATION_NUM <<  "\n";

    cout << "N = " << N <<  "\n";

    cout << "TileSize = " << Tile_SIZE << endl;

    cout << "HBM_CHANNEL_A_NUM = " << HBM_CHANNEL_A_NUM << endl;
    cout << "HBM_CHANNEL_B_NUM = " << HBM_CHANNEL_B_NUM << endl;
    cout << "HBM_CHANNEL_C_NUM = " << HBM_CHANNEL_C_NUM << endl;

    INDEX_TYPE M, K, nnzR, isSymmetric;

    Read_matrix_size(filename,
                     &M,
                     &K,
                     &nnzR,
                     &isSymmetric
                    );

    cout << "\nMatrix Size: \n";
    cout << "Sparse matrix A: #Rows = " << M << ", #Cols = " << K << ", #nnzR = " << nnzR <<  "\n";
    cout << "Dense  matrix B: #Rows = "  << K << ", #Cols = " << N << "\n";
    cout << "Dense  matrix C: #Rows = "  << M << ", #Cols = " << N << "\n";

    cout << "\nCreate Date Struct: \n";
    vector<INDEX_TYPE> RowPtr_CSR(M + 1, 0);
    vector<INDEX_TYPE> ColIdx_CSR(nnzR, 0);
    vector<VALUE_TYPE> Val_CSR(nnzR, 0.0);

    vector<INDEX_TYPE> ColPtr_CSC(K + 1, 0);
    vector<INDEX_TYPE> RowIdx_CSC(nnzR, 0);
    vector<VALUE_TYPE> Val_CSC(nnzR, 0.0);

    vector<INDEX_TYPE> RowIdx_COO(nnzR, 0);
    vector<INDEX_TYPE> ColIdx_COO(nnzR, 0);
    vector<VALUE_TYPE> Val_COO(nnzR, 0.0);


    cout << "Reading Sparse Matrix A... ";
    
    Read_matrix_2_CSC(filename, 
                      M, 
                      K, 
                      nnzR, 
                      ColPtr_CSC, 
                      RowIdx_CSC, 
                      Val_CSC
                     );

    cout << "done\n";
    
    CSC_2_COO(M, 
              K, 
              nnzR,
              ColPtr_CSC, 
              RowIdx_CSC, 
              Val_CSC,
              RowIdx_COO,
              ColIdx_COO,
              Val_COO
             );

    cout << "Create Matrix Band... ";
    vector<Matrix_COO> Matrix_Band_COO(PE_NUM * HBM_CHANNEL_A_NUM);

    Matrix_Scatter(M,
                   K,
                   nnzR,
                   RowIdx_COO,
                   ColIdx_COO,
                   Val_COO,
                   PE_NUM * HBM_CHANNEL_A_NUM,
                   Matrix_Band_COO
                  );

    cout << "done\n";

    cout << "Create Matrix Band Tile... ";

    vector<SparseTile> Matrix_Band_Tile(PE_NUM * HBM_CHANNEL_A_NUM);

    Create_Matrix_Band_SparseTile_ex(Matrix_Band_COO,
                                      Matrix_Band_Tile
                                     );

    cout << "done\n";

    vector<VALUE_TYPE> Matrix_B_CPU_Dense(K * N, 0.0);
    vector<VALUE_TYPE> Matrix_C_CPU_Dense(M * N, 0.0);


    cout << "Create Dense Matirx B... ";

    Generate_Dense_Matrix(K, N, 1.0, Matrix_B_CPU_Dense, false, false);
    
    cout << "done\n";

    cout << "Create Dense Matirx C... ";

    for(INDEX_TYPE nn = 0; nn < N; ++nn) {
        for(INDEX_TYPE mm = 0; mm < M; ++mm) {
            Matrix_C_CPU_Dense[nn * M + mm] = 0.0;
        }
    }

    cout << "done\n";

    cout << "Create SpElement_list... ";

    vector<vector<SpElement> > SpElement_list_pes;
    vector<INDEX_TYPE> SpElement_list_ptr;
    
    
    Create_SpElement_list_for_all_PEs(HBM_CHANNEL_A_NUM * PE_NUM, 
                                      M, 
                                      K, 
                                      Tile_SIZE, 
                                      BATCH_SIZE, 
                                      Matrix_Band_Tile, 
                                      SpElement_list_pes, 
                                      SpElement_list_ptr,
                                      WINDOWS
                                     );   
    cout << "done\n";

    cout << "\nCreate Date for FPGA: \n";
    cout << "Create SpElement_list data for FPGA... ";

    aligned_vector<INDEX_TYPE> SpElement_list_ptr_fpga;
    Create_SpElement_list_data_FPGA(SpElement_list_ptr, SpElement_list_ptr_fpga);

    cout << "done\n";

    cout << "Create Sparse Matrix A data for FPGA... ";

    vector<aligned_vector<unsigned long> > Matrix_A_fpga_data(HBM_CHANNEL_A_NUM);
    Create_SpElement_list_for_all_channels(SpElement_list_pes,
                                           SpElement_list_ptr,
                                           Matrix_A_fpga_data,
                                           HBM_CHANNEL_A_NUM
                                          );

    cout << "done\n";

    cout << "Create Dense Matrix B data for FPGA... ";

    vector<aligned_vector<VALUE_TYPE> > Matrix_B_fpga_data(HBM_CHANNEL_B_NUM);
    Create_Matrix_B_data_FPGA(K,
                              N,
                              HBM_CHANNEL_B_NUM,
                              Matrix_B_CPU_Dense,
                              Matrix_B_fpga_data
                             );
    cout << "done\n";

    cout << "Create Dense Matrix C data for FPGA... ";

    vector<aligned_vector<VALUE_TYPE> > Matrix_C_fpga_data(HBM_CHANNEL_C_NUM);

    Create_Matrix_C_data_FPGA(M,
                              N,
                              HBM_CHANNEL_C_NUM,
                              Matrix_C_CPU_Dense,
                              Matrix_C_fpga_data
                             );

    cout << "done\n";
    
    cout << "\nRun kernel: \n";
    cout << "Run SpMM on CPU... ";
    auto CPU_start = std::chrono::steady_clock::now();

    SpMM_CPU_Tile(M,
                   N, 
                   K, 
                   Matrix_Band_Tile, 
                   Matrix_B_CPU_Dense, 
                   Matrix_C_CPU_Dense
                  );

    auto CPU_end = std::chrono::steady_clock::now();
    cout << "done\n";

    double CPU_time = std::chrono::duration_cast<std::chrono::nanoseconds>(CPU_end - CPU_start).count();
    CPU_time *= 1e-9;
    printf("CPU time is %f ms\n", CPU_time * 1000);
    cout << "CPU GFLOPS: " << (2.0 * N * nnzR) / 1e9 / CPU_time << endl << endl;

    INDEX_TYPE Batch_num = SpElement_list_ptr.size() - 1;
    INDEX_TYPE Sparse_Matrix_len = SpElement_list_ptr[Batch_num];

    cout << "Run SpMM on FPGA... ";
    double FPGA_time = tapa::invoke(Leda, 
                                    bitstream,
                                    tapa::read_only_mmap<INDEX_TYPE>(SpElement_list_ptr_fpga),
                                    tapa::read_only_mmaps<unsigned long, HBM_CHANNEL_A_NUM>(Matrix_A_fpga_data).reinterpret<ap_uint<512>>(),
                                    tapa::read_only_mmaps<VALUE_TYPE,    HBM_CHANNEL_B_NUM>(Matrix_B_fpga_data).reinterpret<VALUE_TYPE_v16>(),
                                    tapa::write_only_mmaps<VALUE_TYPE,   HBM_CHANNEL_C_NUM>(Matrix_C_fpga_data).reinterpret<VALUE_TYPE_v16>(),
                                    Batch_num,
                                    Sparse_Matrix_len,
                                    M,
                                    K,
                                    N,
                                    ITERATION_NUM
                                   );
    cout << "done\n";
    FPGA_time *= (1e-9 / ITERATION_NUM);
    printf("FPGA time is %f ms\n", FPGA_time * 1000);

    float GFLOPS = (2.0 * N * nnzR) / 1e9 / FPGA_time;
    printf("FPGA GFLOPS: %f \n", GFLOPS);

    INDEX_TYPE error_num = 0;
    INDEX_TYPE mat_C_fpga_column_size = ((M + 16 - 1) / 16) * 16;

    cout << "Verify the correctness of result... ";

    for(INDEX_TYPE nn = 0; nn < N; ++nn) {
        for(INDEX_TYPE mm = 0; mm < M; ++mm) {
            VALUE_TYPE CPU_val = Matrix_C_CPU_Dense[mm + nn * M];

            INDEX_TYPE pos = mat_C_fpga_column_size * (nn / 8) + mm;
            VALUE_TYPE FPGA_val = Matrix_C_fpga_data[nn % 8][pos];

            Verify_correctness(error_num, CPU_val, FPGA_val, 1e-4);
        }
    }
    cout << "done\n";

    float diffpercent = 100.0 * error_num / M / N;
    bool ispass = diffpercent < 2.0;

    if(ispass){
        cout << "||PASSED||\n";
    }
    else{
        cout << "[[FAILED]]\n";
    }
    printf("error_num = [%d], percent = [%.2f%%]\n", error_num, diffpercent);

    return EXIT_SUCCESS;
}
