#include <ap_int.h>
#include <cstdio>
#include <cstring>
#include <cassert>

#include <tapa.h>

#include "leda.h"

struct Matrix_Mult {
    ap_uint<18> row;
    VALUE_TYPE_v8 val;
};

template <typename T1, typename T2>
inline void Async_Read(tapa::async_mmap<T1> &mmap_in,
                       tapa::ostream<T1> &Stream_out,
                       const T2 mmap_in_len,
                       T2 &i_request,
                       T2 &i_response
                      ) {

#pragma HLS inline
    if((i_request < mmap_in_len) & !mmap_in.read_addr.full()) {
        mmap_in.read_addr.try_write(i_request);
        ++i_request;
    }
    if(!Stream_out.full() & !mmap_in.read_data.empty()) {
        T1 temp;
        mmap_in.read_data.try_read(temp);
        Stream_out.try_write(temp);
        ++i_response;
    }
}

void SpElement_list_ptr_Loader(const INDEX_TYPE Batch_num,
                               const INDEX_TYPE M,
                               const INDEX_TYPE N,
                               const INDEX_TYPE K, 
                               const INDEX_TYPE Iteration_num,
                               tapa::async_mmap<INDEX_TYPE> &SpElement_list_ptr,
                               tapa::ostream<INDEX_TYPE> &PE_Param
                              ) {
    
    PE_Param.write(Batch_num);
    PE_Param.write(M);
    PE_Param.write(N);
    PE_Param.write(K);
    PE_Param.write(Iteration_num);                           

    const INDEX_TYPE Iteration_time = (Iteration_num == 0) ? 1 : Iteration_num;
    
    const INDEX_TYPE Batch_num_plus_1 = Batch_num + 1;

    const INDEX_TYPE Iteration_time_N = Iteration_time * ((N + 7) >> 3);
iter:
    for(INDEX_TYPE iter = 0; iter < Iteration_time_N; ++iter) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    Load_ptr:
        for(INDEX_TYPE i_request = 0, i_response = 0; i_response < Batch_num_plus_1;) {
#pragma HLS loop_tripcount min=1 max=800
#pragma HLS pipeline II=1
            Async_Read(SpElement_list_ptr,
                       PE_Param,
                       Batch_num_plus_1,
                       i_request, 
                       i_response
                      );
        }
    }
}

void Sparse_Matrix_Loader(const INDEX_TYPE Matrix_len,
                          const INDEX_TYPE N, 
                          const INDEX_TYPE Iteration_num,
                          tapa::async_mmap<ap_uint<512>> &Matrix_A_data,
                          tapa::ostream<ap_uint<512>> &Matrix_A_Stream
                          ) {

    const INDEX_TYPE Iteration_time = (Iteration_num == 0) ? 1 : Iteration_num;
    const INDEX_TYPE Iteration_time_N = Iteration_time * ((N + 7) >> 3);
iter:
    for(INDEX_TYPE iter = 0; iter < Iteration_time_N; ++iter) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16

    Load_A:
        for(INDEX_TYPE i_request = 0, i_response = 0; i_response < Matrix_len;) {
#pragma HLS loop_tripcount min=1 max=10000
#pragma HLS pipeline II=1
            Async_Read(Matrix_A_data,
                       Matrix_A_Stream,
                       Matrix_len,
                       i_request, 
                       i_response
                      );
        }
    }
}

void Segment(tapa::istream<ap_uint<512>> & Matrix_A_Stream,
             tapa::ostreams<ap_uint<256>, 2> & Matrix_A_Stream_256
            ) {
Seg:
    for(;;) {
#pragma HLS pipeline II=1
        bool flag_nop = Matrix_A_Stream.empty();
        for(INDEX_TYPE i = 0; i < 2; ++i) {
            flag_nop |= Matrix_A_Stream_256[i].full();
        }
        if(!flag_nop) {
            ap_uint<512> tmp; Matrix_A_Stream.try_read(tmp);
            for(INDEX_TYPE i = 0; i < 2; ++i) {
                Matrix_A_Stream_256[i].try_write(tmp(255 + i * 256, i * 256));
            }
        }
    }
}

void Dense_Matrix_Loader(const INDEX_TYPE K,
                         const INDEX_TYPE N,
                         const INDEX_TYPE Iteration_num,
                         tapa::async_mmap<VALUE_TYPE_v16> & Matrix_B_data,
                         tapa::ostream<VALUE_TYPE_v16> & Matrix_B_Stream
                        ) {
    const INDEX_TYPE Iteration_time = (Iteration_num == 0) ? 1 : Iteration_num;
    const int Iteration_num_B = ((K + 7) >> 3) * ((N + 7) >> 3);
    
iter:
    for(INDEX_TYPE rp = 0; rp < Iteration_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    Load_B:
        for(INDEX_TYPE i_req = 0, i_resp = 0; i_resp < Iteration_num_B;) {
#pragma HLS loop_tripcount min=1 max=500000
#pragma HLS pipeline II=1
            Async_Read(Matrix_B_data,
                       Matrix_B_Stream,
                       Iteration_num_B,
                       i_req,
                       i_resp);
        }
    }
}

void Dense_Matrix_Writer(const INDEX_TYPE M,
                         const INDEX_TYPE N,
                         const INDEX_TYPE Iteration_num,
                         tapa::istream<VALUE_TYPE_v16> & Matrix_C_Stream,
                         tapa::async_mmap<VALUE_TYPE_v16> & Matrix_C_date
                        ) {
    const INDEX_TYPE Iteration_time = (Iteration_num == 0) ? 1 : Iteration_num;
    const INDEX_TYPE Iteration_num_C = ((M + 15) >> 4) * ((N + 7) >> 3);
    
iter:
    for(INDEX_TYPE rp = 0; rp < Iteration_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    Write_C:
        for(INDEX_TYPE i_req = 0, i_resp = 0; i_resp < Iteration_num_C;) {
#pragma HLS loop_tripcount min=1 max=500000
#pragma HLS pipeline II=1
            if((i_req < Iteration_num_C) & !Matrix_C_Stream.empty() & !Matrix_C_date.write_addr.full() & !Matrix_C_date.write_data.full() ) {
                Matrix_C_date.write_addr.try_write(i_req);
		VALUE_TYPE_v16 tmpv;
		Matrix_C_Stream.try_read(tmpv);
                Matrix_C_date.write_data.try_write(tmpv);
                ++i_req;
            }
	    uint8_t n_resp;
            if(Matrix_C_date.write_resp.try_read(n_resp)) {
                i_resp += INDEX_TYPE(n_resp) + 1;
            }
        }
    }
}


void Outer_Product_Unit_Merge(ap_uint<14> B_row,
                              ap_uint<14> B_row_old,
                              ap_uint<32> A_val,
                              VALUE_TYPE Matrix_B_onchip[8][Tile_WIDTH],
                              VALUE_TYPE Matrix_B_reusequeue[8],
                              VALUE_TYPE_v8 & val
                             ) {
#pragma HLS inline
    VALUE_TYPE A_val_float = tapa::bit_cast<VALUE_TYPE>(A_val);
    if(B_row_old & B_row == 0x3FFF) {
        for(INDEX_TYPE i = 0; i < 8; ++i) {
            val[i] = A_val_float * Matrix_B_reusequeue[i];
        }
    }
    else{
        for(INDEX_TYPE i = 0; i < 8; ++i) {
            val[i] = A_val_float * Matrix_B_onchip[i][B_row];
            Matrix_B_reusequeue[i] = Matrix_B_onchip[i][B_row];
        }
    }
    B_row_old = B_row;
}

void MMU(tapa::istream<INDEX_TYPE> &PE_Param_in,
         tapa::istream<ap_uint<256>> &Matrix_A_Stream_256,
         tapa::istreams<VALUE_TYPE_v16, HBM_CHANNEL_B_NUM> &Matrix_B_Stream_in, 
         tapa::ostream<INDEX_TYPE> &PE_Param_out,
         tapa::ostreams<VALUE_TYPE_v16, HBM_CHANNEL_B_NUM> &Matrix_B_Stream_out,
         tapa::ostream<INDEX_TYPE> &PE_Param_to_C,
         tapa::ostreams<Matrix_Mult, 4> &Matrix_Mult_Matrix_Stream
        ) {
    const INDEX_TYPE Batch_num = PE_Param_in.read();
    const INDEX_TYPE M = PE_Param_in.read();
    const INDEX_TYPE N = PE_Param_in.read();
    const INDEX_TYPE K = PE_Param_in.read();
    const INDEX_TYPE Iteration_num = PE_Param_in.read();

    PE_Param_out.write(Batch_num);
    PE_Param_out.write(M);
    PE_Param_out.write(N);
    PE_Param_out.write(K);
    PE_Param_out.write(Iteration_num);
    
    PE_Param_to_C.write(Batch_num);
    PE_Param_to_C.write(M);
    PE_Param_to_C.write(N);
    PE_Param_to_C.write(Iteration_num);

    const INDEX_TYPE Iteration_time = (Iteration_num == 0) ? 1 : Iteration_num;
    
    const INDEX_TYPE Iteration_time_N = Iteration_time * ((N + 7) >> 3);
    
iter:
    for(INDEX_TYPE rp = 0; rp < Iteration_time_N; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
        
        VALUE_TYPE Matrix_B_onchip[4/2][8][Tile_WIDTH];
#pragma HLS bind_storage variable=Matrix_B_onchip latency=2
#pragma HLS array_partition variable=Matrix_B_onchip complete dim=1
#pragma HLS array_partition variable=Matrix_B_onchip complete dim=2
#pragma HLS array_partition variable=Matrix_B_onchip cyclic factor=B_PARTITION_FACTOR dim=3
        
        INDEX_TYPE start_32 = PE_Param_in.read();
        PE_Param_out.write(start_32);
        PE_Param_to_C.write(start_32);
        
    main:
        for(INDEX_TYPE i = 0; i < Batch_num; ++i) {
#pragma HLS loop_tripcount min=1 max=49
            
        Fill_B_onchip:
            for(INDEX_TYPE j = 0; (j < (Tile_WIDTH >> 3)) && (j < ((K + 7) >> 3) - i * (Tile_WIDTH >> 3)); ) {
#pragma HLS loop_tripcount min=1 max=512
#pragma HLS pipeline II = 1
                
                bool b_2048_ready = true;
                bool b_2048_out_not_full = true;
                for(INDEX_TYPE k = 0; k < HBM_CHANNEL_B_NUM; ++k) {
                    b_2048_ready &= !Matrix_B_Stream_in[k].empty();
                    b_2048_out_not_full &= !Matrix_B_Stream_out[k].full();
                }
                
                if(b_2048_ready & b_2048_out_not_full) {
                    VALUE_TYPE_v16 b_512_x[HBM_CHANNEL_B_NUM];
                    for(INDEX_TYPE k = 0; k < HBM_CHANNEL_B_NUM; ++k) {
                        Matrix_B_Stream_in[k].try_read(b_512_x[k]);
                        Matrix_B_Stream_out[k].try_write(b_512_x[k]);
                    }
                    for(INDEX_TYPE k = 0; k < 8; ++k) {
                        for(INDEX_TYPE m = 0; m < 8; ++m) {
                            for(INDEX_TYPE l = 0; l < 2; ++l) {
                                Matrix_B_onchip[l][m][(j << 3) + k] = b_512_x[m / 2][k + m % 2 * 8];
                            }
                        }
                    }
                    ++j;
                }
            }
            
            const INDEX_TYPE end_32 = PE_Param_in.read();
            PE_Param_out.write(end_32);
            PE_Param_to_C.write(end_32);
            
        Matrix_mult:
            for(INDEX_TYPE j = start_32; j < end_32; ) {
#pragma HLS loop_tripcount min=1 max=200
#pragma HLS pipeline II=1                

                ap_uint<256> a_pes;

                ap_uint<14> col_old[4];
                for(INDEX_TYPE col = 0; col < 4; ++col) {
                    col_old[col] = 0x3FFF;
                }
                VALUE_TYPE Matrix_B_reusequeue[4][8]; 
#pragma HLS bind_storage variable=Matrix_B_reusequeue latency=2
#pragma HLS array_partition variable=Matrix_B_reusequeue complete dim=1
#pragma HLS array_partition variable=Matrix_B_reusequeue complete dim=2

                bool a_pes_ready = Matrix_A_Stream_256.try_read(a_pes);
                
                if(a_pes_ready) {
                         
                PE:
                    for(INDEX_TYPE p = 0; p < 4; ++p) {
                        ap_uint<64> a = a_pes(63 + p * 64, p * 64);
                        
                        ap_uint<14> a_col = a(63, 50);
                        ap_uint<18> a_row = a(49, 32);
                        ap_uint<32> a_val = a(31,  0);
                        
                        Matrix_Mult mult_val;
                        mult_val.row = a_row;

                        if (a_row[17] == 0) {
                            Outer_Product_Unit_Merge(a_col,
                                                     col_old[p],
                                                     a_val,
                                                     Matrix_B_onchip[p/2],
                                                     Matrix_B_reusequeue[p],
                                                     mult_val.val
                                                    );
                        }
                        Matrix_Mult_Matrix_Stream[p].write(mult_val);
                    }
                    ++j;
                }
            }
            start_32 = end_32;
        }
    }
}

void Adder(ap_uint<18> C_row,
           VALUE_TYPE val_d0_float_old,
           VALUE_TYPE val_d1_float_old,
           ap_uint<64> Matrix_C_onchip[URAM_DEPTH]
          ) {
#pragma HLS inline
    ap_uint<64> val_d0_d1_u64 = Matrix_C_onchip[C_row];
    
    ap_uint<32> val_d0_u32 = val_d0_d1_u64(31,  0);
    ap_uint<32> val_d1_u32 = val_d0_d1_u64(63, 32);

    VALUE_TYPE val_d0_float_new = tapa::bit_cast<VALUE_TYPE>(val_d0_u32) + val_d0_float_old;
    VALUE_TYPE val_d1_float_new = tapa::bit_cast<VALUE_TYPE>(val_d1_u32) + val_d1_float_old;

    val_d0_u32 = tapa::bit_cast<ap_uint<32>>(val_d0_float_new);
    val_d1_u32 = tapa::bit_cast<ap_uint<32>>(val_d1_float_new);
    
    val_d0_d1_u64(31,  0) = val_d0_u32;
    val_d0_d1_u64(63, 32) = val_d1_u32;
    
    Matrix_C_onchip[C_row] = val_d0_d1_u64;
}

void Adder_Unit(ap_uint<18> C_row,
                VALUE_TYPE_v8 & val,
                ap_uint<64> Matrix_C_onchip[4][URAM_DEPTH]
               ) {
#pragma HLS inline
    for(INDEX_TYPE i = 0; i < 4; ++i) {
        Adder(C_row,
              val[i * 2 + 0],
              val[i * 2 + 1],
              Matrix_C_onchip[i]
             );
    }
}

void MAU(tapa::istreams<INDEX_TYPE, 2> &PE_inst_in,
         tapa::istreams<Matrix_Mult, 8> &Matrix_Mult_Matrix_Stream,
         tapa::ostream<VALUE_TYPE_v16> &Matrix_C_Stream_out
        ) {

    const INDEX_TYPE Batch_num = PE_inst_in[0].read();
    const INDEX_TYPE M = PE_inst_in[0].read();
    const INDEX_TYPE N = PE_inst_in[0].read();
    const INDEX_TYPE Iteration_num = PE_inst_in[0].read();
    
    INDEX_TYPE tmp;
Destroy_PE_inst:
    for(INDEX_TYPE i = 0; i < 4; ++i) {
        tmp = PE_inst_in[1].read();
    }

    const INDEX_TYPE Iteration_time = (Iteration_num == 0) ? 1 : Iteration_num;
    const INDEX_TYPE Iteration_time_N = Iteration_time * ((N + 7) >> 3);

    const INDEX_TYPE num_v_init = (M + 63) >> 6;
    const INDEX_TYPE num_v_out = (M + 15) >> 4;

    ap_uint<64> Matrix_C_onchip[8][8 / 2][URAM_DEPTH];
#pragma HLS bind_storage variable=Matrix_C_onchip type=RAM_2P impl=URAM latency=1
#pragma HLS array_partition complete variable=Matrix_C_onchip dim=1
#pragma HLS array_partition complete variable=Matrix_C_onchip dim=2
    
iter:
    for(INDEX_TYPE rp = 0; rp < Iteration_time_N; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
        
    Init_C_onchip:
        for(INDEX_TYPE i = 0; i < num_v_init; ++i) {
#pragma HLS loop_tripcount min=1 max=800
#pragma HLS pipeline II=1

            for(INDEX_TYPE j = 0; j < 8; ++j) {
                for(INDEX_TYPE k = 0; k < 4; ++k) {
                    Matrix_C_onchip[j][k][i] = 0;
                }
            }
        }
        
        INDEX_TYPE start_32 = PE_inst_in[0].read();
        tmp = PE_inst_in[1].read();

        
    main:
        for(INDEX_TYPE i = 0; i < Batch_num; ++i) {
#pragma HLS loop_tripcount min=1 max=49
            
            const INDEX_TYPE end_32 = PE_inst_in[0].read();
            tmp = PE_inst_in[1].read();

        Accumulate:
            for(INDEX_TYPE j = start_32; j < end_32; ) {
#pragma HLS loop_tripcount min=1 max=200
#pragma HLS pipeline II=1
#pragma HLS dependence true variable=Matrix_C_onchip distance=WINDOWS
                bool nop_flag = false;

                for(INDEX_TYPE p = 0; p < 8; ++p) {
                    nop_flag |= Matrix_Mult_Matrix_Stream[p].empty();
                }
                
                if(!nop_flag) {

                    for(INDEX_TYPE p = 0; p < 8; ++p) {
                        Matrix_Mult mult_val; 
                        Matrix_Mult_Matrix_Stream[p].try_read(mult_val);
                        ap_uint<18> a_row = mult_val.row;
                        
                        if(a_row[17] == 0) {
                            Adder_Unit(a_row,
                                       mult_val.val,
                                       Matrix_C_onchip[p]
                                      );
                        }
                    }
                    ++j;
                }
            }
            start_32 = end_32;
        }

Write_C_onchip:
        for(INDEX_TYPE i = 0; i < num_v_out; ++i) {
#pragma HLS loop_tripcount min=1 max=1800
#pragma HLS pipeline II=1

            ap_uint<64> u_64_pe_d[2][4];
#pragma HLS array_partition variable=u_64_pe_d complete
            ap_uint<32> u_32_d[8][2];
#pragma HLS array_partition variable=u_32_d complete

            switch(i % 4) {
					case 0:
						u_64_pe_d[0][0] = Matrix_C_onchip[0][0][i/4];
						u_64_pe_d[0][1] = Matrix_C_onchip[0][1][i/4];
						u_64_pe_d[0][2] = Matrix_C_onchip[0][2][i/4];
						u_64_pe_d[0][3] = Matrix_C_onchip[0][3][i/4];

						u_64_pe_d[1][0] = Matrix_C_onchip[1][0][i/4];
						u_64_pe_d[1][1] = Matrix_C_onchip[1][1][i/4];
						u_64_pe_d[1][2] = Matrix_C_onchip[1][2][i/4];
						u_64_pe_d[1][3] = Matrix_C_onchip[1][3][i/4];

						break;
					case 1:
						u_64_pe_d[0][0] = Matrix_C_onchip[2][0][i/4];
						u_64_pe_d[0][1] = Matrix_C_onchip[2][1][i/4];
						u_64_pe_d[0][2] = Matrix_C_onchip[2][2][i/4];
						u_64_pe_d[0][3] = Matrix_C_onchip[2][3][i/4];

						u_64_pe_d[1][0] = Matrix_C_onchip[3][0][i/4];
						u_64_pe_d[1][1] = Matrix_C_onchip[3][1][i/4];
						u_64_pe_d[1][2] = Matrix_C_onchip[3][2][i/4];
						u_64_pe_d[1][3] = Matrix_C_onchip[3][3][i/4];

						break;
					case 2:
						u_64_pe_d[0][0] = Matrix_C_onchip[4][0][i/4];
						u_64_pe_d[0][1] = Matrix_C_onchip[4][1][i/4];
						u_64_pe_d[0][2] = Matrix_C_onchip[4][2][i/4];
						u_64_pe_d[0][3] = Matrix_C_onchip[4][3][i/4];

						u_64_pe_d[1][0] = Matrix_C_onchip[5][0][i/4];
						u_64_pe_d[1][1] = Matrix_C_onchip[5][1][i/4];
						u_64_pe_d[1][2] = Matrix_C_onchip[5][2][i/4];
						u_64_pe_d[1][3] = Matrix_C_onchip[5][3][i/4];

						break;
					case 3:
						u_64_pe_d[0][0] = Matrix_C_onchip[6][0][i/4];
						u_64_pe_d[0][1] = Matrix_C_onchip[6][1][i/4];
						u_64_pe_d[0][2] = Matrix_C_onchip[6][2][i/4];
						u_64_pe_d[0][3] = Matrix_C_onchip[6][3][i/4];

						u_64_pe_d[1][0] = Matrix_C_onchip[7][0][i/4];
						u_64_pe_d[1][1] = Matrix_C_onchip[7][1][i/4];
						u_64_pe_d[1][2] = Matrix_C_onchip[7][2][i/4];
						u_64_pe_d[1][3] = Matrix_C_onchip[7][3][i/4];

						break;
				}

				for(INDEX_TYPE pe = 0; pe < 2; ++pe) {
					for(INDEX_TYPE d = 0; d < 4; ++d) {
						u_32_d[2 * d    ][pe] = (u_64_pe_d[pe][d])(31,  0);
						u_32_d[2 * d + 1][pe] = (u_64_pe_d[pe][d])(63, 32);
					}
				}

                VALUE_TYPE_v16 out;
				for(INDEX_TYPE d = 0; d < 8; ++d) {
					out[d * 2 + 0] = tapa::bit_cast<VALUE_TYPE>(u_32_d[d][0]);
                    out[d * 2 + 1] = tapa::bit_cast<VALUE_TYPE>(u_32_d[d][1]);
				}
            Matrix_C_Stream_out.write(out);
        }
    }
}


void Merger(tapa::istreams<VALUE_TYPE_v16, HBM_CHANNEL_C_NUM> & Matrix_C_Stream_in,
            tapa::ostreams<VALUE_TYPE_v16, HBM_CHANNEL_C_NUM> & Matrix_C_Stream_out
           ) {

    for (;;) {
#pragma HLS pipeline II=1

        bool flag_not_ready = true;
        bool flag_not_full = true;

        for(INDEX_TYPE i = 0; i < HBM_CHANNEL_C_NUM; ++i) {
            flag_not_ready &= !Matrix_C_Stream_in[i].empty();
            flag_not_full &= !Matrix_C_Stream_out[i].full();
        }

        if(flag_not_ready & flag_not_full) {

            VALUE_TYPE_v16 C_in[8];

            for(INDEX_TYPE i = 0; i < HBM_CHANNEL_C_NUM; ++i) {
                Matrix_C_Stream_in[i].try_read(C_in[i]);
            }

            VALUE_TYPE_v16 C_out;
#pragma HLS aggregate variable=C_out

            for(INDEX_TYPE i = 0; i < HBM_CHANNEL_C_NUM; ++i) {
                for(INDEX_TYPE j = 0; j < 8; ++j) {
                    for(INDEX_TYPE k = 0; k < 2; ++k) {
                        C_out[j * 2 + k] = C_in[j][i * 2 + k];
                    }
                }
                Matrix_C_Stream_out[i].try_write(C_out);
            }
        }
    }
}

void Destroy_int(tapa::istream<INDEX_TYPE> &Stream_in) {
    for(;;) {
#pragma HLS pipeline II=1
        INDEX_TYPE tmp; 
        Stream_in.try_read(tmp);
    }
}

void Destroy_float_v16(tapa::istream<VALUE_TYPE_v16> &Stream_in) {
    for(;;) {
#pragma HLS pipeline II=1
        VALUE_TYPE_v16 tmp; 
        Stream_in.try_read(tmp);
    }
}


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
          ) {
    tapa::streams<INDEX_TYPE, HBM_CHANNEL_A_NUM * UNIT_NUM + 1, FIFO_DEPTH> PE_Param("PE_Param");
        
    tapa::streams<INDEX_TYPE, HBM_CHANNEL_C_NUM * UNIT_NUM, FIFO_DEPTH> PE_Param_to_C("PE_Param_to_C");
    
    tapa::streams<ap_uint<512>, HBM_CHANNEL_A_NUM, FIFO_DEPTH> Matrix_A_Stream("Matrix_A_Stream");

    tapa::streams<ap_uint<256>, HBM_CHANNEL_A_NUM * UNIT_NUM, FIFO_DEPTH> Matrix_A_Stream_256("Matrix_A_Stream_256");

    tapa::streams<VALUE_TYPE_v16, (HBM_CHANNEL_A_NUM * UNIT_NUM + 1) * HBM_CHANNEL_B_NUM, FIFO_DEPTH> Matrix_B_Stream("Matrix_B_Stream");

    tapa::streams<VALUE_TYPE_v16, HBM_CHANNEL_C_NUM, FIFO_DEPTH> Matrix_C_Stream("Matrix_C_Stream");

    tapa::streams<Matrix_Mult, HBM_CHANNEL_A_NUM * 8, FIFO_DEPTH> Matrix_Mult_Matrix_Stream("Matrix_Mult_Matrix_Stream");
    
    tapa::streams<VALUE_TYPE_v16, HBM_CHANNEL_C_NUM, FIFO_DEPTH> Matrix_C_Result_Stream("Matrix_C_Result_Stream");
    
    tapa::task()

        .invoke(SpElement_list_ptr_Loader,
                Batch_num,
                M,
                N,
                K,
                Iteration_num,
                SpElement_list_ptr,
                PE_Param
                )
    
        .invoke<tapa::join, HBM_CHANNEL_A_NUM>(Sparse_Matrix_Loader,
                                               Sparse_Matrix_len,
                                               N,
                                               Iteration_num,
                                               Matrix_A_data,
                                               Matrix_A_Stream
                                              )

        .invoke<tapa::detach, HBM_CHANNEL_A_NUM>(Segment,
                                                 Matrix_A_Stream,
                                                 Matrix_A_Stream_256
                                                )                                   

        .invoke<tapa::join, HBM_CHANNEL_B_NUM>(Dense_Matrix_Loader,
                                               K,
                                               N,
                                               Iteration_num,
                                               Matrix_B_data,
                                               Matrix_B_Stream
                                              )
    
        .invoke<tapa::join, HBM_CHANNEL_A_NUM * UNIT_NUM>(MMU,
                                                          PE_Param,
                                                          Matrix_A_Stream_256,
                                                          Matrix_B_Stream,
                                                          PE_Param, 
                                                          Matrix_B_Stream,
                                                          PE_Param_to_C,
                                                          Matrix_Mult_Matrix_Stream
                                                         )

        .invoke<tapa::join, HBM_CHANNEL_C_NUM>(MAU,
                                               PE_Param_to_C,
                                               Matrix_Mult_Matrix_Stream,
                                               Matrix_C_Stream
                                              )
    
        .invoke<tapa::detach>(Destroy_int,
                              PE_Param
                             )

        .invoke<tapa::detach, HBM_CHANNEL_B_NUM>(Destroy_float_v16,
                                                 Matrix_B_Stream
                                                )

        .invoke<tapa::detach>(Merger,
                              Matrix_C_Stream,
                              Matrix_C_Result_Stream
                             )
        
        .invoke<tapa::join, HBM_CHANNEL_C_NUM>(Dense_Matrix_Writer,
                                               M,
                                               N,
                                               Iteration_num,
                                               Matrix_C_Result_Stream,
                                               Matrix_C_data
                                              )
    ;
}



