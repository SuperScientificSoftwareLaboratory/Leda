tapac \
  --work-dir run \
  --top Leda \
  --platform xilinx_u280_xdma_201920_3 \
  --clock-period 3.33 \
  -o Leda.xo \
  --constraint Leda_floorplan.tcl \
  --connectivity ../link_config_4.ini \
  --read-only-args SpElement_list_ptr \
  --read-only-args Matrix_A_data* \
  --read-only-args Matrix_B_data* \
  --write-only-args Matrix_C_data* \
  --enable-synth-util \
  --max-parallel-synth-jobs 16 \
  --enable-hbm-binding-adjustment \
  --run-floorplan-dse \
  ../src/leda.cpp \
  2>&1 | tee leda_tapa.log
