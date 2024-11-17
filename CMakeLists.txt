cmake_minimum_required(VERSION 3.14)

project(leda)

set(PLATFORM
    xilinx_u280_xdma_201920_3
    CACHE STRING "Target FPGA platform")

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-write-strings")

find_package(TAPA REQUIRED)
find_package(SDx REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(leda)
target_sources(leda PRIVATE src/leda_host.cpp src/leda.cpp)
target_link_libraries(leda PRIVATE tapa::tapa)

target_link_libraries(leda PUBLIC OpenMP::OpenMP_CXX)

add_tapa_target(
  hls
  --enable-synth-util
  INPUT src/leda.cpp
  TOP Leda
  CONNECTIVITY ${CMAKE_CURRENT_SOURCE_DIR}/link_config_4.ini
  CONSTRAINT ${CMAKE_CURRENT_BINARY_DIR}/constraint.tcl
  --enable-hbm-binding-adjustment
  --read-only-args SpElement_list_ptr
  --read-only-args Matrix_A_data*
  --read-only-args Matrix_B_data*
  --write-only-args Matrix_C_data*
  --max-slr-width-limit 11000
  PLATFORM ${PLATFORM})

add_xocc_hw_link_targets(
  ${CMAKE_CURRENT_BINARY_DIR}
  --config=${CMAKE_CURRENT_SOURCE_DIR}/link_config_4.ini
  --vivado.prop run.impl_1.STEPS.PHYS_OPT_DESIGN.is_enabled=1
  --vivado.prop run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=Explore
  --vivado.prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=EarlyBlockPlacement
  --vivado.prop run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=Explore
  --vivado.prop run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore
  --vivado.prop run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=${CMAKE_CURRENT_BINARY_DIR}/constraint.tcl
  INPUT hls
  HW_EMU_XCLBIN hw_emu_xclbin
  HW_XCLBIN hw_xclbin)

add_custom_target(
  swsim
  COMMAND $<TARGET_FILE:leda> ../matrices/G55/G55.mtx 8 1
  DEPENDS leda
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(
  hwsim
  COMMAND BITFILE=$<TARGET_PROPERTY:${hw_emu_xclbin},FILE_NAME> $<TARGET_FILE:leda>  ../matrices/G55/G55.mtx 8 1
  DEPENDS leda ${hw_emu_xclbin}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
