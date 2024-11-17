# Leda

Leda is a high-performance SpMV accelerator on HBM-equipped FPGAs for GNNs.

## Installation

```text
mkdir build
cd build
cmake ..
```

## Software Emulation

```text
make swsim
```

## Hardware Emulation

```text
make hwsim
```

## Generate Bitstream

```text
sh run_generate.sh
```

## Run Cuper on FPGA

```text
BITFILE=../bitfile/Leda_xilinx_u280_xdma_201920_3.xclbin ./leda ../matrices/G55/G55.mtx 8 100
```

## Reference

Enxin Yi, Jiarui Bai, Yijie Nie, Dan Niu, Zhou Jin, Weifeng Liu. "Leda: Leveraging Tiling Dataflow to Accelerate SpMM on HBM-Equipped FPGAs for GNNs", ACM/IEEE International Conference on Computer-Aided Design (ICCAD), Oct 27-31, 2024.