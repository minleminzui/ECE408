# ECE408
https://github.com/aschuh703/ECE408
https://wiki.illinois.edu/wiki/display/ECE408/Labs+and+Project
## MP0
- we simply fetch the CUDA device information with `cudaGetDeviceProperties`, my results are as follows
```
There are 8 devices supporting CUDA
Device 0 name: Tesla V100-SXM2-32GB
 Computational Capabilities: 7.0
 Maximum global memory size: 34089730048
 Maximum constant memory size: 65536
 Maximum shared memory size per block: 49152
 Maximum block dimensions: 1024 x 1024 x 64
 Maximum grid dimensions: 2147483647 x 65535 x 65535
 Warp size: 32
Device 1 name: Tesla V100-SXM2-32GB
 Computational Capabilities: 7.0
 Maximum global memory size: 34089730048
 Maximum constant memory size: 65536
 Maximum shared memory size per block: 49152
 Maximum block dimensions: 1024 x 1024 x 64
 Maximum grid dimensions: 2147483647 x 65535 x 65535
 Warp size: 32
Device 2 name: Tesla V100-SXM2-32GB
 Computational Capabilities: 7.0
 Maximum global memory size: 34089730048
 Maximum constant memory size: 65536
 Maximum shared memory size per block: 49152
 Maximum block dimensions: 1024 x 1024 x 64
 Maximum grid dimensions: 2147483647 x 65535 x 65535
 Warp size: 32
Device 3 name: Tesla V100-SXM2-32GB
 Computational Capabilities: 7.0
 Maximum global memory size: 34089730048
 Maximum constant memory size: 65536
 Maximum shared memory size per block: 49152
 Maximum block dimensions: 1024 x 1024 x 64
 Maximum grid dimensions: 2147483647 x 65535 x 65535
 Warp size: 32
Device 4 name: Tesla V100-SXM2-32GB
 Computational Capabilities: 7.0
 Maximum global memory size: 34089730048
 Maximum constant memory size: 65536
 Maximum shared memory size per block: 49152
 Maximum block dimensions: 1024 x 1024 x 64
 Maximum grid dimensions: 2147483647 x 65535 x 65535
 Warp size: 32
Device 5 name: Tesla V100-SXM2-32GB
 Computational Capabilities: 7.0
 Maximum global memory size: 34089730048
 Maximum constant memory size: 65536
 Maximum shared memory size per block: 49152
 Maximum block dimensions: 1024 x 1024 x 64
 Maximum grid dimensions: 2147483647 x 65535 x 65535
 Warp size: 32
Device 6 name: Tesla V100-SXM2-32GB
 Computational Capabilities: 7.0
 Maximum global memory size: 34089730048
 Maximum constant memory size: 65536
 Maximum shared memory size per block: 49152
 Maximum block dimensions: 1024 x 1024 x 64
 Maximum grid dimensions: 2147483647 x 65535 x 65535
 Warp size: 32
Device 7 name: Tesla V100-SXM2-32GB
 Computational Capabilities: 7.0
 Maximum global memory size: 34089730048
 Maximum constant memory size: 65536
 Maximum shared memory size per block: 49152
 Maximum block dimensions: 1024 x 1024 x 64
 Maximum grid dimensions: 2147483647 x 65535 x 65535
 Warp size: 32
```
## MP1 Vector Addition
```
cd MP1
sh run_datasets
```
## MP2 Simple Matrix Multiply
```
cd MP2
sh run_datasets
```