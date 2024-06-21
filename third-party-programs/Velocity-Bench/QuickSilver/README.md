# Migration example: Migrate QuickSilver to SYCL version
[SYCLomatic](https://github.com/oneapi-src/SYCLomatic) is a project to assist developers in migrating their existing code written in different programming languages to the SYCL* C++ heterogeneous programming model. It is an open-source version of the Intel® DPC++ Compatibility Tool.

This file lists the detailed steps to migrate CUDA version of [QuickSilver](https://github.com/oneapi-src/Velocity-Bench/tree/main/QuickSilver) to SYCL version with SYCLomatic. As follow table summarizes the migration environment, the software required, and so on.

   | Optimized for         | Description
   |:---                   |:---
   | OS                    | Linux* Ubuntu* 22.04
   | Software              | Intel® oneAPI Base Toolkit, SYCLomatic
   | What you will learn   | Migration of CUDA code, Run SYCL code on oneAPI and Intel device
   | Time to complete      | 15 minutes


## Migrating QuickSilver to SYCL

### 1 Prepare the migration
#### 1.1 Get the source code of QuickSilver
```sh
   $ git clone https://github.com/oneapi-src/Velocity-Bench.git
   $ export QuickSilver_HOME=/path/to/Velocity-Bench/QuickSilver
```
Summary of QuickSilver project source code: total 111 files.

```
   CUDA
   └── src
      ├── AtomicMacro.hh
      ├── BulkStorage.hh
      ├── cmdLineParser.cc
      ├── cmdLineParser.hh
      ├── CollisionEvent.hh
      ├── CommObject.hh
      ├── compile_commands.json
      ├── CoralBenchmark.cc
      ..................
      ├── Tallies.cc
      ├── Tallies.hh
      ├── Tuple4.hh
      ├── Tuple4ToIndex.hh
      ├── Tuple.hh
      ├── TupleToIndex.hh
      ├── utils.cc
      ├── utils.hh
      ├── utilsMpi.cc
      └── utilsMpi.hh
```
#### 1.2 Prepare migration tool and SYCL run environment

 * Install SYCL run environment [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html). After installation, Intel® DPC++ Compatibility tool is also available, set up the SYCL run environment as follows:

```
   $ source /opt/intel/oneapi/setvars.sh
   $ dpct --version  # Intel® DPC++ Compatibility tool version
```
 * If want to try the latest version of the compatibility tool, try to install SYCLomatic by downloading prebuild of [SYCLomatic release](https://github.com/oneapi-src/SYCLomatic/blob/SYCLomatic/README.md#Releases) or [build from source](https://github.com/oneapi-src/SYCLomatic/blob/SYCLomatic/README.md), as follow give the steps to install prebuild version: 
 ```
   $ export SYCLomatic_HOME=/path/to/install/SYCLomatic
   $ mkdir $SYCLomatic_HOME
   $ cd $SYCLomatic_HOME
   $ wget https://github.com/oneapi-src/SYCLomatic/releases/download/20240203/linux_release.tgz   #Change the timestamp 20240203 to latest one
   $ tar xzvf linux_release.tgz
   $ source setvars.sh
   $ dpct --version #SYCLomatic version
 ```
 
 
For more information on configuring environment variables, see [Use the setvars Script with Linux*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).
 
### 2 Generate the compilation database 
``` sh
$ cd ${QuickSilver_HOME}/CUDA/src
$ make clean
$ intercept-build make
$ ls compile_commands.json  # make sure compile_commands.json is generated
compile_commands.json
```
### 3 Migrate the source code and build script
```sh
# From the CUDA directory as root directory:
$ cd ${QuickSilver_HOME}/CUDA
$ dpct --in-root=. -p=./src/compile_commands.json --out-root=out --gen-build-script --cuda-include-path=/usr/local/cuda/include
```
Description of the options: 
 * `--in-root`: provide input files to specify where to locate the CUDA files that need migration.
 * `-p`: specify the compilation database to migrate the whole project.
 * `--out-root`: designate where to generate the resulting files (default is `dpct_output`).
 * `--gen-build-script`: generate the `Makefile.dpct` for the migrated code.

Now you can see the migrated files in the `out` folder as follow: 
   ```
      out/
      ├── MainSourceFiles.yaml
      ├── cudaImage.dp.cpp
      ├── cudaImage.h
      ├── QuickSilver.h
      ├── QuickSilver.h.yaml
      ├── QuickSilverD.dp.cpp
      ├── QuickSilverD.h
      ├── QuickSilverH.dp.cpp
      ├── QuickSilverH.h
      ├── cudautils.h
      ├── cudautils.h.yaml
      ├── geomFuncs.cpp
      ├── mainSift.cpp.dp.cpp
      └── matching.dp.cpp

   ```
### 4 Review the migrated source code and fix all `DPCT` warnings

SYCLomatic and [Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html) define a list of `DPCT` warnings and embed the warning in migrated source code if need manual effort to check. All the warnings in the migrated code should be reviewed and fixed. For details of `DPCT` warnings and corresponding fix examples, refer to [Intel® DPC++ Compatibility Tool Developer Guide and Reference](https://www.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/diagnostics-reference.html) or [SYCLomatic doc page](https://oneapi-src.github.io/SYCLomatic/dev_guide/diagnostics-reference.html). 

c++17 is required to build SYCL code, so you need to update -std option as follows:
```
$ grep "c++11" out/Makefile.dpct 
TARGET_0_FLAG_0 = -DHAVE_CUDA -DHAVE_UVM=1 -std=c++11 -O3 ${FLAGS}
TARGET_0_FLAG_1 = -DHAVE_CUDA -DHAVE_UVM=1 -std=c++11 -O3 ${FLAGS}
TARGET_0_FLAG_2 = -DHAVE_CUDA -DHAVE_UVM=1 -std=c++11 -O3 ${FLAGS}
TARGET_0_FLAG_7 = -DHAVE_CUDA -DHAVE_UVM=1 -std=c++11 -O3 ${FLAGS}
TARGET_0_FLAG_8 = -DHAVE_CUDA -DHAVE_UVM=1 -std=c++11 -O3 ${FLAGS}
TARGET_0_FLAG_11 = -DHAVE_CUDA -DHAVE_UVM=1 -std=c++11 -O3 ${FLAGS}
TARGET_0_FLAG_12 = -DHAVE_CUDA -DHAVE_UVM=1 -std=c++11 -O3 ${FLAGS}
TARGET_0_FLAG_14 = -DHAVE_CUDA -DHAVE_UVM=1 -std=c++11 -O3 ${FLAGS}
.................
```
```
$ sed -i "s/c++11/c++17/g" ./out/Makefile.dpct
```

### 5 Build the migrated QuickSilver
```
$ cd ${QuickSilver_HOME}/CUDA/out
$ make -f Makefile.dpct
```
### 6 Run migrated SYCL version QuickSilver
```
  $ ./qs -i ../../../Examples/AllScattering/scatteringOnly.inp
Copyright (c) 2016
Lawrence Livermore National Security, LLC
All Rights Reserved
Quicksilver Version     : 2023-Mar-22-18:05:07
Quicksilver Git Hash    : 5b22f83364f62d6d8ec383402793a2ed38a25276
MPI Version             : 3.0
Number of MPI ranks     : 1
Number of OpenMP Threads: 1
Number of OpenMP CPUs   : 1

Simulation:
   dt: 1e-08
   fMax: 0.1
   inputFile: ../../../Examples/AllScattering/scatteringOnly.inp
   energySpectrum: 
   boundaryCondition: octant
   loadBalance: 1
   cycleTimers: 0
   debugThreads: 0
   lx: 100
   ly: 100
   lz: 100
   nParticles: 10000000
   batchSize: 0
   nBatches: 10
   nSteps: 10
   nx: 10
   ny: 10
   nz: 10
   seed: 1029384756
   xDom: 0
   yDom: 0
   zDom: 0
   eMax: 20
   eMin: 1e-09
   nGroups: 230
   lowWeightCutoff: 0.001
   bTally: 1
   fTally: 1
   cTally: 1
   coralBenchmark: 0
   crossSectionsOut:

Geometry:
   material: sourceMaterial
   shape: brick
   xMax: 100
   xMin: 0
   yMax: 100
   yMin: 0
   zMax: 100
   zMin: 0

Material:
   name: sourceMaterial
   mass: 1000
   nIsotopes: 10
   nReactions: 9
   sourceRate: 1e+10
   totalCrossSection: 0.1
   absorptionCrossSection: flat
   fissionCrossSection: flat
   scatteringCrossSection: flat
   absorptionCrossSectionRatio: 0
   fissionCrossSectionRatio: 0
   scatteringCrossSectionRatio: 1

CrossSection:
   name: flat
   A: 0
   B: 0
   C: 0
   D: 0
   E: 1
   nuBar: 2.4

WARNING: The environment variable SYCL_DEVICE_FILTER is deprecated. Please use ONEAPI_DEVICE_SELECTOR instead.
For more details, please refer to:
https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#oneapi_device_selector

Building partition 0
Building partition 1
Building partition 2
Building partition 3
Building MC_Domain 0
Building MC_Domain 1
Building MC_Domain 2
Building MC_Domain 3
Starting Consistency Check
Finished Consistency Check
Finished initMesh
   cycle  start  source     rr  split     absorb    scatter fission    produce    collisn   escape     census    num_seg   scalar_flux      cycleInit  cycleTracking  cycleFinalize
       0      0  999000      0 9000911          0   18533102       0          0   18533102  1151713    8848198   55527552  1.853054e+09   3.941620e-01   4.599402e+00   1.000000e-06
       1 8848198  999000      0 152496          0   34283265       0          0   34283265  1664199    8335495   94636539  5.042460e+09   2.619460e-01   6.905509e+00   1.000000e-06
       2 8335495  999000      0 664828          0   34355643       0          0   34355643  1366796    8632527   95014421  7.698305e+09   2.519900e-01   7.205290e+00   2.000000e-06
       3 8632527  999000      0 368918          0   34304499       0          0   34304499  1242218    8758227   94957812  9.981711e+09   2.782180e-01   7.377506e+00   1.000000e-06
       4 8758227  999000      0 243133          0   34142788       0          0   34142788  1168494    8831866   94602592  1.198623e+10   2.527620e-01   7.409457e+00   1.000000e-06
       5 8831866  999000      0 169147          0   33951332       0          0   33951332  1121053    8878960   94154708  1.376283e+10   2.558220e-01   7.424416e+00   1.000000e-06
       6 8878960  999000      0 121378          0   33761535       0          0   33761535  1088931    8910407   93691628  1.534119e+10   2.562770e-01   7.463845e+00   1.000000e-06
       7 8910407  999000      0  90638          0   33552845       0          0   33552845  1065102    8934943   93219723  1.675345e+10   2.657750e-01   7.471188e+00   1.000000e-06
       8 8934943  999000      0  66353          0   33384900       0          0   33384900  1047516    8952780   92770771  1.802717e+10   2.571480e-01   7.569291e+00   2.000000e-06
       9 8952780  999000      0  48021          0   33199785       0          0   33199785  1033858    8965943   92326431  1.918272e+10   2.796770e-01   7.676362e+00   1.000000e-06

Timer                       Cumulative   Cumulative   Cumulative   Cumulative   Cumulative   Cumulative
Name                            number    microSecs    microSecs    microSecs    microSecs   Efficiency
                              of calls          min          avg          max       stddev       Rating
main                                 1    7.386e+07    7.386e+07    7.386e+07    0.000e+00       100.00
cycleInit                           10    2.754e+06    2.754e+06    2.754e+06    0.000e+00       100.00
cycleTracking                       10    7.110e+07    7.110e+07    7.110e+07    0.000e+00       100.00
cycleTracking_Kernel               107    7.093e+07    7.093e+07    7.093e+07    0.000e+00       100.00
cycleTracking_MPI                  117    1.751e+05    1.751e+05    1.751e+05    0.000e+00       100.00
cycleTracking_Test_Done              0    0.000e+00    0.000e+00    0.000e+00    0.000e+00         0.00
cycleFinalize                       20    3.690e+02    3.690e+02    3.690e+02    0.000e+00       100.00
Figure Of Merit                  12.67 [Num Mega Segments / Cycle Tracking Time]
```
**Note:** 
* The testing result was collected when run on Intel(R) Core(TM) i7-13700K CPU backend with Intel® oneAPI Base Toolkit(2023.2 version).
* The Reference migrated code is attached in **migrated** folder.

If an error occurs during runtime, refer to [Diagnostics Utility for Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## QuickSilver License
[License.txt](https://github.com/oneapi-src/Velocity-Bench/blob/main/QuickSilver/LICENSE.md)

## Reference 
* Command Line Options of [SYCLomatic](https://oneapi-src.github.io/SYCLomatic/dev_guide/command-line-options-reference.html) or [Intel® DPC++ Compatibility Tool](https://software.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/command-line-options-reference.html)
* [oneAPI GPU Optimization Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/)
* [SYCLomatic project](https://github.com/oneapi-src/SYCLomatic/)


## Trademarks information
Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.
\*Other names and brands may be claimed as the property of others. SYCL is a trademark of the Khronos Group Inc.
