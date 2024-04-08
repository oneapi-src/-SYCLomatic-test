# Migration example: Migrate hplinpack to SYCL version
[SYCLomatic](https://github.com/oneapi-src/SYCLomatic) is a project to assist developers in migrating their existing code written in different programming languages to the SYCL* C++ heterogeneous programming model. It is an open source version of Intel® DPC++ Compatibility Tool.

This file lists the detailed steps to migrate CUDA version of [hplinpack](https://github.com/oneapi-src/Velocity-Bench/tree/main/hplinpack) to SYCL version with SYCLomatic. As follow table summarizes the migration environment, the software required, and so on.

   | Optimized for         | Description
   |:---                   |:---
   | OS                    | Linux* Ubuntu* 22.04
   | Software              | Intel® oneAPI Base Toolkit, SYCLomatic
   | What you will learn   | Migration of CUDA code, Run SYCL code on oneAPI and Intel device
   | Time to complete      | 15 minutes


## Migrating hplinpack to SYCL

### 1 Prepare the migration
#### 1.1 Get the source code of hplinpack and install the dependency library
```sh
   $ git clone https://github.com/oneapi-src/Velocity-Bench.git
   $ export hplinpack_HOME=/path/to/Velocity-Bench/hplinpack
```
#### 1.2 Prepare migration tool and SYCL run environment

 * Install SYCL run environment [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html). After installation, the Intel® DPC++ Compatibility tool is also available, set up the SYCL run environment as follows:

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
$ cd ${hplinpack_HOME}/cuda/hpl-2.3
$ make clean
$ intercept-build make
$ ls compile_commands.json  # make sure compile_commands.json is generated
compile_commands.json
```
### 3 Migrate the source code and build script
```sh
# From the CUDA directory as root directory:
$ cd ${hplinpack_HOME}/cuda
$ dpct --in-root=. -p=./hpl-2.3/compile_commands.json --out-root=out --gen-build-script --cuda-include-path=/usr/local/cuda/include
```
Description of the options: 
 * `--in-root`: provide input files to specify where to locate the CUDA files that need migration.
 * `-p`: specify compilation database to migrate the whole project.
 * `--out-root`: designate where to generate the resulting files (default is `dpct_output`).
 * `--gen-build-script`: generate the `Makefile.dpct` for the migrated code.

Now you can see the migrated files in the `out` folder.
### 4 Review the migrated source code and fix all `DPCT` warnings

SYCLomatic and [Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html) define a list of `DPCT` warnings and embed the warning in migrated source code if need manual effort to check. All the warnings in the migrated code should be reviewed and fixed. For detail of `DPCT` warnings and corresponding fix examples, refer to [Intel® DPC++ Compatibility Tool Developer Guide and Reference](https://www.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/diagnostics-reference.html) or [SYCLomatic doc page](https://oneapi-src.github.io/SYCLomatic/dev_guide/diagnostics-reference.html). 

Fix the warning in the migrated hplinpack code:
```
$ cat ${hplinpack_HOME}/CUDA/out/Makefile.dpct
...
1 CC := icpx
2 
3 LD := $(CC)
4 
5 #DPCT2001:4: You can link with more library by add them here.
6 LIB :=
7
8 FLAGS :=
9
......
582 TARGET :=  ${TARGET_0} ${TARGET_1} ${TARGET_2}
......
589 $(TARGET_0): $(OBJS_0)
590         $(CC) -fsycl -o $@ $^ $(LIB) -qmkl
......
628 $(TARGET_1): $(OBJS_1)
629         ar -r $@ $^ $(LIB) -qmkl
......
1009 $(TARGET_2): $(OBJS_2)
1010         $(CC) -fsycl -o $@ $^ $(LIB) -qmkl
1011
1012 $(TARGET_2_OBJ_0):$(TARGET_2_SRC_0)
1013         $(CC) -c ${TARGET_2_SRC_0} -o ${TARGET_2_OBJ_0} $(TARGET_2_FLAG_0)
1014
1015 $(TARGET_2_OBJ_1):$(TARGET_2_SRC_1)
1016         $(CC) -c ${TARGET_2_SRC_1} -o ${TARGET_2_OBJ_1} $(TARGET_2_FLAG_1)
```
 **hplinpack** needs to link the **mkl** libraries, **libdgemm.so.1.0.1** and **../lib/intel64/libhpl.a** in link time. And need to update the CC compiler from **icpx** to **mpicc**. Add **-lmpi**, **-fPIC** for LIB and FLAGS so fix the LIB variable as follows:
```
1 CC := mpicc
2 
3 LD := $(CC)
4 
5 #DPCT2001:4: You can link with more library by add them here.
6 LIB := -lmpi
7
8 FLAGS := -fPIC
9
......
582 TARGET :=   ${TARGET_1} ${TARGET_2} ${TARGET_0}
......
589 $(TARGET_0): $(OBJS_0)
590         $(CC)  -o $@ $^ $(LIB) -qmkl libdgemm.so.1.0.1 ../lib/intel64/libhpl.a
627
628 $(TARGET_1): $(OBJS_1)
629         ar -r $@ $^ $(LIB)
630
1008
1009 $(TARGET_2): $(OBJS_2)
1010         $(CC) -fPIC -shared -fsycl -o $@ $^ $(LIB) -qmkl
1011
1012 $(TARGET_2_OBJ_0):$(TARGET_2_SRC_0)
1013         $(CC) -c ${TARGET_2_SRC_0} -o ${TARGET_2_OBJ_0} $(TARGET_2_FLAG_0)
1014
1015 $(TARGET_2_OBJ_1):$(TARGET_2_SRC_1)
1016         icpx -c -fsycl  ${TARGET_2_SRC_1} -o ${TARGET_2_OBJ_1} $(TARGET_2_FLAG_1)
1017
```
### 5 Build the migrated hplinpack
```
$ cd ${hplinpack_HOME}/cuda/out
$ make -f Makefile.dpct
```
### 6 Run migrated SYCL version hplinpack
```
   $ cd ${hplinpack_HOME}/CUDA/out
   $ ./xhpl 
   ================================================================================
HPLinpack 2.3  --  High-Performance Linpack benchmark  --   December 2, 2018
Written by A. Petitet and R. Clint Whaley,  Innovative Computing Laboratory, UTK
Modified by Piotr Luszczek, Innovative Computing Laboratory, UTK
Modified by Julien Langou, University of Colorado Denver
================================================================================

An explanation of the input/output parameters follows:
T/V    : Wall time / encoded variant.
N      : The order of the coefficient matrix A.
NB     : The partitioning blocking factor.
P      : The number of process rows.
Q      : The number of process columns.
Time   : Time in seconds to solve the linear system.
Gflops : Rate of execution for solving the linear system.

The following parameter values will be used:

N      :   24576    24576
NB     :    2048
PMAP   : Row-major process mapping
P      :       1
Q      :       1
PFACT  :    Left
NBMIN  :       2
NDIV   :       2
RFACT  :    Left
BCAST  :   1ring
DEPTH  :       1
SWAP   : Spread-roll (long)
L1     : no-transposed form
U      : no-transposed form
EQUIL  : yes
ALIGN  : 8 double precision words

--------------------------------------------------------------------------------

- The matrix A is randomly generated for each test.
- The following scaled residual check will be computed:
      ||Ax-b||_oo / ( eps * ( || x ||_oo * || A ||_oo + || b ||_oo ) * N )
- The relative machine precision (eps) is taken to be               1.110223e-16
- Computational tests pass if scaled residuals are less than                16.0

================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       24576  2048     1     1             228.26             4.3357e+01
HPL_pdgesv() start time Fri Feb  2 14:50:51 2024

HPL_pdgesv() end time   Fri Feb  2 14:54:39 2024

--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=   4.05852126e+08 ...... FAILED
||Ax-b||_oo  . . . . . . . . . . . . . . . . . =        5455.998041
||A||_oo . . . . . . . . . . . . . . . . . . . =        6229.846527
||A||_1  . . . . . . . . . . . . . . . . . . . =        6230.505776
||x||_oo . . . . . . . . . . . . . . . . . . . =         790.874477
||x||_1  . . . . . . . . . . . . . . . . . . . =      203978.961025
||b||_oo . . . . . . . . . . . . . . . . . . . =           0.499978
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       24576  2048     1     1             222.76             4.4426e+01
HPL_pdgesv() start time Fri Feb  2 14:54:57 2024

HPL_pdgesv() end time   Fri Feb  2 14:58:40 2024

--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=   4.05852126e+08 ...... FAILED
||Ax-b||_oo  . . . . . . . . . . . . . . . . . =        5455.998041
||A||_oo . . . . . . . . . . . . . . . . . . . =        6229.846527
||A||_1  . . . . . . . . . . . . . . . . . . . =        6230.505776
||x||_oo . . . . . . . . . . . . . . . . . . . =         790.874477
||x||_1  . . . . . . . . . . . . . . . . . . . =      203978.961025
||b||_oo . . . . . . . . . . . . . . . . . . . =           0.499978
================================================================================

Finished      2 tests with the following results:
              0 tests completed and passed residual checks,
              2 tests completed and failed residual checks,
              0 tests skipped because of illegal input values.
--------------------------------------------------------------------------------

End of Tests.
```
**Note:** 
* The testing result was collected when run on Intel(R) Core(TM) i7-13700K CPU backend with Intel® oneAPI Base Toolkit(2023.2 version).
* The Reference migrated code is attached in **migrated** folder.

If an error occurs during runtime, refer to [Diagnostics Utility for Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## hplinpack License
[License.txt](https://github.com/oneapi-src/Velocity-Bench/blob/main/hplinpack/LICENSE.md)

## Reference 
* Command Line Options of [SYCLomatic](https://oneapi-src.github.io/SYCLomatic/dev_guide/command-line-options-reference.html) or [Intel® DPC++ Compatibility Tool](https://software.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/command-line-options-reference.html)
* [oneAPI GPU Optimization Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/)
* [SYCLomatic project](https://github.com/oneapi-src/SYCLomatic/)


## Trademarks information
Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.
\*Other names and brands may be claimed as the property of others. SYCL is a trademark of the Khronos Group Inc.
