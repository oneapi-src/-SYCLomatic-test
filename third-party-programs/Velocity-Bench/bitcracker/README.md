# Migration example: Migrate bitcracker to SYCL version
[SYCLomatic](https://github.com/oneapi-src/SYCLomatic) is a project to assist developers in migrating their existing code written in different programming languages to the SYCL* C++ heterogeneous programming model. It is an open source version of the Intel® DPC++ Compatibility Tool.

This file lists the detailed steps to migrate CUDA version of [bitcracker](https://github.com/oneapi-src/Velocity-Bench/tree/main/bitcracker) to SYCL version with SYCLomatic. As follow table summarizes the migration environment, the software required, and so on.

   | Optimized for         | Description
   |:---                   |:---
   | OS                    | Linux* Ubuntu* 22.04
   | Software              | Intel® oneAPI Base Toolkit, SYCLomatic
   | What you will learn   | Migration of CUDA code, Run SYCL code on oneAPI and Intel device
   | Time to complete      | 15 minutes


## Migrating bitcracker to SYCL

### 1 Prepare the migration
#### 1.1 Get the source code of bitcracker and install the dependency library
```sh
   $ git clone https://github.com/oneapi-src/Velocity-Bench.git
   $ export bitcracker_HOME=/path/to/Velocity-Bench/bitcracker
   $ cd ${bitcracker_HOME}/CUDA && mkdir build
   $ cd build && cmake ..   # Make sure all dependency libraries are installed.
```
Summary of bitcracker project source code:
```
   CUDA/
   ├── CMakeLists.txt
   └── src
      ├── aes.h
      ├── attack.cu
      ├── bitcracker.h
      ├── main.cu
      ├── sha256.h
      ├── utils.cu
      └── w_blocks.cu
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
$ cd ${bitcracker_HOME}/CUDA/build
$ make clean
$ intercept-build make
$ ls compile_commands.json  # make sure compile_commands.json is generated
compile_commands.json
```
### 3 Migrate the source code and build script
```sh
# From the CUDA directory as root directory:
$ cd ${bitcracker_HOME}/CUDA
$ dpct --in-root=. -p=./build/compile_commands.json --out-root=out --gen-build-script --cuda-include-path=/usr/local/cuda/include
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
   ├── Makefile.dpct
   └── src
      ├── aes.h
      ├── aes.h.yaml
      ├── attack.dp.cpp
      ├── bitcracker.h
      ├── bitcracker.h.yaml
      ├── main.dp.cpp
      ├── sha256.h
      ├── sha256.h.yaml
      ├── utils.dp.cpp
      └── w_blocks.dp.cpp
```
### 4 Review the migrated source code and fix all `DPCT` warnings

SYCLomatic and [Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html) define a list of `DPCT` warnings and embed the warning in migrated source code if need manual effort to check. All the warnings in the migrated code should be reviewed and fixed. For detail of `DPCT` warnings and corresponding fix examples, refer to [Intel® DPC++ Compatibility Tool Developer Guide and Reference](https://www.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/diagnostics-reference.html) or [SYCLomatic doc page](https://oneapi-src.github.io/SYCLomatic/dev_guide/diagnostics-reference.html).

Fix the warning in the migrated bitcracker code:
```
$ cat ${bitcracker_HOME}/CUDA/out/src/attack.dp.cpp
...
/*
DPCT1110:3: The total declared local variable size in device function
decrypt_vmk_with_mac exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/

void decrypt_vmk_with_mac(
...
```
This message is shown because the Compatibility Tool finding the user-declared private memory size of the local variable in the kernel may exceed 128 bytes, which is the largest register size for each work-item. It may cause high register pressure. For more details, you can refer [oneAPI GPU Optimization Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-0/overview.html)

### 5 Build the migrated bitcracker
```
$ cd ${bitcracker_HOME}/CUDA/out
$ make -f Makefile.dpct
```
### 6 Run migrated SYCL version bitcracker
```
$: ./bitcracker -f ../../hash_pass/img_win8_user_hash.txt -d ../../hash_pass/user_passwords_60000.txt -b 60000
---------> BitCracker: BitLocker password cracking tool <---------


==================================
Retrieving Info
==================================

Reading hash file "../../hash_pass/img_win8_user_hash.txt"
================================================
                  Attack
================================================
Type of attack: User Password
Psw per thread: 1
max_num_pswd_per_read: 60000
Dictionary: ../../hash_pass/user_passwords_60000.txt
MAC Comparison (-m): Yes


Iter: 1, num passwords read: 60000
Kernel execution:
        Effective passwords: 60000
        Passwords Range:
                npknpByH7N2m3OnLNH1X9DJxLrzIFWk
                .....
                dL_7uuf3QCz-c6K3xDu0
--------------------
================================================
Bitcracker attack completed
Total passwords evaluated: 60000
Password not found!
================================================
time to subtract from total: 0.0148924 s
bitcracker - total time for whole calculation: 452.283 s
```
**Note:**
* The testing result was collected run on Intel(R) Core(TM) i7-13700K CPU backend with Intel® oneAPI Base Toolkit(2023.2 version).
* The Reference migrated code is attached in **migrated** folder.

If an error occurs during runtime, refer to [Diagnostics Utility for Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## bitcracker License
[License.txt](https://github.com/oneapi-src/Velocity-Bench/blob/main/bitcracker/LICENSE.md)

## Reference
* Command Line Options of [SYCLomatic](https://oneapi-src.github.io/SYCLomatic/dev_guide/command-line-options-reference.html) or [Intel® DPC++ Compatibility Tool](https://software.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/command-line-options-reference.html)
* [oneAPI GPU Optimization Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/)
* [SYCLomatic project](https://github.com/oneapi-src/SYCLomatic/)


## Trademarks information
Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.
\*Other names and brands may be claimed as the property of others. SYCL is a trademark of the Khronos Group Inc.
