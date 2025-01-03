#define PRINT_PASS 1

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Vulkan headers
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <Windows.h>
#include <vulkan/vulkan_win32.h>
#else
#include <vulkan/vulkan_core.h>
#endif // _WIN32

#define CHECK_VK_ERROR(call, errMsg) \
    do { \
        VkResult vk_status = call; \
        if (vk_status != VK_SUCCESS) { \
            std::cout << "[ERROR] " << errMsg << std::endl; \
        } \
    } while(0)

int passed = 0;
int failed = 0;

void checkResult(std::string name, bool IsPassed) {
  std::cout << name;
  if (IsPassed) {
    std::cout << " ---- passed" << std::endl;
    passed++;
  } else {
    std::cout << " ---- failed" << std::endl;
    failed++;
  }
}

VkInstance createInstance() {
    VkInstance instance;

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Interop";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    const char* instanceExtensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
    #ifdef _WIN32
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
    #else
        VK_KHR_DISPLAY_EXTENSION_NAME,
    #endif // _WIN32
    };

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = sizeof(instanceExtensions) / sizeof(instanceExtensions[0]);
    createInfo.ppEnabledExtensionNames = instanceExtensions;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance!");
    }

    return instance;
}

VkPhysicalDevice pickPhysicalDevice(VkInstance instance) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Pick the first suitable device
    return devices[0];
}

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice,
                                    const std::vector<VkFormat> &candidates,
                                    VkImageTiling tiling,
                                    VkFormatFeatureFlags features) {
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }

  throw std::runtime_error("Failed to find supported format!");
}

VkDevice create_vk_dev(VkPhysicalDevice physicalDevice) {
    VkDevice device;

    // Device Queue creation
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = 0;  // Assuming queue family 0 supports graphics
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    const char* deviceExtensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#ifdef _WIN32
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
#endif // _WIN32
    };

    // Logical Device creation
    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.enabledExtensionCount = sizeof(deviceExtensions) / sizeof(deviceExtensions[0]);
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = NULL;
    deviceCreateInfo.pEnabledFeatures = NULL;

    CHECK_VK_ERROR(
        vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device),
        "Unable to create Vulkan device"
    );

    return device;
}

VkImage create_vk_tex(VkDevice device, VkPhysicalDevice physicalDevice, VkMemoryRequirements &memRequirements, VkDeviceMemory &imageMemory,
                        int width, int height, VkFormat format = VK_FORMAT_R32_SFLOAT) {
    VkImage image;

    // Define the image properties
    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.extent.width = width;
    imageCreateInfo.extent.height = height;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.format = format;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    CHECK_VK_ERROR(
    vkCreateImage(device, &imageCreateInfo, nullptr, &image),
        "Unable to create Vulkan image"
    );

    // Allocate memory for the image
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    // Check for suitable memory type
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            memoryTypeIndex = i;
            break;
        }
    }

    if (memoryTypeIndex == UINT32_MAX) {
        throw std::runtime_error("Failed to find suitable memory type!");
    }

    VkExportMemoryAllocateInfo exportAllocInfo = {};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
#ifdef _WIN32
    exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif // _WIN32

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, physicalDevice);
    allocInfo.pNext = &exportAllocInfo;

    CHECK_VK_ERROR(
        vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory),
        "Unable to allocate Vulkan image memory"
    );

    vkBindImageMemory(device, image, imageMemory, 0);

    return image;
}

#ifdef _WIN32
HANDLE create_shared_nt_handle(VkDevice device, VkDeviceMemory imageMemory) {
    VkMemoryGetWin32HandleInfoKHR handleInfo = {};
    handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    handleInfo.memory = imageMemory;  // Assuming image's memory is allocated and binded

    HANDLE win32Handle;
    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR =
        (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");

    if (vkGetMemoryWin32HandleKHR == NULL) {
        throw std::runtime_error("Failed to get vkGetMemoryWin32HandleKHR function pointer!");
    }

    CHECK_VK_ERROR(
        vkGetMemoryWin32HandleKHR(device, &handleInfo, &win32Handle),
        "Unable to get Win32 handle"
    );

    return win32Handle;
}
#else
int create_shared_fd(VkDevice device, VkDeviceMemory imageMemory) {
    VkMemoryGetFdInfoKHR handleInfo = {};
    handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    handleInfo.memory = imageMemory;  // Assuming image's memory is allocated and binded

    int fd;
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR =
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");

    if (vkGetMemoryFdKHR == NULL) {
        throw std::runtime_error("Failed to get vkGetMemoryFdKHR function pointer!");
    }

    CHECK_VK_ERROR(
        vkGetMemoryFdKHR(device, &handleInfo, &fd),
        "Unable to get file descriptor"
    );

    return fd;
}
#endif // _WIN32

void cleanupInterop(cudaExternalMemory_t externalMemory) {
    // Destroy the CUDA resource
    cudaDestroyExternalMemory(externalMemory);
}

template <typename T>
int run_test(T* input, T* output, cudaExternalMemoryHandleDesc memHandleDesc, cudaExternalMemoryMipmappedArrayDesc mipmapDesc, int w, int h) {
    cudaExternalMemory_t externalMemory;

    cudaError_t cudaStatus;
    cudaStatus = cudaImportExternalMemory(&externalMemory, &memHandleDesc);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Cannot import external memory: " << cudaGetErrorString(cudaStatus) << std::endl;
        return EXIT_FAILURE;
    }

    cudaMipmappedArray_t cudaMipmappedArray = nullptr;
    cudaStatus = cudaExternalMemoryGetMappedMipmappedArray(&cudaMipmappedArray, externalMemory, &mipmapDesc);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Cannot acquire texture data as a CUDA mipmapped array: " << cudaGetErrorString(cudaStatus) << std::endl;
        cleanupInterop(externalMemory);
        return EXIT_FAILURE;
    }

    // Retrieve the tex data as a cudaArray from cudaMipmappedArray
    cudaArray_t cudaArr;
    cudaStatus = cudaGetMipmappedArrayLevel(&cudaArr, cudaMipmappedArray, 0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Cannot create CUDA Array from CUDA mipmapped array: " << cudaGetErrorString(cudaStatus) << std::endl;
        cleanupInterop(externalMemory);
        return EXIT_FAILURE;
    }

    cudaStream_t stream;
    cudaStatus = cudaStreamCreate(&stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(cudaStatus) << std::endl;
        cleanupInterop(externalMemory);
        return EXIT_FAILURE;
    }

    // Access the underlying memory of interop CUDA resource
    cudaStatus = cudaMemcpy2DToArrayAsync(cudaArr, 0, 0, input, sizeof(T) * w,
                                        sizeof(T) * w, h, cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to copy data to array: " << cudaGetErrorString(cudaStatus) << std::endl;
        cleanupInterop(externalMemory);
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    // Debug: Verify data transfer
    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to synchronize stream after memcpy: " << cudaGetErrorString(cudaStatus) << std::endl;
        cleanupInterop(externalMemory);
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    // Access the underlying memory of interop CUDA resource
    cudaStatus = cudaMemcpy2DFromArrayAsync(output, sizeof(T) * w, cudaArr, 0, 0,
                                            sizeof(T) * w, h, cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to copy data to array: " << cudaGetErrorString(cudaStatus) << std::endl;
        cleanupInterop(externalMemory);
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to synchronize stream after kernel: " << cudaGetErrorString(cudaStatus) << std::endl;
        cleanupInterop(externalMemory);
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    cleanupInterop(externalMemory);
    cudaStreamDestroy(stream);

    return EXIT_SUCCESS;
}

int main() {
    // Init VK env
    VkInstance instance = createInstance();
    VkPhysicalDevice physicalDevice = pickPhysicalDevice(instance);
    
    // Initialize Vulkan
    VkDevice device = create_vk_dev(physicalDevice);

    // VK_FORMAT_R32_SFLOAT test
    {
        // Init test data
        const int w = 4;
        const int h = 4;
        const int channels = 1;

        float input[h * w * channels];
        for (int i = 0; i < h * w * channels; i++) {
            input[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        float *output;
        cudaMallocManaged(&output, sizeof(input));

        // Create a texture for DirectX and CUDA interoperability
        VkMemoryRequirements memRequirements;
        VkDeviceMemory imageMemory;
        VkImage vkTexture = create_vk_tex(device, physicalDevice, memRequirements, imageMemory, w, h);

        // Map the external memory object to a CUDA device pointer
        cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {};
        mipmapDesc.numLevels = 1;
        mipmapDesc.extent = make_cudaExtent(w, h, 1);
        mipmapDesc.formatDesc = cudaCreateChannelDesc<float>();

        // Import the Vulkan memory into CUDA
        cudaExternalMemoryHandleDesc memHandleDesc = {};
#ifdef _WIN32
        memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        memHandleDesc.handle.win32.handle = create_shared_nt_handle(device, imageMemory);
#else
        memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memHandleDesc.handle.fd = create_shared_fd(device, imageMemory);
#endif // _WIN32
        memHandleDesc.size = memRequirements.size;

        if (run_test(input, output, memHandleDesc, mipmapDesc, w * channels, h) != EXIT_SUCCESS) {
            std::cerr << "Test failed" << std::endl;
            return EXIT_FAILURE;
        }

        // Verify output data
        bool pass = true;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                for (int c = 0; c < channels; c++) {
                    int idx = (i * w + j) * channels + c;
                    if (output[idx] != input[idx]) {
                        pass = false;
                        break;
                    }
                }

                if (!pass)
                    break;
            }

            if (!pass)
                break;
        }

        checkResult("VK_FORMAT_R32_SFLOAT", pass);
        if (PRINT_PASS || !pass) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    for (int c = 0; c < channels; c++) {
                        int idx = (i * w + j) * channels + c;
                        if (output[idx] != input[idx]) {
                            std::cout << "Failed: output[" << i << "][" << j << "][" << c << "] = "
                                    << output[idx] << ", expected: " << input[idx] << std::endl;
                        }
                    }
                }
            }
        }

        // cudaDestroyTextureObject(tex);
        cudaFree(output);
        vkDestroyImage(device, vkTexture, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
    }

    // VK_FORMAT_R16G16_SFLOAT test
    {
        // Init test data
        const int w = 4;
        const int h = 4;
        const int channels = 2;  // RGB components

        half input[h * w * channels];
        for (int i = 0; i < h * w * channels; i++) {
            input[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
        }

        half *output;
        cudaMallocManaged(&output, sizeof(input));

        // Create a texture for Vulkan and CUDA interoperability
        VkMemoryRequirements memRequirements;
        VkDeviceMemory imageMemory;
        VkImage vkTexture = create_vk_tex(device, physicalDevice, memRequirements, imageMemory,
                                          w, h, VK_FORMAT_R16G16_SFLOAT);

        // Map the external memory object to a CUDA device pointer
        cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {};
        mipmapDesc.numLevels = 1;
        mipmapDesc.extent = make_cudaExtent(w, h, 1);
        mipmapDesc.formatDesc = cudaCreateChannelDescHalf2();

        // Import the Vulkan memory into CUDA
        cudaExternalMemoryHandleDesc memHandleDesc = {};
#ifdef _WIN32
        memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        memHandleDesc.handle.win32.handle = create_shared_nt_handle(device, imageMemory);
#else
        memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memHandleDesc.handle.fd = create_shared_fd(device, imageMemory);
#endif // _WIN32
        memHandleDesc.size = memRequirements.size;

        if (run_test(input, output, memHandleDesc, mipmapDesc, w * channels, h) != EXIT_SUCCESS) {
            std::cerr << "Test failed" << std::endl;
            return EXIT_FAILURE;
        }

        // Verify output data
        bool pass = true;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                for (int c = 0; c < channels; c++) {
                    int idx = (i * w + j) * channels + c;
                    if (output[idx] != input[idx]) {
                        pass = false;
                        break;
                    }
                }

                if (!pass)
                    break;
            }

            if (!pass)
                break;
        }

        checkResult("VK_FORMAT_R16G16_SFLOAT", pass);
        if (PRINT_PASS || !pass) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    for (int c = 0; c < channels; c++) {
                        int idx = (i * w + j) * channels + c;
                        if (output[idx] != input[idx]) {
                            std::cout << "Failed: output[" << i << "][" << j << "][" << c << "] = " 
                                      << (float)output[idx] << ", expected: " << (float)input[idx] << std::endl;
                        }
                    }
                }
            }
        }

        cudaFree(output);
        vkDestroyImage(device, vkTexture, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
    }

    // VK_FORMAT_R16G16B16A16_SFLOAT test
    {
        // Init test data
        const int w = 4;
        const int h = 4;
        const int channels = 4;  // RGB components

        half input[h * w * channels];
        for (int i = 0; i < h * w * channels; i++) {
            input[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
        }

        half *output;
        cudaMallocManaged(&output, sizeof(input));

        // Create a texture for Vulkan and CUDA interoperability
        VkMemoryRequirements memRequirements;
        VkDeviceMemory imageMemory;
        VkImage vkTexture = create_vk_tex(device, physicalDevice, memRequirements, imageMemory, 
                                          w, h, VK_FORMAT_R16G16B16A16_SFLOAT);

        // Map the external memory object to a CUDA device pointer
        cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {};
        mipmapDesc.numLevels = 1;
        mipmapDesc.extent = make_cudaExtent(w, h, 1);
        mipmapDesc.formatDesc = cudaCreateChannelDescHalf4();

        // Import the Vulkan memory into CUDA
        cudaExternalMemoryHandleDesc memHandleDesc = {};
#ifdef _WIN32
        memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        memHandleDesc.handle.win32.handle = create_shared_nt_handle(device, imageMemory);
#else
        memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memHandleDesc.handle.fd = create_shared_fd(device, imageMemory);
#endif // _WIN32
        memHandleDesc.size = memRequirements.size;

        if (run_test(input, output, memHandleDesc, mipmapDesc, w * channels, h) != EXIT_SUCCESS) {
            std::cerr << "Test failed" << std::endl;
            return EXIT_FAILURE;
        }

        // Verify output data
        bool pass = true;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                for (int c = 0; c < channels; c++) {
                    int idx = (i * w + j) * channels + c;
                    if (output[idx] != input[idx]) {
                        pass = false;
                        break;
                    }
                }

                if (!pass)
                    break;
            }

            if (!pass)
                break;
        }

        checkResult("VK_FORMAT_R16G16B16A16_SFLOAT", pass);
        if (PRINT_PASS || !pass) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    for (int c = 0; c < channels; c++) {
                        int idx = (i * w + j) * channels + c;
                        if (output[idx] != input[idx]) {
                            std::cout << "Failed: output[" << i << "][" << j << "][" << c << "] = " 
                                      << (float)output[idx] << ", expected: " << (float)input[idx] << std::endl;
                        }
                    }
                }
            }
        }

        cudaFree(output);
        vkDestroyImage(device, vkTexture, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
    }

    // VK_FORMAT_R8G8B8A8_UNORM test
    {
        // Init test data
        const int w = 4;
        const int h = 4;
        const int channels = 4;  // RGB components

        unsigned char input[h * w * channels];
        for (int i = 0; i < h * w * channels; i++) {
            input[i] = static_cast<unsigned char>(rand() / 256);
        }

        unsigned char *output;
        cudaMallocManaged(&output, sizeof(input));

        // Create a texture for Vulkan and CUDA interoperability
        VkMemoryRequirements memRequirements;
        VkDeviceMemory imageMemory;
        VkImage vkTexture = create_vk_tex(device, physicalDevice, memRequirements, imageMemory, 
                                          w, h, VK_FORMAT_R8G8B8A8_UNORM);

        // Map the external memory object to a CUDA device pointer
        cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {};
        mipmapDesc.numLevels = 1;
        mipmapDesc.extent = make_cudaExtent(w, h, 1);
        mipmapDesc.formatDesc = cudaCreateChannelDesc<char4>();

        // Import the Vulkan memory into CUDA
        cudaExternalMemoryHandleDesc memHandleDesc = {};
#ifdef _WIN32
        memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        memHandleDesc.handle.win32.handle = create_shared_nt_handle(device, imageMemory);
#else
        memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memHandleDesc.handle.fd = create_shared_fd(device, imageMemory);
#endif // _WIN32
        memHandleDesc.size = memRequirements.size;

        if (run_test(input, output, memHandleDesc, mipmapDesc, w * channels, h) != EXIT_SUCCESS) {
            std::cerr << "Test failed" << std::endl;
            return EXIT_FAILURE;
        }

        // Verify output data
        bool pass = true;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                for (int c = 0; c < channels; c++) {
                    int idx = (i * w + j) * channels + c;
                    if (output[idx] != input[idx]) {
                        pass = false;
                        break;
                    }
                }

                if (!pass)
                    break;
            }

            if (!pass)
                break;
        }

        checkResult("VK_FORMAT_R8G8B8A8_UNORM", pass);
        if (PRINT_PASS || !pass) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    for (int c = 0; c < channels; c++) {
                        int idx = (i * w + j) * channels + c;
                        if (output[idx] != input[idx]) {
                            std::cout << "Failed: output[" << i << "][" << j << "][" << c << "] = " 
                                    << output[idx] << ", expected: " << input[idx] << std::endl;
                        }
                    }
                }
            }
        }

        cudaFree(output);
        vkDestroyImage(device, vkTexture, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
    }

    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    std::cout << "Passed " << passed << "/" << passed + failed << " cases!" << std::endl;
    if (failed) {
        std::cout << "Failed!" << std::endl;
    }

    return failed;
}
