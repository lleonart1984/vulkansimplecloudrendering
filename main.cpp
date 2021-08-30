#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>
#include <vector>

#include "loader.h"

// --- Scene setup

#define RESOLUTION_WIDTH 1264
#define RESOLUTION_HEIGHT 761

#define CAMERA_POSITION (glm::vec3(648.064, -82.473, -63.856) / 600.0f - glm::vec3(-9.984, 73.008, -42.64) / 600.0f)
#define CAMERA_TARGET (glm::vec3(6.021, 100.043, -43.679) / 600.0f - glm::vec3(-9.984, 73.008, -42.64) / 600.0f)
#define CAMERA_UP (glm::normalize(glm::vec3(0.273, 0.962, -0.009)))

#define CLOUD_EXTINCTION_SCALE 1024.0f
#define CLOUD_EXTINCTION_BASE glm::vec3(1.0, 1.0, 1.0)
#define CLOUD_SCATTERING_ALBEDO glm::vec3(1.0, 1.0, 1.0)
#define CLOUG_HG_FACTOR 0.875
#define CLOUD_PATH "C:/Users/mendez/Desktop/clouds/disney_big.xyz"

#define SUNLIGHT_COLOR glm::vec3(2.6, 2.5, 2.3)
#define SUNLIGHT_DIR glm::normalize(glm::vec3(0.5826, 0.7660, 0.2717))
     
#pragma region FIELDS
 
/// <summary>
/// Window instance for all renderings
/// </summary>
GLFWwindow* window;
/// <summary>
/// Gets the resolution width of the rendered image and window
/// </summary>
unsigned int Width;
/// <summary>
/// Gets the resolution height of the rendered image and window
/// </summary>
unsigned int Height;
/// <summary>
/// Vulkan instance used by this renderer
/// </summary>
VkInstance instance;
/// <summary>
/// Vulkan physical device used.
/// </summary>
VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
/// <summary>
/// Index of the Graphics and Compute engine
/// </summary>
int QueueFamilyIndex = -1;
/// <summary>
/// Index of the Presentation engine
/// </summary>
int PresentationFamilyIndex = -1;
/// <summary>
/// Vulkan logical device
/// </summary>
VkDevice device;
/// <summary>
/// Vulkan queue for graphics and computations
/// </summary>
VkQueue graphicsQueue;
/// <summary>
/// Vulkan surface of a window
/// </summary>
VkSurfaceKHR surface;
/// <summary>
/// Vulkan queue for presentation in surfaces
/// </summary>
VkQueue presentQueue;
/// <summary>
/// Vulkan swap chain used for presentation to surface
/// </summary>
VkSwapchainKHR swapChain;
VkFormat PresentationFormat;
VkExtent2D PresentationResolution;
/// <summary>
/// Render targets in swapChain
/// </summary>
std::vector<VkImage> swapChainImages;
/// <summary>
/// Render target views in swapChain
/// </summary>
std::vector<VkImageView> swapChainImageViews;
/// <summary>
/// Frame buffers with the render targets
/// </summary>
std::vector<VkFramebuffer> swapChainFramebuffers;

// Pipeline objects

/// <summary>
/// Vertex Shader module
/// </summary>
VkShaderModule computeShader;
/// <summary>
/// Pipeline layout object (to store rootsignature)
/// </summary>
VkPipelineLayout pipelineLayout;
/// <summary>
/// Render Pass configuration for the renderer
/// </summary>
VkRenderPass renderPass;
/// <summary>
/// Pipeline with the graphics rendering process setup
/// </summary>
VkPipeline graphicsPipeline;

// Commands

/// <summary>
/// Represents the command pool memory for storing command buffers
/// </summary>
VkCommandPool commandPool;
/// <summary>
/// List of command buffers, one for each swapchain image
/// </summary>
std::vector<VkCommandBuffer> commandBuffers;

// Synchronization

VkSemaphore imageAvailableSemaphore;
VkSemaphore renderFinishedSemaphore;

// Resources
VkBuffer uniformBuffer;
VkDeviceMemory uniformBufferMemory;

std::vector<VkBuffer> frameInfoBuffer;
std::vector<VkDeviceMemory> frameInfoBufferMemory;

VkImage textureImage; // Grid
VkExtent3D gridExtent;
VkDeviceMemory textureImageMemory;
VkImageView textureImageView;
VkSampler textureSampler;

VkImage accImage;
VkDeviceMemory accImageMemory;
VkImageView accImageView;

VkImage firstXImage;
VkDeviceMemory firstXImageMemory;
VkImageView firstXImageView;

VkImage firstWImage;
VkDeviceMemory firstWImageMemory;
VkImageView firstWImageView;

VkDescriptorSetLayout descriptorSetLayout;
VkDescriptorPool descriptorPool; // Only one pool of descriptor sets is needed
std::vector<VkDescriptorSet> descriptorSets; // Only a single set of descriptors will be set

#pragma endregion

#pragma region Function Tools

std::vector<char> LoadBytecode(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
}

void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
}

void createImage(uint32_t width, uint32_t height, uint32_t depth, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_3D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = depth;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
}


void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    endSingleTimeCommands(commandBuffer);
}

void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = {
        width,
        height,
        1
    };

    vkCmdCopyBufferToImage(
        commandBuffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    endSingleTimeCommands(commandBuffer);
}

void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t depth) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = {
        width,
        height,
        depth
    };

    vkCmdCopyBufferToImage(
        commandBuffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    endSingleTimeCommands(commandBuffer);
}

#pragma endregion

#pragma region Structs for Buffers

struct UniformBufferObject {
    glm::mat4 proj2world;

    // Cloud properties
    glm::vec3 box_minim; float pad0;
    glm::vec3 box_maxim; float pad1;
    glm::vec3 Extinction; float pad2;
    glm::vec3 ScatteringAlbedo;
    float G;
    glm::vec3 SunDirection; float pad3;
    glm::vec3 SunIntensity; float pad4;
};

struct FrameInfo {
    unsigned int frameCount;
    glm::uvec3 other;
};

#pragma endregion

/// <summary>
/// Initializes GLFW api and creates the window
/// </summary>
void GLFW_Initialization(int width, int height) {
    Width = width;
    Height = height;
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, "Rendering clouds with vulkan", nullptr, nullptr);
}

/// <summary>
/// Initializes vulkan instance and queues
/// </summary>
void Vulkan_Initialization() {
    // Creating Vulkan Instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Rendering clouds with vulkan";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    uint32_t glfwExtensionCount = 0;                // Getting available extensions
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    VkInstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &appInfo;
    instanceCreateInfo.enabledExtensionCount = glfwExtensionCount;
    instanceCreateInfo.ppEnabledExtensionNames = glfwExtensions;

    if (vkCreateInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }

    std::cout << "[INFO] Vulkan instance created..." << std::endl;

    // Creating a surface for a window (IN WINDOWS!)
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    std::cout << "[INFO] Vulkan surface created..." << std::endl;

    // Selecting a physical device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& device : devices) {
        VkPhysicalDeviceProperties deviceProperties;
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
        if(deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physicalDevice = device;
            break;
        }
    }

    // Find queue families
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if ((queueFamily.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
            QueueFamilyIndex = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
        if (presentSupport) {
            PresentationFamilyIndex = i;
        }

        i++;
    }

    if (QueueFamilyIndex == -1 || PresentationFamilyIndex == -1) // can not draw or present
    {
        throw std::runtime_error("failed to access to required engines!");
    }

    float queuePriority = 1.0f;

    // Creating device and queues
    VkDeviceQueueCreateInfo queueCreateInfos[2];
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = QueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos[0] = queueCreateInfo; // queue for drawing
    
    queueCreateInfo.queueFamilyIndex = PresentationFamilyIndex;
    queueCreateInfos[1] = queueCreateInfo; // queue for presenting

    VkPhysicalDeviceFeatures deviceFeatures{
    };

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos;
    deviceCreateInfo.queueCreateInfoCount = 2;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.enabledExtensionCount = deviceExtensions.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    std::cout << "[INFO] Vulkan device created..." << std::endl;
    
    // Retrieve queues to work with
    vkGetDeviceQueue(device, QueueFamilyIndex, 0, &graphicsQueue);
    vkGetDeviceQueue(device, PresentationFamilyIndex, 0, &presentQueue);

    std::cout << "[INFO] Vulkan queues created..." << std::endl;
}

void Presenter_Initialization() {
    
    // Create swap chain
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = 3;
    PresentationFormat = VkFormat::VK_FORMAT_R8G8B8A8_SRGB;
    createInfo.imageFormat = PresentationFormat; 
    createInfo.imageColorSpace = VkColorSpaceKHR::VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    PresentationResolution = { Width, Height };
    createInfo.imageExtent = PresentationResolution;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT;
    
    if (QueueFamilyIndex != PresentationFamilyIndex) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        const unsigned int queueFamilyIndices[2]{ QueueFamilyIndex , PresentationFamilyIndex };
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
    }

    createInfo.preTransform = VkSurfaceTransformFlagBitsKHR::VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VkPresentModeKHR::VK_PRESENT_MODE_MAILBOX_KHR;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    std::cout << "[INFO] Vulkan swapchain created..." << std::endl;

    // Retrieve images from swapchain
    unsigned int imageCount;
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

    // Create image views
    swapChainImageViews.resize(imageCount);
    for (int i = 0; i < imageCount; i++)
    {
        VkImageViewCreateInfo ivcreateInfo{};
        ivcreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivcreateInfo.image = swapChainImages[i];
        ivcreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivcreateInfo.format = PresentationFormat;
        ivcreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ivcreateInfo.subresourceRange.baseMipLevel = 0;
        ivcreateInfo.subresourceRange.levelCount = 1;
        ivcreateInfo.subresourceRange.baseArrayLayer = 0;
        ivcreateInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &ivcreateInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }

    // Create Render Pass
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = PresentationFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    
    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }

    // Create Framebuffers
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        VkImageView attachments[] = {
            swapChainImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = PresentationResolution.width;
        framebufferInfo.height = PresentationResolution.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }

    // Create semaphores for presentation
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS) {
        throw std::runtime_error("failed to create semaphores!");
    }
}

VkShaderModule Create_ShaderModule(std::string path) {
    auto code = LoadBytecode(path);
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}

void Create_GraphicsPipeline() {
    computeShader = Create_ShaderModule("cloud.comp.spv");

    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShader;
    computeShaderStageInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1; // Optional
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout; // Optional
    pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = computeShaderStageInfo;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
}

void Create_CommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = QueueFamilyIndex;
    poolInfo.flags = 0; // Optional
    
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    commandBuffers.resize(swapChainFramebuffers.size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void Create_DescriptorPoolAndLayout() {
    // Creating pool...
    VkDescriptorPoolSize poolSizes[3];
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 2*swapChainImages.size(); // only the uniform so far
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = swapChainImages.size(); // only the uniform so far
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[2].descriptorCount = 4*swapChainImages.size(); // only the uniform so far
      
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = swapChainImages.size();

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }

    // Creating descriptor set layout
    VkDescriptorSetLayoutBinding outputBinding{};
    outputBinding.binding = 0;
    outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    outputBinding.descriptorCount = 1;
    outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    outputBinding.pImmutableSamplers = nullptr; // Optional

    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 2;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding frameLayoutBinding{};
    frameLayoutBinding.binding = 3;
    frameLayoutBinding.descriptorCount = 1;
    frameLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    frameLayoutBinding.pImmutableSamplers = nullptr;
    frameLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding accBinding{};
    accBinding.binding = 4;
    accBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    accBinding.descriptorCount = 1;
    accBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    accBinding.pImmutableSamplers = nullptr; // Optional

    VkDescriptorSetLayoutBinding firstXBinding{};
    firstXBinding.binding = 5;
    firstXBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    firstXBinding.descriptorCount = 1;
    firstXBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    firstXBinding.pImmutableSamplers = nullptr; // Optional
    
    VkDescriptorSetLayoutBinding firstWBinding{};
    firstWBinding.binding = 6;
    firstWBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    firstWBinding.descriptorCount = 1;
    firstWBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    firstWBinding.pImmutableSamplers = nullptr; // Optional

    VkDescriptorSetLayoutBinding bindings[7] = { 
        outputBinding, 
        samplerLayoutBinding, 
        uboLayoutBinding, 
        frameLayoutBinding, 
        accBinding,
        firstXBinding,
        firstWBinding
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 7;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts(swapChainImages.size(), descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = swapChainImages.size();
    allocInfo.pSetLayouts = descriptorSetLayouts.data();

    descriptorSets.resize(swapChainImages.size());
    // Allocate descriptor set
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (int i = 0; i < swapChainImages.size(); i++) {
        // Descriptor for the uniform buffer
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);
        
        VkDescriptorBufferInfo framebufferInfo{};
        framebufferInfo.buffer = frameInfoBuffer[i];
        framebufferInfo.offset = 0;
        framebufferInfo.range = sizeof(FrameInfo);

        VkDescriptorImageInfo rtInfo{};
        rtInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        rtInfo.imageView = swapChainImageViews[i];
        rtInfo.sampler = VK_NULL_HANDLE;

        VkDescriptorImageInfo accInfo{};
        accInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        accInfo.imageView = accImageView;
        accInfo.sampler = VK_NULL_HANDLE;

        VkDescriptorImageInfo firstXInfo{};
        firstXInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        firstXInfo.imageView = firstXImageView;
        firstXInfo.sampler = VK_NULL_HANDLE;

        VkDescriptorImageInfo firstWInfo{};
        firstWInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        firstWInfo.imageView = firstWImageView;
        firstWInfo.sampler = VK_NULL_HANDLE;


        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;

        VkWriteDescriptorSet descriptorWrites[7] = { };
        
        descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[6].dstSet = descriptorSets[i];
        descriptorWrites[6].dstBinding = 6;
        descriptorWrites[6].dstArrayElement = 0;
        descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[6].descriptorCount = 1;
        descriptorWrites[6].pImageInfo = &firstWInfo;

        descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[5].dstSet = descriptorSets[i];
        descriptorWrites[5].dstBinding = 5;
        descriptorWrites[5].dstArrayElement = 0;
        descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[5].descriptorCount = 1;
        descriptorWrites[5].pImageInfo = &firstXInfo;

        descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[4].dstSet = descriptorSets[i];
        descriptorWrites[4].dstBinding = 4;
        descriptorWrites[4].dstArrayElement = 0;
        descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[4].descriptorCount = 1;
        descriptorWrites[4].pImageInfo = &accInfo;

        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = descriptorSets[i];
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].dstArrayElement = 0;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pBufferInfo = &framebufferInfo;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = descriptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo; 
        
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pImageInfo = &rtInfo; 
        vkUpdateDescriptorSets(device, 7, descriptorWrites, 0, nullptr);
    } 
}   
  
void Create_Resources() {
    // Create uniform for each image in swapchain
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);
    createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffer, uniformBufferMemory);

    VkDeviceSize frameBufferSize = sizeof(FrameInfo);
    frameInfoBuffer.resize(swapChainImages.size());
    frameInfoBufferMemory.resize(swapChainImages.size());
    for (int i=0; i<swapChainImages.size(); i++)
        createBuffer(frameBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, frameInfoBuffer[i], frameInfoBufferMemory[i]);
     
    // Create image for testing
    /*int pixels[4] = {
        0xFFFF0000, 0xFF00FF00,
        0xFF0000FF, 0xFFFF00FF
    };*/

    // Load Grid File
    float* gridVoxels;
    unsigned int sizeX, sizeY, sizeZ;
    if (!load_grid_xyz(CLOUD_PATH, gridVoxels, sizeX, sizeY, sizeZ))
    {
        throw std::runtime_error("Error loading grid file.");
    }

    gridExtent = { sizeX, sizeY, sizeZ };
     
    VkDeviceSize imageSize = sizeof(float) * gridExtent.width * gridExtent.height * gridExtent.depth;
      
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
     
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, gridVoxels, static_cast<size_t>(imageSize)); 
    vkUnmapMemory(device, stagingBufferMemory); 

    createImage(gridExtent.width, gridExtent.height, gridExtent.depth, VK_FORMAT_R32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);
     
    transitionImageLayout(textureImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(gridExtent.width), static_cast<uint32_t>(gridExtent.height), static_cast<uint32_t>(gridExtent.depth));
    transitionImageLayout(textureImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    // Create the image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = textureImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    viewInfo.format = VK_FORMAT_R32_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &viewInfo, nullptr, &textureImageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    } 
     
    // Sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }  
     
    // Create AccImage
    createImage(PresentationResolution.width, PresentationResolution.height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, accImage, accImageMemory);
    VkImageViewCreateInfo accviewInfo{};
    accviewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    accviewInfo.image = accImage;
    accviewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    accviewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    accviewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    accviewInfo.subresourceRange.baseMipLevel = 0;
    accviewInfo.subresourceRange.levelCount = 1;
    accviewInfo.subresourceRange.baseArrayLayer = 0;
    accviewInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &accviewInfo, nullptr, &accImageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }
}

void Update_Resources() {
    
    // Updating UBO 
    UniformBufferObject ubo{};
    auto time = (float)glfwGetTime();
     
    glm::vec3 cameraPosition = CAMERA_POSITION;
    glm::vec3 cameraTarget = CAMERA_TARGET;
    glm::vec3 cameraUp = CAMERA_UP;
    auto viewMatrix = glm::lookAt(cameraPosition, cameraTarget, cameraUp);
    auto projMatrix = glm::perspective(glm::radians(54.43f), PresentationResolution.width / (float)PresentationResolution.height, 0.1f, 40.0f);

    // compute transform from proj to world
    auto proj2World = glm::inverse(projMatrix * viewMatrix);
    ubo.proj2world = proj2World;  
      
    // compute cloud world extension
    int maxDim = std::max(gridExtent.width, std::max(gridExtent.height, gridExtent.depth));
    ubo.box_maxim = glm::vec3(gridExtent.width, gridExtent.height, gridExtent.depth) * 0.5f / (float)maxDim;
    ubo.box_minim = -ubo.box_maxim;
    ubo.Extinction = CLOUD_EXTINCTION_BASE * CLOUD_EXTINCTION_SCALE;
    ubo.ScatteringAlbedo = CLOUD_SCATTERING_ALBEDO;
    ubo.G = 0.875f; 

    ubo.SunDirection = SUNLIGHT_DIR;
    ubo.SunIntensity = SUNLIGHT_COLOR;
     
    void* data;
    vkMapMemory(device, uniformBufferMemory, 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, uniformBufferMemory);
}

void Update_Frame(int current_image, int frame) {

    // Updating UBO
    FrameInfo ubo{};
    ubo.frameCount = frame;

    void* data; 
    vkMapMemory(device, frameInfoBufferMemory[current_image], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, frameInfoBufferMemory[current_image]);
}

void Populate_Commands() {
    for (size_t i = 0; i < commandBuffers.size(); i++) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0; // Optional
        beginInfo.pInheritanceInfo = nullptr; // Optional

        if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[i];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = PresentationResolution;
        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        // Begin render pass 
        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Rendering stuff
        // Set the pipeline
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, graphicsPipeline);

        vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

        vkCmdDispatch(commandBuffers[i], (PresentationResolution.width / 32) + 1, (PresentationResolution.height / 32) + 1, 1);
        //vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);

        // End render pass
        vkCmdEndRenderPass(commandBuffers[i]);

        // Finish loading commands in this command buffer
        if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }
}

void DrawFrame(int frame) {
    // Get Framebuffer
    uint32_t imageIndex; // Index of the current target in swapchain
    vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

    Update_Frame(imageIndex, frame);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
    VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    // Render
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    // Present 
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = { swapChain };
    presentInfo.swapchainCount = 1; 
    presentInfo.pSwapchains = swapChains;   
    presentInfo.pImageIndices = &imageIndex; 
    presentInfo.pResults = nullptr; // Optional
    vkQueuePresentKHR(presentQueue, &presentInfo);
}  
 
void Cleanup() {
    
    vkDestroySampler(device, textureSampler, nullptr);
    vkDestroyImageView(device, textureImageView, nullptr);
    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);

    vkDestroyImageView(device, accImageView, nullptr);
    vkDestroyImage(device, accImage, nullptr);
    vkFreeMemory(device, accImageMemory, nullptr);

    vkDestroyImageView(device, firstXImageView, nullptr);
    vkDestroyImage(device, firstXImage, nullptr);
    vkFreeMemory(device, firstXImageMemory, nullptr);

    vkDestroyImageView(device, firstWImageView, nullptr);
    vkDestroyImage(device, firstWImage, nullptr);
    vkFreeMemory(device, firstWImageMemory, nullptr);

    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    for (int i = 0; i < swapChainImages.size(); i++) {
        vkDestroyBuffer(device, frameInfoBuffer[i], nullptr);
        vkFreeMemory(device, frameInfoBufferMemory[i], nullptr);
    }
    vkDestroyBuffer(device, uniformBuffer, nullptr);
    vkFreeMemory(device, uniformBufferMemory, nullptr);
    
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);

    vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
    vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    for (auto framebuffer : swapChainFramebuffers) vkDestroyFramebuffer(device, framebuffer, nullptr);
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);
    vkDestroyShaderModule(device, computeShader, nullptr);
    for (auto imageView : swapChainImageViews) vkDestroyImageView(device, imageView, nullptr);
    vkDestroySwapchainKHR(device, swapChain, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}

int main() {
    
    GLFW_Initialization(RESOLUTION_WIDTH, RESOLUTION_HEIGHT); // glfw and window

    Vulkan_Initialization(); // physical device, device and queues.

    Presenter_Initialization(); // presentation objects, swapchain and render targets

    Create_CommandPool(); // creates a command pool for storing command buffers
    
    Create_Resources(); // creates the resources used during rendering

    Create_DescriptorPoolAndLayout(); // Creates the descriptor pool, layouts and sets.

    Create_GraphicsPipeline(); // creates the pipeline for rendering

    Populate_Commands(); // populates all command buffers with same rendering pass

    std::cout << "[INFO] Initialization done..." << std::endl;

    double start_time = glfwGetTime();
    int current_frame = 0;

    Update_Resources();

    // Main Loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
         
        DrawFrame(current_frame);

        current_frame++;
        
        double mspf = (glfwGetTime() - start_time) / current_frame;

        if (current_frame % 1000 == 0) {
            std::cout << "Time per frame (ms): " << (mspf*1000) << std::endl;
        }
    }

    Cleanup();

    return 0;
}