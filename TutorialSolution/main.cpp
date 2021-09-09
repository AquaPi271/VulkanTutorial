#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <optional>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <set> 
#include <cstdint> // Necessary for UINT32_MAX
#include <algorithm> // Necessary for std::min/std::max

// Vulkan does not have the concept of a "default framebuffer", hence it 
// requires an infrastructure that will own the buffers we will render to 
// before we visualize them on the screen.  This infrastructure is known as the 
// swap chain and must be created explicitly in Vulkan.  The swap chain is 
// essentially a queue of images that are waiting to be presented to the 
// screen.  Our application will acquire such an image to draw to it, and then 
// return it to the queue.  How exactly the queue works and the conditions for 
// presenting an image from the queue depend on how the swap chain is set up, 
// but the general purpose of the swap chain is to synchronize the presentation 
// of images with the refresh rate of the screen.
//
// The code must not only see if a swap chain is available but also see if it is
// compatible with the window surface.  Need to check:
//   1) Basic surface capabilities (min/max number of images in swap chain, 
//      min/max width and height of images).
//   2) Surface formats (pixel format, color space)
//   3) Available presentation modes.
// 
// Choose right settings for the swap chain:
//   1) Surface format (color depth)
//   2) Presentation mode (conditions for "swapping" images to the screen)
//   3) Swap extent (resolution of images in swap chain)
//
// Presentation Mode:
// There are 4 possible modes in Vulkan:
// VK_PRESENT_MODE_IMMEDIATE_KHR: Images submitted by your application are 
// transferred to the screen right away, which may result in tearing.
// VK_PRESENT_MODE_FIFO_KHR: The swap chain is a queue where the display takes 
// an image from the front of the queue when the display is refreshed and the 
// program inserts rendered images at the back of the queue.  If the queue is 
// full then the program has to wait.  This is most similar to vertical sync 
// as found in modern games.  The moment that the display is refreshed is 
// known as "vertical blank". (This is the only mode that is guaranteed to be
// supported).
// VK_PRESENT_MODE_FIFO_RELAXED_KHR : This mode only differs from the previous 
// one if the application is late and the queue was empty at the last vertical 
// blank.  Instead of waiting for the next vertical blank, the image is 
// transferred right away when it finally arrives.  This may result in visible 
// tearing.
// VK_PRESENT_MODE_MAILBOX_KHR : This is another variation of the second mode.
// Instead of blocking the application when the queue is full, the images that 
// are already queued are simply replaced with the newer ones.  This mode can 
// be used to render frames as fast as possible while still avoiding tearing, 
// resulting in fewer latency issues than standard vertical sync.  This is 
// commonly known as "triple buffering", although the existence of three 
// buffers alone does not necessarily mean that the framerate is unlocked.
// 

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// List required extensions.  In this case, add the swap chain which is not
// part of Vulkan API proper (hence extension).

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// Adds validation layer.

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

// Enable validation layers based on debug or release builds.

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// Proxy function to find the address of the extension function since it is not 
// within the API.  Almost like a dynamic function load.

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
	const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator,
	VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

// This function mirrors the CreateDebugUtilsMessengerEXT callback and is used 
// for cleanup of resources.

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

// Store swap chain details in this structure.
struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

// Bundle different types of queues into a struct to lookup later.
struct QueueFamilyIndices {
	// std::optional is a new C++17 feature that allows a variable to be used yet not hold a value.
	// Since any uint32_t value could technically be a valid graphicsFamily, there's no safe value
	// to assign to it to indicate that it is invalid.  The optional wrapper has the method has_value()
	// to indicate if a value has been assigned to it.
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:

	GLFWwindow* window;                      // Window generated for Vulkan usage
	VkInstance instance;                     // Vulkan handle to instance
	VkDebugUtilsMessengerEXT debugMessenger; // Handle to the debug messenger callback (even this needs a handle, like all things in Vulkan)
	VkSurfaceKHR surface;                    // Abstraction of presentation surface used to draw images.  Basically, this is a way to communicate
	                                         // to the underlying graphics system in the OS.  It's usage is platform agnostic, but its creation
	                                         // is platform specific.
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // Holds the handle to the graphics card we're using.  Instance destruction automatically 
	                                                  // releases this handle.  No explicit destroy call is needed.
	VkDevice device;  // Handle to the logical device.
	VkQueue graphicsQueue; // Handle to the queue created with the logical device above.  It is created automatically when the logical device is
						   // is created.  It must be retrieved via VkGetDeviceQueue(...);
	VkQueue presentQueue; // Handle to the presentation queue.
	VkSwapchainKHR swapChain; // Handle to the swap chain.
	std::vector<VkImage> swapChainImages; // These are the images that the graphics will be written to.  Kinda wonder if I can write to them directly.
	// The images are created when the swap chain is created and therefore are deleted when the swap chain is deleted.
	VkFormat swapChainImageFormat; // Store image formats for future use.
	VkExtent2D swapChainExtent;    // Store dimensions of swap chain for future use.

	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Prevent generation of OpenGL context
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);    // No window resize for now...

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);  // 4th parameter picks monitor... interesting
	}
	void initVulkan() {
		createInstance();
		setupDebugMessenger();   // Call to setup debug callbacks.
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
	}

	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		// Number of images in the swap chain...
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;  // attempt 1 more than minimum for good performance
		// Make sure not to exceed maximum....
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		// Fill in the structure to send all of this information to Vulkan and get our magic 
		// handles.

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1; // Always 1 unless developing a stereoscopic 3D application.
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;  // Specifies operations used in the image swap chain

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);  // Remember this?
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		// In the event two different graphics cards separately provide graphics and presentation queues...
		// VK_SHARING_MODE_EXCLUSIVE: An image is owned by one queue family at a timeand ownership must be explicitly 
		//    transferred before using it in another queue family.This option offers the best performance.
		// VK_SHARING_MODE_CONCURRENT : Images can be used across multiple queue families without explicit ownership
		//    transfers.

		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0; // Optional
			createInfo.pQueueFamilyIndices = nullptr; // Optional
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;  // Can do things like 90 degree turns or horizontal flips.
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;  // Blend with other windows in system or not (in this case, not).

		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;  // Don't care about color of pixels that are obscured by another window, for example.

		createInfo.oldSwapchain = VK_NULL_HANDLE;  // If Window resizes, a new swap chain must be created from scratch.  However, the NULL HANDLE
		                                           // says to simply give up if a resize happens.

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create the swap chain!");
		}
		      
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr); // first get number of images
		swapChainImages.resize(imageCount); // resize vector to match the count
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data()); // Load up our images.

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void createSurface() {
		// Super easy call to create surface.  No structures needed.
		// Parameters are VkInstance created earlier, the window pointer created earlier by GLFW, 
		// custom allocators, and pointer to assigned surface handle.
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		// Since we need more than one queue, use a set to store multiple queues.

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		// Can specify a priority to queues from 0.0 to 1.0.  Setting a priority, even if only one queue, is required.
		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;  // Need only 1 queue... current driver state only supports low number queues.
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		// Specify device features we want.  Currently, we want nothing but will be updating in later sections.
		VkPhysicalDeviceFeatures deviceFeatures{};

		// Fill in main logical device structure with information supplied so far.
		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();

		createInfo.pEnabledFeatures = &deviceFeatures;
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());  // Enable extensions here.
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers) { 
			// Below two fields are no longer used in newer Vulkan releases.  However, they are set here in 
			// case an older version of Vulkan is used.
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		// Create the logical device....
		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		// Retrieve the queue created with this device.
		// parameters = logical device, queue family, queue index, and pointer to store the queue handle.
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	// Resolution of swap chain images and it's almost exactly equal to the resolution of the window that
	// we're drawing in pixels.  Usually pixels match screen width and height but this is not always the
	// case.
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != UINT32_MAX) {
			return capabilities.currentExtent;
		}
		else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) { // Nice trade-off if energy use is not a concern.
				return availablePresentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	// Choose a format that suits our needs.
	// The format indicates how the pixels are laid out and what size they are.
	// We'll look for BGRA (blue green red alpha of 8 bits each and in that order): B8G8R8A8
	// Color space indicates the range of colors to be used.  SRGB tends to produce
	// the most accuate perceived colors.
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}
		return availableFormats[0];  // Pick default if we can't find what we want.  =/
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);  

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);  // typical query call to find number of devices 
		
		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()); // get handles to all physical devices

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	// Modified to check if extensions are supported.  In this case it will look
	// for a swap chain.

	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		if (extensionsSupported) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return indices.isComplete() && extensionsSupported&& swapChainAdequate;
	}

	// Enumerate extensions and check if all required extensions are found.

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		// Delete each required extension that was found.
		// If requiredExtensions is fully empty then all extensions have been met.
		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;
		// Logic to find graphics queue family
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		// For now we need a queue that supports graphics.  QueueFamily properties also can return
		// the number of queues supported among other information.
		//
		// Now checking for presentation support.  Generally, if one card supports both, that card
		// should be preferred than having two different cards for each queue.  One card running
		// the entire graphics program is more efficient.  However, it is possible to run different
		// queues to different cards.

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
			if (presentSupport) {
				indices.presentFamily = i;
			}
			if (indices.isComplete()) {
				break;
			}
			i++;
		}

		return indices;
	}

	// The normal debug messenger needs an instance to receive information about Vulkan.  However, it does not
	// allow debug of instance creation and destructor since that is a required parameter to those callbacks.
	//
	// To get debug information about instance creation and destruction, a separate callback is created and
	// attached to the pNext extension field of VkInstanceCreateInfo.
	// 
	// populateDebugMessengerCreateInfo is a refactor of code common to both debugMessengers.

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
		createInfo.pUserData = nullptr;
	}

	void setupDebugMessenger() {
		if (!enableValidationLayers) return;

		// Fill details about the callback so Vulkan engine can register the callback properly.
		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		// messageSeverity is message severity classes that should be passed to the callback.
		// messageType is a filter on the types of messages to receive.
		// pfnUserCall is the callback function set earlier.

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}

		// The messenger is specific to the Vulkan instance we're using (instance variable which is a "child").
		// Of course, createInfo is the specification of the callback.
		// The third paramter is for custom allocators which is not used, and hence nullptr.
		// Finally the debugMessenger is the callback function that was created.
	}

	void createInstance() {

		// Check validation layers if in debug mode.

		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};    // Setup optional info to let Vulkan know how to optimize.
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;  // Explicitly tell Vulkan the type of struct used.
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};  // Non-optional information which will specify global extensions and validation layers to use.
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; // Tell Vulkan type of struct, per usual...
		createInfo.pApplicationInfo = &appInfo;

		// Add extensions during createInstance() call.

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		//  In debug mode, get validation layer information.
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		// Vulkan is platform agnostic... so extension is pointed to the engine for Windows windows (GLFW).
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;

		displayAllExtensions();

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		//for (auto i = 0; i < glfwExtensionCount; ++i ) {
		//	std::cout << "REQUIRED EXTENSION: " << glfwExtensions[i] << std::endl;
		//}

		// TODO:  Setting these while setting up the debug callback causes instance creation to fail.
		//createInfo.enabledExtensionCount = glfwExtensionCount;
		//createInfo.ppEnabledExtensionNames = glfwExtensions;
		createInfo.enabledLayerCount = 0;
		// Other 'createInfo' structure elements left blank and will be explained later.

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)& debugCreateInfo;
		} else {
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}

		// The above can be simplified to:
		// if ( vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
		//     throw std::runtime_error("failed to create instance!");
		// }

		// General pattern of object creation with Vulkan:
		// 1) Pointer to struct with creation info.
		// 2) Pointer to custom allocator callbacks, always nullptr in this tutorial.
		// 3) Pointer to the variable that stores the handle to the new object.

		// VkResult will either be VK_SUCCESS on successful function call or an error code otherwise.
	}

	void displayAllExtensions() {
		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
		
		std::cout << "available extensions:\n";

		for (const auto& extension : extensions) {
			std::cout << "\t" << extension.extensionName << "\n";
		}
	}

	// Query the SDK to find which validation layers are supported.
	// Per usual this starts with getting a count of the total layers supported
	// followed by a call to receive the supported layer information.

	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()); 
		
		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	// Return required list of extensions based on whether validation layers are enabled or not.
	// GLFW specified extensions are always required.  VK_EXT_DEBUG_UTILS_EXTENSION_NAME is a string
	// macro and is used to avoid string typos.

	std::vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	// Debug callback function with signature cleaned up by VKAPI_ATTR VkBool32 VKAPI_CALL.
	// The function must be static in a class or external to the class.
	//
	// The first parameter uses the following flags to specify the severity of the messages.
	// VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT -- Diagnostic message
	// VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT -- Informational message like the creatio of a 
	//   resource.
	// VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT -- Message about behavior that is not 
	//   necessarily an error, but very likely a bug in your application.
	// VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT -- Message about behavior that is invalid and
	//   may cause crashes.
	// 
	// These are setup in such a way that they can be compared (e.g.):
	// if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ) {
	//   // Message is important enough to show.
	// }
	//
	// The parameter for messageType can be:
	// VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT: Some event has happened that is unrelated to 
	//   the specification or performance
	// VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT : Something has happened that violates the 
	//   specification or indicates a possible mistake
	// VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT : Potential non - optimal use of Vulkan
	// 
	// The pCallbackData parameter refers to a VkDebugUtilsMessengerCallbackDataEXT struct 
	// containing the details of the message itself, with the most important members being:
	//   pMessage: The debug message as a null - terminated string
	//   pObjects : Array of Vulkan object handles related to the message
	//   objectCount : Number of objects in array
	//
	// The final parameter, pUserData contains a pointer used to pass along data during callback 
	//   setup.
	//
	// The callback returns true or false Boolean and indicates if the program should abort.
	// A true value indicates a fatal, abort error.


	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData) {

		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void cleanup() {
		// Clean the swap chain.
		vkDestroySwapchainKHR(device, swapChain, nullptr);

		// Logical devices must be cleaned up.
		vkDestroyDevice(device, nullptr);  

		// Must return resources set aside for the debug messenger.
		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}
};

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	}
	catch (const std::exception & e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}