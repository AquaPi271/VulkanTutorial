#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <optional>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <set> 

// Since Vulkan is a platform agnostic API, it can not interface directly with 
// the window system on its own.To establish the connection between Vulkan and 
// the window system to present results to the screen, we need to use the WSI 
// (Window System Integration) extensions.  In this chapter we'll discuss the 
// first one, which is VK_KHR_surface. It exposes a VkSurfaceKHR object that 
// represents an abstract type of surface to present rendered images to.  The 
// surface in our program will be backed by the window that we've already 
// opened with GLFW.

// The VK_KHR_surface extension is an instance level extension and we've 
// actually already enabled it, because it's included in the list returned by 
// glfwGetRequiredInstanceExtensions.  The list also includes some other WSI 
// extensions that we'll use in the next couple of chapters.

// The window surface needs to be created right after the instance creation, 
// because it can actually influence the physical device selection.  The 
// reason we postponed this is because window surfaces are part of the larger 
// topic of render targets and presentation for which the explanation would have 
// cluttered the basic setup.  It should also be noted that window surfaces are 
// an entirely optional component in Vulkan, if you just need off - screen 
// rendering.  Vulkan allows you to do that without hacks like creating an 
// invisible window (necessary for OpenGL).
//
// The GLFW extension hides the details of the platform call to generate a 
// surface.  However, the specific platform calls can be made directly as 
// show below for edification:
//
// Update includes to:
//
// #define VK_USE_PLATFORM_WIN32_KHR
// #define GLFW_INCLUDE_VULKAN
// #include <GLFW/glfw3.h>
// #define GLFW_EXPOSE_NATIVE_WIN32
// #include <GLFW/glfw3native.h>
//
// Fill in createInfo for a WIN32 surface creation with Windows handles and
// instances (HWND and HINSTANCE are Windows platform specific):
//
// VkWin32SurfaceCreateInfoKHR createInfo{};
// createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
// createInfo.hwnd = glfwGetWin32Window(window);  // Get raw Windows handle HWND.
// createInfo.hinstance = GetModuleHandle(nullptr);  // Get HINSTANCE of current process.
//
// Create native surface will call to:
//
// if (vkCreateWin32SurfaceKHR(instance, &createInfo, nullptr, &surface) != VK_SUCCESS) {
//    throw std::runtime_error("failed to create window surface!");
// }
//
// All of the above is replaced with the GLFW library function, glfwCreateWindowSurface(...)
//
// Regarding presentation support from the tutorial text:
//
// Although the Vulkan implementation may support window system integration, that 
// does not mean that every device in the system supports it. Therefore we need to 
// extend isDeviceSuitable to ensure that a device can present images to the 
// surface we created. Since the presentation is a queue-specific feature, the 
// problem is actually about finding a queue family that supports presenting to 
// the surface we created.
//
// It's actually possible that the queue families supporting drawing commands and 
// the ones supporting presentation do not overlap. Therefore we have to take into 
// account that there could be a distinct presentation queue by modifying the 
// QueueFamilyIndices structure:

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

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
		createInfo.enabledExtensionCount = 0;

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

	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);
		return indices.isComplete();
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