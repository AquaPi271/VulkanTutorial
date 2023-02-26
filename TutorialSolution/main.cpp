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
#include <fstream>

// Fixed functions
// 
// The older graphics APIs provided default state for most of the stages of the
// graphics pipeline. In Vulkan you have to be explicit about everything, from 
// viewport size to color blending function. In this chapter we'll fill in all 
// of the structures to configure these fixed-function operations.
//
// The stages for fixed functions include:
//
// 1) Vertex Input
// 2) Input Assembly
// 3) Viewports and Scissors
// 4) Rasterizer
// 5) Multisampling
// 6) Depth and stencil testing
// 7) Color blending
// 8) Dynamic state
// 9) Pipeline layout

// Vertex Shader
//
// Graphics are normalized from framebuffer coordinates to normalized device 
// coordinates:
// 
// Center of image is (0,0) and then +1,-1 for the corners: 
//   UL = (-1,-1)
//   UR = (1,-1)
//   LL = (-1,1)
//   LR = (1,1)
// In 3D the range is from 0 to 1 (like Direct3D).

// Graphics Pipeline (keeping for reference)
//
// Barely anything touched here.  This step is more about information.
// The full pipe line runs sequentially through the following stages:
// 
// 1)  Vertex / Index Buffer data feeds ->
// 2)  Input Assembler Stage* 
// 3)  Vertex Shader
// 4)  Tessellation
// 5)  Geometry Shader
// 6)  Rasterization*
// 7)  Fragment Shader
// 8)  Color Blending*
// 9)  Framebuffer
//
// Stages with stars (*) can only modify parameters into the stage.
// Other stages are fully programmable.

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
	VkSurfaceCapabilitiesKHR capabilities{ 0 };
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
		std::cerr << "initWindow() done" << std::endl;
		initVulkan();
		std::cerr << "initVulkan() done" << std::endl;
		mainLoop();
		cleanup();
	}

private:

	GLFWwindow* window;                      // Window generated for Vulkan usage
	VkInstance instance{};                     // Vulkan handle to instance
	VkDebugUtilsMessengerEXT debugMessenger{}; // Handle to the debug messenger callback (even this needs a handle, like all things in Vulkan)
	VkSurfaceKHR surface{};                    // Abstraction of presentation surface used to draw images.  Basically, this is a way to communicate
	// to the underlying graphics system in the OS.  It's usage is platform agnostic, but its creation
	// is platform specific.
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // Holds the handle to the graphics card we're using.  Instance destruction automatically 
	// releases this handle.  No explicit destroy call is needed.
	VkDevice device{};  // Handle to the logical device.
	VkQueue graphicsQueue{}; // Handle to the queue created with the logical device above.  It is created automatically when the logical device is
	// is created.  It must be retrieved via VkGetDeviceQueue(...);
	VkQueue presentQueue{}; // Handle to the presentation queue.
	VkSwapchainKHR swapChain{}; // Handle to the swap chain.
	std::vector<VkImage> swapChainImages; // These are the images that the graphics will be written to.  Kinda wonder if I can write to them directly.
	// The images are created when the swap chain is created and therefore are deleted when the swap chain is deleted.
	VkFormat swapChainImageFormat{}; // Store image formats for future use.
	VkExtent2D swapChainExtent{};    // Store dimensions of swap chain for future use.
	std::vector<VkImageView> swapChainImageViews;
	VkRenderPass renderPass{};
	VkPipelineLayout pipelineLayout{};
	std::vector<VkFramebuffer> swapChainFramebuffers;  // Attachments created during render pass are bound to VkFramebuffer.  One VkFramebuffer
	// must exist per image in swap chain.  Hence a vector is used to track each one.
	VkCommandPool commandPool{}; // Commands are constructed on CPU side and sent as a complete set.  This allows GPU to optimize since it is 
	// directed with the complete sequence of commands.  Several may be used especially in a threaded environment.
	VkCommandBuffer commandBuffer{}; // Actual single buffer use to issue commands.  Added to commandPool.
	VkPipeline graphicsPipeline{}; // Full-blown pipeline is here.

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
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFrameBuffers();
		createCommandPool();
		createCommandBuffer();  // Allocate the command buffer.
	}

	// This will be called to write commands to the commandBuffer.
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0; // Optional
		//The flags parameter specifies how we're going to use the command buffer. The following values are available:
		// -- VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: The command buffer will be rerecorded right after executing it once.
		// -- VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT : This is a secondary command buffer that will be entirely within a single render pass.
		// -- VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : The command buffer can be resubmitted while it is also already pending execution.
		beginInfo.pInheritanceInfo = nullptr; // Optional

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		// Start the render pass.
		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
		renderPassInfo.renderArea.offset = { 0, 0 };  // Define area with one corner...
		renderPassInfo.renderArea.extent = swapChainExtent;  // and the other corner.  Should match attachment size for best performance.

		VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };  // Black with 100% opacity.
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor; // Clear colors for VK_ATTACHMENT_LOAD_OP_CLEAR.

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		// vkCmd prefix are all record commands.  All return void with errors only when finished recording.
		// The final parameter controls how the drawing commands within the render pass will be provided. It can have one of two values:
		// -- VK_SUBPASS_CONTENTS_INLINE: The render pass commands will be embedded in the primary command buffer itself and no secondary command 
		//    buffers will be executed.
		// -- VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : The render pass commands will be executed from secondary command buffers.

		// Bind to the graphics pipeline.
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

	}

	void createCommandBuffer() {
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		// The level parameter specifies if the allocated command buffers are primary or secondary command buffers.
		// -- VK_COMMAND_BUFFER_LEVEL_PRIMARY: Can be submitted to a queue for execution, but cannot be called from other command buffers.
		// -- VK_COMMAND_BUFFER_LEVEL_SECONDARY : Cannot be submitted directly, but can be called from primary command buffers.
		allocInfo.commandBufferCount = 1;

		if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}

	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		// There are two possible flags for command pools :
		// -- VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: Hint that command buffers are rerecorded with new commands 
		//    very often(may change memory allocation behavior)
		// -- VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : Allow command buffers to be rerecorded individually, 
		//    without this flag they all have to be reset together
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}

	}

	void createFrameBuffers() {
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
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void createRenderPass() {
		// Start with single color buffer attachment represented by one of the images from
		// the swap chain.
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		// The above applies to depth data.
		// The loadOp and storeOp determine what to do with the data in the attachment before rendering and after rendering.
		// We have the following choices for loadOp:
		// -- VK_ATTACHMENT_LOAD_OP_LOAD: Preserve the existing contents of the attachment
		// -- VK_ATTACHMENT_LOAD_OP_CLEAR : Clear the values to a constant at the start
		// -- VK_ATTACHMENT_LOAD_OP_DONT_CARE : Existing contents are undefined; we don't care about them
		// In our case we're going to use the clear operation to clear the framebuffer to black before drawing a new frame. 
		// There are only two possibilities for the storeOp:
		// -- VK_ATTACHMENT_STORE_OP_STORE: Rendered contents will be stored in memory and can be read later
		// -- VK_ATTACHMENT_STORE_OP_DONT_CARE : Contents of the framebuffer will be undefined after the rendering operation
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; 
		// These apply to stencil data.
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		// Textures and framebuffers in Vulkan are represented by VkImage objects with a certain pixel format, however the 
		// layout of the pixels in memory can change based on what you're trying to do with an image.
		// Some of the most common layouts are :
		// -- VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: Images used as color attachment
		// -- VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : Images to be presented in the swap chain
		// -- VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : Images to be used as destination for a memory copy operation

		// No subpasses needed in basic demo so create attachment reference to previous colorAttachment.
		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;  // Index to an array of VkAttachmentDescription.
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// Define subpasses.
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments= &colorAttachmentRef; 
		// The index of the attachment in this array is directly referenced from the fragment shader with the 
		// layout(location = 0) out vec4 outColor directive!
		// The following other types of attachments can be referenced by a subpass :
		// -- pInputAttachments: Attachments that are read from a shader
		// -- pResolveAttachments : Attachments used for multisampling color attachments
		// -- pDepthStencilAttachment : Attachment for depth and stencil data
		// -- pPreserveAttachments : Attachments that are not used by this subpass, but for which the data must be preserved

		// Finally create the render pass.
		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

	}

	void createGraphicsPipeline() {
		auto vertShaderCode = readFile("vert.spv");
		auto fragShaderCode = readFile("frag.spv");

		// The modules are just a thin wrapper around the bytecode.

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		// Create shader stages for these modules.

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;  // Assign this step to vertex shader.
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";
		// Can combine multiple fragment shaders into a single shader module and use different entry 
		// points to select behavior.  Here, "main" is the entry point.
		// pSpecializationInfo -- optional member to specify values for shader constants.  So this
		// can be done to optimize the shader module by select paths before runtime to optimize 
		// against, if the variable controls shader 'if' statements, for example.

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		// Store these for future reference.

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		// Dynamic state
		// While most of the pipeline state needs to be baked into the pipeline state, a limited amount 
		// of the state can actually be changed without recreating the pipeline at draw time.  Examples 
		// are the size of the viewport, line width and blend constants.  If you want to use dynamic 
		// state and keep these properties out, then you'll have to fill in a 
		// VkPipelineDynamicStateCreateInfo structure like this:

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		// Vertex Input Fixed Stage

		// In this tutorial, the vertices were hard-coded in the vertex shader.  
		// Therefore, no vertex buffer is needed.  Nevertheless, it must be specified
		// that we're not using it.

		// -- Bindings: spacing between data and whether the data is per-vertex or
		//    per-instance (see instancing)
		// -- Attribute descriptions : type of the attributes passed to the vertex shader, 
		//    which binding to load them fromand at which offset
		//
		// The pVertexBindingDescriptions and pVertexAttributeDescriptions members point 
		// to an array of structs that describe the aforementioned details for loading 
		// vertex data. Add this structure to the createGraphicsPipeline function right 
		// after the shaderStages array.

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 0;
		vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
		vertexInputInfo.vertexAttributeDescriptionCount = 0;
		vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

		// Input Assembly Fixed Stage

		// The VkPipelineInputAssemblyStateCreateInfo struct describes two things: what 
		// kind of geometry will be drawn from the vertices and if primitive restart 
		// should be enabled. The former is specified in the topology member and can have 
		// values like:
		//
		// -- VK_PRIMITIVE_TOPOLOGY_POINT_LIST: points from vertices
		// -- VK_PRIMITIVE_TOPOLOGY_LINE_LIST : line from every 2 vertices without reuse
		// -- VK_PRIMITIVE_TOPOLOGY_LINE_STRIP : the end vertex of every line is used as 
		//    start vertex for the next line
		// -- VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : triangle from every 3 vertices 
		//    without reuse
		// -- VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : the secondand third vertex of every
		//    triangle are used as first two vertices of the next triangle
		//
		// Normally, the vertices are loaded from the vertex buffer by index in 
		// sequential order, but with an element buffer you can specify the indices to 
		// use yourself.  This allows you to perform optimizations like reusing vertices.
		// If you set the primitiveRestartEnable member to VK_TRUE, then it's possible to 
		// break up lines and triangles in the _STRIP topology modes by using a special 
		// index of 0xFFFF or 0xFFFFFFFF.

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		// Viewports and Scissors
		//
		// A viewport basically describes the region of the framebuffer that the output 
		// will be rendered to. This will almost always be (0, 0) to (width, height).

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		// While viewports define the transformation from the image to the framebuffer, 
		// scissor rectangles define in which regions pixels will actually be stored.
		// Any pixels outside the scissor rectangles will be discarded by the rasterizer.
		// They function like a filter rather than a transformation.

		// Create Scissor the size of the frame buffer to include it all.

		VkRect2D scissor{};
		scissor.offset = { 0,0 };
		scissor.extent = swapChainExtent;

		// Combine viewport and scissor into Viewport state create.  Note that some 
		// graphics cards can support multiple viewports and scissors.  This 
		// functionality must be enabled in the logical device.

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;  // Without dynamic state these should be specified here.
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor; // Without dynamic state these should be specified here.

		// Rasterizer

		// The rasterizer takes the geometry that is shaped by the vertices from the 
		// vertex shader and turns it into fragments to be colored by the fragment 
		// shader.  It also performs depth testing, face culling and the scissor test, 
		// and it can be configured to output fragments that fill entire polygons or 
		// just the edges (wireframe rendering).  All this is configured using the 
		// VkPipelineRasterizationStateCreateInfo structure.

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;  // If set to true, pixels outside of 
		// depth are clamped to depth boundary rather than being discarded.  Useful in
		// some special cases like shadow maps.  Need to enable GPU feature for it.
		rasterizer.rasterizerDiscardEnable = VK_FALSE;  // IF set to true, no output
		// is sent to this state, effectively disabling image production.
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		// polygonMode determines how fragments are generated for geometry:
		//
		// -- VK_POLYGON_MODE_FILL: fill the area of the polygon with fragments
		// -- VK_POLYGON_MODE_LINE: polygon edges are drawn as lines
		// -- VK_POLYGON_MODE_POINT : polygon vertices are drawn as points
		// 
		// Any mode other than FILL requires GPU feature to be enabled.
		rasterizer.lineWidth = 1.0f;  // determines thickness of lines in units of
		// fragments.  Any thicker than 1.0f requires wideLines GPU feature enabled.
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		// Face culling:  none, front, back, or both.
		// Front face:  Vertex order of face in "front"  (clockwise or counterclockwise).
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f; // Optional
		rasterizer.depthBiasClamp = 0.0f; // Optional
		rasterizer.depthBiasSlopeFactor = 0.0f; // Optional
		// Depth bias is used for shadow mapping.  Not used here, hence VK_FALSE.

		// Keep multisampling disabled for now, it will be revisited in later chapters.
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE; // Disable for now.
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT; // # of samples per bit
		multisampling.minSampleShading = 1.0f; // Optional
		multisampling.pSampleMask = nullptr; // Optional
		multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
		multisampling.alphaToOneEnable = VK_FALSE; // Optional

		// Color blending:  After a fragment shader has returned a color, it needs to be 
		// combined with the color that is already in the framebuffer.

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

		// For an alpha blend attachment use this instead:
		// colorBlendAttachment.blendEnable = VK_TRUE;
		// colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		// colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		// colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		// colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		// colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		// colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		// Reference array of structures for all framebuffers.  Provides values for the constants used by the attachment.
		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE; // To use second method that's commented above, uncomment and change this to VK_TRUE.
		colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; // Optional
		colorBlending.blendConstants[1] = 0.0f; // Optional
		colorBlending.blendConstants[2] = 0.0f; // Optional
		colorBlending.blendConstants[3] = 0.0f; // Optional
 

		// Need to create default empty pipeline layout even if using nothing.
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 0; // Optional
		pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
		pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
		pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		// We can now combine all of the structures and objects from the previous chapters to create the graphics pipeline!
		// Here's the types of objects we have now, as a quick recap:
		//
		//	Shader stages : the shader modules that define the functionality of the programmable stages of the graphics pipeline
		//	Fixed - function state : all of the structures that define the fixed - function stages of the pipeline, 
		//  like input assembly, rasterizer, viewport and color blending
		//	Pipeline layout : the uniform and push values referenced by the shader that can be updated at draw time
		//	Render pass : the attachments referenced by the pipeline stages and their usage

		// We start by referencing the array of VkPipelineShaderStageCreateInfo structs.

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;

		// Then we reference all of the structures describing the fixed-function stage.

		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr; // Optional
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;

		// After that comes the pipeline layout, which is a Vulkan handle rather than a struct pointer.

		pipelineInfo.layout = pipelineLayout;

		// And finally we have the reference to the render pass and the index of the sub pass where this graphics pipeline will 
		// be used.It is also possible to use other render passes with this pipeline instead of this specific instance, but they 
		// have to be compatible with renderPass.The requirements for compatibility are described here, but we won't be using 
		// that feature in this tutorial.

		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;

		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
		pipelineInfo.basePipelineIndex = -1; // Optional

		// Create the pipeline..... finally!
		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		// Compilation and linking of GPU machine code  until graphics pipeline is
		// created.  Since we already created it, the modules are no longer needed
		// and can be destroyed.

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);

	}

	// Wrap byte-code into a shader module to be used in the pipeline.

	VkShaderModule createShaderModule(const std::vector<char>& code) {
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

	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;  // can be 1D, 2D, 3D textures or cube maps
			createInfo.format = swapChainImageFormat;

			// Colors channels can be mapped to different values.
			// Below is the default mapping.
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			// Subresource range tells the image's purpose and what part of it
			// should be accessed.  Here we're creating color targets without
			// any mipmapping or multiple layers.
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;  // This might be different for stereoscopic 3D images.

			if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image views!");
			}
		}
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

		return indices.isComplete() && extensionsSupported && swapChainAdequate;
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
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else {
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

	// Helper function to load a shader.

	static std::vector<char> readFile(const std::string& filename) {
		// The ate option starts reading from the end of the file.
		// It allows us to know how to allocate a byte vector for it.
		// The input shader is bytecode and therefore is binary.
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

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void cleanup() {
		vkDestroyCommandPool(device, commandPool, nullptr);

		// Destroy framebuffers.
		for (auto framebuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}


		// Destroy the pipeline.
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

		vkDestroyRenderPass(device, renderPass, nullptr);

		// Clean up the image views that were created for us.
		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}

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
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}