module llm;

import vk;
import std::math;
import std::io;
import std::os::env;
import std::collections::list;
import std::collections::map;

faultdef NO_SUITABLE_DEVICE, FILE_NOT_FOUND, COMPUTE_ERROR;

struct DeviceContext
{
	Instance instance;
	Device device;
	PhysicalDevice pdevice;
	CommandPool command_pool;
	CommandBuffer command_buffer;
	DeviceQueue compute_queue;
	vk::MemoryAllocator allocator;
	Fence compute_fence;
	bool coopmat_supported; // VK_KHR_cooperative_matrix + fp16x16x16x16->fp32 mode available
}

// VRAM pressure snapshot for the main device-local (VRAM) heap, via
// VK_EXT_memory_budget. `used` = this process's current device-local usage;
// `budget` = how much it may use before the driver starts evicting pages to GTT
// (system RAM over PCIe). `budget` already discounts VRAM held by other
// processes (e.g. the desktop compositor), so `used` reaching `budget` is the
// real "about to spill and run slow" signal — not just our own footprint vs the
// card size. `total` = physical heap size. If the driver reports no budget,
// falls back to our own committed bytes vs ~94% of the heap.
fn void DeviceContext.vram_status(&self, ulong* used, ulong* budget, ulong* total)
{
	PhysicalDeviceMemoryBudgetPropertiesEXT bud = {
		.sType = vk::StructureType.STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
		.pNext = null,
	};
	PhysicalDeviceMemoryProperties2 mp = {
		.sType = vk::StructureType.STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
		.pNext = &bud,
	};
	vk::getPhysicalDeviceMemoryProperties2(self.pdevice, &mp);

	// The main VRAM heap is the largest DEVICE_LOCAL heap (skip the small
	// resizable-BAR host-visible+device-local heap some GPUs also expose).
	uint idx   = 0;
	ulong best = 0;
	for (uint i = 0; i < mp.memoryProperties.memoryHeapCount; i++)
	{
		MemoryHeap h = mp.memoryProperties.memoryHeaps[i];
		// VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x1 (constdef member doesn't resolve
		// unqualified across modules; the value is fixed by the Vulkan spec).
		if (((uint)h.flags & 0x1) != 0 && (ulong)h.size > best)
		{
			best = (ulong)h.size;
			idx = i;
		}
	}
	*total = best;
	if ((ulong)bud.heapBudget[idx] > 0)
	{
		*used = (ulong)bud.heapUsage[idx];
		*budget = (ulong)bud.heapBudget[idx];
	}
	else
	{
		*used = (ulong)self.allocator.total_allocated();
		*budget = best * 94 / 100;
	}
}

fn DeviceContext? createContext()
{
	ZStringList device_extensions;
	defer device_extensions.free();
	device_extensions.push_all({ "VK_KHR_push_descriptor" });
	// Phase 3: cooperative matrix tensor-core path. Both extensions are
	// optional — checked at runtime, falls back to the tiled GEMM path if
	// missing or if the right matrix configuration isn't supported.
	ZString[2] coopmat_exts = { "VK_KHR_cooperative_matrix", "VK_KHR_shader_float16_int8" };

	Instance instance;
	PhysicalDevice physical_device;
	DeviceQueue compute_queue;
	Device device;

	// Nothing links against Vulkan: every command is a function pointer bound at
	// runtime, so the loader has to be opened before the first call. On macOS
	// there is no system Vulkan — vk ships the loader and a Metal driver
	// (KosmicKrisp) in its macos-aarch64/ folder, and both are found next to the
	// executable, which is what the `macos-libs` prepare target puts them there
	// for. createDefaultInstance runs vk::init, hands the bundled driver to the
	// loader via VK_LUNARG_direct_driver_loading when it finds one, and binds
	// every command off the instance it builds.
	Instance? created = vk::createDefaultInstance(
		{ .app_name = "LLM Runner", .api_version = vk::@makeApiVersion(0, 1, 3, 0), }
	);

	if (catch err = created)
	{
		io::printfn("Instance creation error: %s", err);
		return NO_SUITABLE_DEVICE~;
	}

	instance = created;

	// Deferred rather than freed after the loop: the "no suitable device" exit
	// below returns from the middle of this function.
	DeviceList devices = instance.getDevices()!!;
	defer devices.free();

	uint best_score = 0;
	PhysicalDeviceProperties device_props;

	foreach (usz i, entry : devices)
	{
		bool supports_extensions             = entry.checkExtensionSupport(device_extensions)!!;
		QueuePropertiesList queue_properties = entry.getQueueProperties()!!;
		defer queue_properties.free();

		int queue_index = queue_properties.find(
			fn (
				sz index,
				QueueFamilyProperties props
			) => ((QueueFlagBits)props.queueFlags & QUEUE_COMPUTE_BIT) != (QueueFlagBits)0
				&& ((QueueFlagBits)props.queueFlags & QUEUE_TRANSFER_BIT) != (QueueFlagBits)0
		);

		PhysicalDeviceProperties properties;
		vk::getPhysicalDeviceProperties(entry, &properties);

		if (queue_index >= 0 && supports_extensions && properties.vendorID != 0)
		{
			uint score = 0;
			if (properties.deviceType == (PhysicalDeviceType)1) score += 100;
			if (properties.deviceType == (PhysicalDeviceType)2) score += 200;

			if (score > best_score)
			{
				best_score = score;
				physical_device = entry;
				compute_queue.index = (uint)queue_index;
				device_props = properties;
			}
		}
	}

	if (physical_device == null)
	{
		io::printfn("Found no suitable device");
		return NO_SUITABLE_DEVICE~;
	}

	io::printfn("Device selected: %s", (ZString)device_props.deviceName[0..]);

	PhysicalDeviceLimits limits = device_props.limits;
	io::printfn(
		"  maxComputeWorkGroupCount: (%d, %d, %d)",
		limits.maxComputeWorkGroupCount[0],
		limits.maxComputeWorkGroupCount[1],
		limits.maxComputeWorkGroupCount[2]
	);
	io::printfn(
		"  maxComputeWorkGroupSize: (%d, %d, %d)",
		limits.maxComputeWorkGroupSize[0],
		limits.maxComputeWorkGroupSize[1],
		limits.maxComputeWorkGroupSize[2]
	);

	// Query the driver ID so we can blacklist drivers with broken cooperative
	// matrix support. Mesa RADV and Mesa ANV both produce wrong results from
	// OpCooperativeMatrixStoreKHR — see freedesktop Mesa issue #10847 (open as
	// of 2025), and the ncnn project applies the same blacklist (Tencent/ncnn
	// PR #5739). The Phase 3 path is gated on this check, so the program runs
	// the existing tiled GEMM on broken drivers and the coopmat shader
	// entirely on NVIDIA / fixed Mesa.
	PhysicalDeviceDriverProperties driverProps = {
		.sType = vk::StructureType.STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
		.pNext = null,
	};
	PhysicalDeviceProperties2 props2 = {
		.sType = vk::StructureType.STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &driverProps,
	};
	vk::getPhysicalDeviceProperties2(physical_device, &props2);
	bool driver_blacklisted_for_coopmat = driverProps.driverID == vk::DriverId.DRIVER_ID_MESA_RADV
		|| driverProps.driverID == vk::DriverId.DRIVER_ID_INTEL_OPEN_SOURCE_MESA;

	// Opt-in override: set LLM_FORCE_COOPMAT=1 to bypass the Mesa blacklist.
	// Mesa #10847 may be fixed for newer Mesa versions / specific hardware,
	// so this lets the user test correctness vs the tiled fallback.
	bool force_coopmat = false;
	if (try String v = env::tget_var("LLM_FORCE_COOPMAT"))
	{
		if (v == "1" || v == "true" || v == "yes") force_coopmat = true;
	}
	if (force_coopmat && driver_blacklisted_for_coopmat)
	{
		io::printfn("  cooperative matrix: LLM_FORCE_COOPMAT=1 — bypassing Mesa blacklist (verify output!)");
		driver_blacklisted_for_coopmat = false;
	}

	// Opt-out override: set LLM_DISABLE_COOPMAT=1 to force the tiled GEMM
	// fallback even on drivers that aren't on the Mesa blacklist above (e.g.
	// NVIDIA), in case that specific driver/GPU also miscomputes
	// OpCooperativeMatrixStoreKHR for some input shapes. Symmetric with
	// LLM_FORCE_COOPMAT — one opts in past a blacklist, this opts out of a
	// driver that isn't (yet) blacklisted.
	bool disable_coopmat = false;
	if (try String v = env::tget_var("LLM_DISABLE_COOPMAT"))
	{
		if (v == "1" || v == "true" || v == "yes") disable_coopmat = true;
	}
	if (disable_coopmat)
	{
		io::printfn("  cooperative matrix: LLM_DISABLE_COOPMAT=1 — forcing tiled GEMM fallback");
	}

	// Probe for cooperative matrix support: device must advertise the extension
	// AND expose a 16x16x16 fp16xfp16->fp32 subgroup-scope configuration.
	bool coopmat_supported = false;
	if (driver_blacklisted_for_coopmat)
	{
		io::printfn("  cooperative matrix: blacklisted on this driver (Mesa #10847 — wrong results)");
		io::printfn("  set LLM_FORCE_COOPMAT=1 to override and test if your driver is fixed");
	}
	else if (disable_coopmat)
	{
		// Already printed the LLM_DISABLE_COOPMAT message above; skip the probe entirely.
	}
	else
	{
		ZStringList probe;
		probe.push_all({ "VK_KHR_cooperative_matrix" });
		bool ext_ok = false;
		if (try b = physical_device.checkExtensionSupport(probe)) ext_ok = b;
		probe.free();
		if (ext_ok)
		{
			uint cnt = 0;
			vk::getPhysicalDeviceCooperativeMatrixPropertiesKHR(physical_device, &cnt, null)!!;
			if (cnt > 0 && cnt <= 64)
			{
				CooperativeMatrixPropertiesKHR[64] props;
				for (uint i = 0; i < cnt; i++)
				{
					props[i].sType = vk::StructureType.STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
				}
				vk::getPhysicalDeviceCooperativeMatrixPropertiesKHR(physical_device, &cnt, &props[0])!!;
				for (uint i = 0; i < cnt; i++)
				{
					if (props[i].mSize == 16 && props[i].nSize == 16 && props[i].kSize == 16
						&& props[i].aType == vk::ComponentTypeKHR.COMPONENT_TYPE_FLOAT16_KHR
						&& props[i].bType == vk::ComponentTypeKHR.COMPONENT_TYPE_FLOAT16_KHR
						&& props[i].cType == vk::ComponentTypeKHR.COMPONENT_TYPE_FLOAT32_KHR
						&& props[i].resultType == vk::ComponentTypeKHR.COMPONENT_TYPE_FLOAT32_KHR
						&& props[i].scope == vk::ScopeKHR.SCOPE_SUBGROUP_KHR)
					{
						coopmat_supported = true;
						break;
					}
				}
			}
		}
		if (coopmat_supported)
		{
			io::printfn("  cooperative matrix: 16x16x16 fp16->fp32 supported");
			device_extensions.push_all({ coopmat_exts[0], coopmat_exts[1] });
		}
		else
		{
			io::printfn("  cooperative matrix: not supported (using tiled GEMM fallback)");
		}
	}

	// Feature chain (built bottom-up). Coop matrix tail node is conditional.
	PhysicalDeviceCooperativeMatrixFeaturesKHR coopmatFeatures = {
		.sType = vk::StructureType.STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
		.pNext = null,
		.cooperativeMatrix = vk::TRUE,
		.cooperativeMatrixRobustBufferAccess = 0,
	};

	PhysicalDeviceVulkan13Features vulkan13Features = {
		.sType = vk::StructureType.STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.pNext = coopmat_supported ? (void*)&coopmatFeatures : null,
		.synchronization2 = vk::TRUE,
		.maintenance4 = vk::TRUE,
	};

	PhysicalDeviceVulkan12Features vulkan12Features = {
		.sType = vk::StructureType.STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = &vulkan13Features,
	};

	PhysicalDeviceVulkan11Features vulkan11Features = {
		.sType = vk::StructureType.STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		.pNext = &vulkan12Features,
	};

	PhysicalDeviceFeatures2 device_features = {
		.sType = vk::StructureType.STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &vulkan11Features,
	};

	vk::getPhysicalDeviceFeatures2(physical_device, &device_features);

	// Force-enable the bits we need for the coopmat path even if the GPU
	// would have left them disabled.
	if (coopmat_supported)
	{
		vulkan11Features.storageBuffer16BitAccess = vk::TRUE;
		vulkan12Features.shaderFloat16 = vk::TRUE;
		coopmatFeatures.cooperativeMatrix = vk::TRUE;
	}

	DeviceQueueCreateInfo computeInfo = vk::deviceQueueCreateInfo().setQueueFamilyIndex(compute_queue.index)
		.setQueuePriorities({ 1.0 });

	device = vk::deviceCreateInfo().setQueueCreateInfos({ computeInfo })
		.setEnabledExtensionNames(device_extensions.array_view())
		.setNext(&device_features)
		.build(physical_device)!!;

	compute_queue.handle = device.getQueue(compute_queue.index);

	CommandPool command_pool          = device.createCommandPool(queue: compute_queue.index)!;
	// Only the handle list is ours; the command buffers themselves belong to
	// the pool and go when it does.
	CommandBufferList command_buffers = device.createCommandBuffers(command_pool, 1)!;
	defer command_buffers.free();

	Fence compute_fence = vk::fenceCreateInfo().setFlags(FENCE_CREATE_SIGNALED_BIT).build(device)!!;

	// 256MB page size for model weights
	vk::MemoryAllocator allocator = (vk::MemoryAllocator) {
		.device = device,
		.pdevice = physical_device,
		.queue = compute_queue,
		.default_page_size = 256_000_000,
		.flags = MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT | MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
	}.init()!;

	return (DeviceContext) {
		.instance = instance,
		.device = device,
		.pdevice = physical_device,
		.command_pool = command_pool,
		.command_buffer = command_buffers[0],
		.compute_queue = compute_queue,
		.allocator = allocator,
		.compute_fence = compute_fence,
		.coopmat_supported = coopmat_supported,
	};
}

fn void DeviceContext.free(&self)
{
	self.device.waitIdle();
	self.command_pool.free(self.device);
	vk::destroyFence(self.device, self.compute_fence, null);
	self.allocator.free();
	self.device.free();
	self.instance.free();
}
