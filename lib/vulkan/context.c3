module llm;

import vk;
import std::math;
import std::io;
import std::collections::list;
import std::collections::map;

faultdef NO_SUITABLE_DEVICE, FILE_NOT_FOUND, COMPUTE_ERROR;

struct DeviceContext {
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

fn DeviceContext? createContext() {
    ZStringList extensions;
    ZStringList device_extensions;
    device_extensions.push_all({"VK_KHR_push_descriptor"});
    // Phase 3: cooperative matrix tensor-core path. Both extensions are
    // optional — checked at runtime, falls back to the tiled GEMM path if
    // missing or if the right matrix configuration isn't supported.
    ZString[2] coopmat_exts = {"VK_KHR_cooperative_matrix", "VK_KHR_shader_float16_int8"};

    Instance instance;
    PhysicalDevice physical_device;
    DeviceQueue compute_queue;
    Device device;

    vk::checkInstanceExtensionSupport(extensions)!!;

    ApplicationInfo info = {
        .pApplicationName = "LLM Runner",
        .pEngineName = "LLM Engine",
        .applicationVersion = vk::@makeApiVersion(0,1,0,0),
        .engineVersion = vk::@makeApiVersion(0,1,0,0),
        .apiVersion = vk::@makeApiVersion(0,1,3,0)
    };

    InstanceCreateInfo instanceInfo = vk::instanceCreateInfo()
        .setApplicationInfo(&info)
        .setFlags(env::os_is_darwin() ? vk::INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0);

    if (catch err = vk::createInstance(&instanceInfo, null, &instance)) {
        io::printfn("Instance creation error");
        return NO_SUITABLE_DEVICE~;
    }

    vk::loadExtensions(instance);

    DeviceList devices = instance.getDevices()!!;

    uint best_score = 0;
    PhysicalDeviceProperties device_props;

    foreach (usz i, entry: devices) {
        bool supports_extensions = entry.checkExtensionSupport(device_extensions)!!;
        QueuePropertiesList queue_properties = entry.getQueueProperties()!!;

        int queue_index = queue_properties.find(fn (usz index, QueueFamilyProperties props) =>
            (QueueFlagBits)props.queueFlags & vk::QUEUE_COMPUTE_BIT != 0 && (QueueFlagBits)props.queueFlags & vk::QUEUE_TRANSFER_BIT);

        PhysicalDeviceProperties properties;
        vk::getPhysicalDeviceProperties(entry, &properties);

        if (queue_index >= 0 && supports_extensions && properties.vendorID != 0) {
            uint score = 0;
            if (properties.deviceType == 1) score += 100;
            if (properties.deviceType == 2) score += 200;

            if (score > best_score) {
                best_score = score;
                physical_device = entry;
                compute_queue.index = queue_index;
                device_props = properties;
            }
        }
    }

    if (physical_device == null) {
        io::printfn("Found no suitable device");
        return NO_SUITABLE_DEVICE~;
    }

    io::printfn("Device selected: %s", (ZString)device_props.deviceName[0..]);

    PhysicalDeviceLimits limits = device_props.limits;
    io::printfn("  maxComputeWorkGroupCount: (%d, %d, %d)", limits.maxComputeWorkGroupCount[0], limits.maxComputeWorkGroupCount[1], limits.maxComputeWorkGroupCount[2]);
    io::printfn("  maxComputeWorkGroupSize: (%d, %d, %d)", limits.maxComputeWorkGroupSize[0], limits.maxComputeWorkGroupSize[1], limits.maxComputeWorkGroupSize[2]);

    devices.free();

    // Query the driver ID so we can blacklist drivers with broken cooperative
    // matrix support. Mesa RADV and Mesa ANV both produce wrong results from
    // OpCooperativeMatrixStoreKHR — see freedesktop Mesa issue #10847 (open as
    // of 2025), and the ncnn project applies the same blacklist (Tencent/ncnn
    // PR #5739). The Phase 3 path is gated on this check, so the program runs
    // the existing tiled GEMM on broken drivers and the coopmat shader
    // entirely on NVIDIA / fixed Mesa.
    PhysicalDeviceDriverProperties driverProps = {
        .sType = vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
        .pNext = null,
    };
    PhysicalDeviceProperties2 props2 = {
        .sType = vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &driverProps,
    };
    vk::getPhysicalDeviceProperties2(physical_device, &props2);
    bool driver_blacklisted_for_coopmat =
        driverProps.driverID == vk::DRIVER_ID_MESA_RADV
        || driverProps.driverID == vk::DRIVER_ID_INTEL_OPEN_SOURCE_MESA;

    // Probe for cooperative matrix support: device must advertise the extension
    // AND expose a 16x16x16 fp16xfp16->fp32 subgroup-scope configuration.
    bool coopmat_supported = false;
    if (driver_blacklisted_for_coopmat) {
        io::printfn("  cooperative matrix: blacklisted on this driver (Mesa #10847 — wrong results)");
    } else {
        ZStringList probe;
        probe.push_all({"VK_KHR_cooperative_matrix"});
        bool ext_ok = false;
        if (try b = physical_device.checkExtensionSupport(probe)) ext_ok = b;
        probe.free();
        if (ext_ok) {
            uint cnt = 0;
            vk::getPhysicalDeviceCooperativeMatrixPropertiesKHR(physical_device, &cnt, null)!!;
            if (cnt > 0 && cnt <= 64) {
                CooperativeMatrixPropertiesKHR[64] props;
                for (uint i = 0; i < cnt; i++) {
                    props[i].sType = vk::STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                }
                vk::getPhysicalDeviceCooperativeMatrixPropertiesKHR(physical_device, &cnt, &props[0])!!;
                for (uint i = 0; i < cnt; i++) {
                    if (props[i].mSize == 16 && props[i].nSize == 16 && props[i].kSize == 16
                        && props[i].aType == vk::COMPONENT_TYPE_FLOAT16_KHR
                        && props[i].bType == vk::COMPONENT_TYPE_FLOAT16_KHR
                        && props[i].cType == vk::COMPONENT_TYPE_FLOAT32_KHR
                        && props[i].resultType == vk::COMPONENT_TYPE_FLOAT32_KHR
                        && props[i].scope == vk::SCOPE_SUBGROUP_KHR) {
                        coopmat_supported = true;
                        break;
                    }
                }
            }
        }
        if (coopmat_supported) {
            io::printfn("  cooperative matrix: 16x16x16 fp16->fp32 supported");
            device_extensions.push_all({coopmat_exts[0], coopmat_exts[1]});
        } else {
            io::printfn("  cooperative matrix: not supported (using tiled GEMM fallback)");
        }
    }

    // Feature chain (built bottom-up). Coop matrix tail node is conditional.
    PhysicalDeviceCooperativeMatrixFeaturesKHR coopmatFeatures = {
        .sType = vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .pNext = null,
        .cooperativeMatrix = vk::TRUE,
        .cooperativeMatrixRobustBufferAccess = 0,
    };

    PhysicalDeviceVulkan13Features vulkan13Features = {
        .sType = vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = coopmat_supported ? (void*)&coopmatFeatures : null,
        .synchronization2 = vk::TRUE,
        .maintenance4 = vk::TRUE,
    };

    PhysicalDeviceVulkan12Features vulkan12Features = {
        .sType = vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &vulkan13Features,
    };

    PhysicalDeviceVulkan11Features vulkan11Features = {
        .sType = vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &vulkan12Features,
    };

    PhysicalDeviceFeatures2 device_features = {
        .sType = vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &vulkan11Features,
    };

    vk::getPhysicalDeviceFeatures2(physical_device, &device_features);

    // Force-enable the bits we need for the coopmat path even if the GPU
    // would have left them disabled.
    if (coopmat_supported) {
        vulkan11Features.storageBuffer16BitAccess = vk::TRUE;
        vulkan12Features.shaderFloat16 = vk::TRUE;
        coopmatFeatures.cooperativeMatrix = vk::TRUE;
    }

    DeviceQueueCreateInfo computeInfo = vk::deviceQueueCreateInfo()
        .setQueueFamilyIndex(compute_queue.index)
        .setQueuePriorities({1.0});

    device = vk::deviceCreateInfo()
        .setQueueCreateInfos({ computeInfo })
        .setEnabledExtensionNames(device_extensions.array_view())
        .setNext(&device_features)
        .build(physical_device)!!;

    compute_queue.handle = device.getQueue(compute_queue.index);

    CommandPool command_pool = device.createCommandPool(queue: compute_queue.index)!;
    CommandBufferList command_buffers = device.createCommandBuffers(command_pool, 1)!;

    Fence compute_fence = vk::fenceCreateInfo()
        .setFlags(vk::FENCE_CREATE_SIGNALED_BIT)
        .build(device)!!;

    // 256MB page size for model weights
    DeviceQueue* queue_ptr = mem::alloc(DeviceQueue);
    *queue_ptr = compute_queue;

    // 256MB page size for model weights
    vk::MemoryAllocator allocator = (vk::MemoryAllocator){
        .device = device,
        .pdevice = physical_device,
        .queue = queue_ptr,
        .default_page_size = 256_000_000,
        .flags = vk::MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT | vk::MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
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

fn void DeviceContext.free(&self) {
    self.device.waitIdle();
    self.command_pool.free(self.device);
    vk::destroyFence(self.device, self.compute_fence, null);
    mem::free(self.allocator.queue);
    self.allocator.free();
    self.device.free();
    self.instance.free();
}
