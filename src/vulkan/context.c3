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
}

fn DeviceContext? createContext() {
    ZStringList extensions;
    ZStringList device_extensions;
    device_extensions.push_all({"VK_KHR_push_descriptor"});

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

    devices.free();

    PhysicalDeviceVulkan13Features vulkan13Features = {
        .sType = vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = null,
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
