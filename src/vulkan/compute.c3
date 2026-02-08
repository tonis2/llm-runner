module llm;

import vk;
import std::io;
import std::core::mem;

const MAX_BINDINGS = 8;

struct ComputeKernel {
    Pipeline pipeline;
    PipelineLayout pipeline_layout;
    DescriptorSetLayout desc_layout;
    uint num_buffers;
    uint push_constant_size;
}

fn ComputeKernel? create_kernel(
    DeviceContext* ctx,
    char[] spirv,
    uint num_storage_buffers,
    uint push_constant_size = 0
) {
    Device device = ctx.device;

    // Create shader module
    ShaderModule shader_module = vk::shaderModuleCreateInfo()
        .setCodeSize(spirv.len)
        .setCode((uint*)&spirv[0])
        .build(device)!!;

    // Create descriptor set layout with push descriptor flag
    DescriptorSetLayoutBinding[MAX_BINDINGS] bindings;
    for (uint i = 0; i < num_storage_buffers; i++) {
        bindings[i] = {
            .binding = i,
            .descriptorType = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = vk::SHADER_STAGE_COMPUTE_BIT,
        };
    }

    DescriptorSetLayout desc_layout = vk::descriptorSetLayoutCreateInfo()
        .setFlags(vk::DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR)
        .setBindings(bindings[0 .. num_storage_buffers - 1])
        .build(device)!!;

    // Create pipeline layout
    PipelineLayout pipeline_layout;
    if (push_constant_size > 0) {
        PushConstantRange pc_range = {
            .stageFlags = vk::SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = push_constant_size,
        };
        pipeline_layout = vk::pipelineLayoutCreateInfo()
            .setSetLayouts({ desc_layout })
            .setPushConstantRanges({ pc_range })
            .build(device)!!;
    } else {
        pipeline_layout = vk::pipelineLayoutCreateInfo()
            .setSetLayouts({ desc_layout })
            .build(device)!!;
    }

    // Create compute pipeline
    PipelineShaderStageCreateInfo stage = vk::pipelineShaderStageCreateInfo()
        .setStage(vk::SHADER_STAGE_COMPUTE_BIT)
        .setMod(shader_module)
        .setName("main");

        // .setName("vertexMain"),

    Pipeline pipeline = vk::computePipelineCreateInfo()
        .setStage(stage)
        .setLayout(pipeline_layout)
        .build(device)!!;

    // Shader module can be destroyed after pipeline creation
    shader_module.free(device);

    return {
        .pipeline = pipeline,
        .pipeline_layout = pipeline_layout,
        .desc_layout = desc_layout,
        .num_buffers = num_storage_buffers,
        .push_constant_size = push_constant_size,
    };
}

fn void ComputeKernel.free(&self, Device device) {
    self.pipeline.free(device);
    self.pipeline_layout.free(device);
    self.desc_layout.free(device);
}

fn void dispatch_kernel(
    CommandBuffer cmd,
    ComputeKernel* kernel,
    vk::Buffer[] buffers,
    usz[] sizes,
    void* push_constants,
    uint groups_x,
    uint groups_y = 1,
    uint groups_z = 1
) {
    vk::cmdBindPipeline(cmd, vk::PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline);

    // Build write descriptor sets for push descriptors
    DescriptorBufferInfo[MAX_BINDINGS] buffer_infos;
    WriteDescriptorSet[MAX_BINDINGS] writes;

    for (uint i = 0; i < kernel.num_buffers; i++) {
        buffer_infos[i] = {
            .buffer = buffers[i],
            .offset = 0,
            .range = sizes[i],
        };
        writes[i] = vk::writeDescriptorSet()
            .setDstBinding(i)
            .setDescriptorType(vk::DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .setBufferInfo(buffer_infos[i .. i]);
    }

    vk::cmdPushDescriptorSetKHR(
        cmd,
        vk::PIPELINE_BIND_POINT_COMPUTE,
        kernel.pipeline_layout,
        0,
        kernel.num_buffers,
        &writes[0]
    );

    if (kernel.push_constant_size > 0 && push_constants != null) {
        vk::cmdPushConstants(
            cmd,
            kernel.pipeline_layout,
            vk::SHADER_STAGE_COMPUTE_BIT,
            0,
            kernel.push_constant_size,
            push_constants
        );
    }

    vk::cmdDispatch(cmd, groups_x, groups_y, groups_z);
}

fn void compute_barrier(CommandBuffer cmd) {
    MemoryBarrier2 barrier = vk::memoryBarrier2()
        .setSrcStageMask(vk::PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
        .setSrcAccessMask(vk::ACCESS_2_SHADER_WRITE_BIT)
        .setDstStageMask(vk::PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
        .setDstAccessMask(vk::ACCESS_2_SHADER_READ_BIT | vk::ACCESS_2_SHADER_WRITE_BIT);

    vk::cmdPipelineBarrier2(cmd, &&vk::dependencyInfo().setMemoryBarriers({ barrier }));
}

fn void? begin_compute(CommandBuffer cmd) {
    vk::resetCommandBuffer(cmd, 0)!!;
    vk::beginCommandBuffer(cmd, &&vk::commandBufferBeginInfo())!!;
}

fn void? submit_and_wait(DeviceContext* ctx) {
    CommandBuffer cmd = ctx.command_buffer;
    vk::endCommandBuffer(cmd)!!;

    ctx.compute_fence.wait(ctx.device)!!;
    ctx.compute_fence.reset(ctx.device)!!;

    CommandBufferSubmitInfo cmd_info = vk::commandBufferSubmitInfo()
        .setCommandBuffer(cmd);

    SubmitInfo2 submit = vk::submitInfo2()
        .setCommandBufferInfos({ cmd_info });

    vk::queueSubmit2(ctx.compute_queue.handle, 1, &submit, ctx.compute_fence)!!;

    ctx.compute_fence.wait(ctx.device)!!;
}
