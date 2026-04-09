module llm;

import vk;

// RenderGraph: lightweight wrapper around the dispatch + barrier + submit
// boundary lifecycle. Designed to remove the verbose buffer/size array
// boilerplate at every dispatch site and to insert compute barriers
// automatically between dispatches in the same submission frame.
//
// Submission boundaries (`checkpoint`) remain explicit because they encode
// timeout-avoidance intent that cannot be inferred locally.
//
// Typical use:
//
//     RenderGraph g;
//     g.init(ctx)!!;
//     g.dispatch(&k.silu, { tmp }, &silu_pc, n_groups);
//     g.dispatch(&k.conv, { in, w, b, out }, &conv_pc, conv_groups);
//     g.checkpoint()!!;   // submit + begin a new frame
//     ...
//     g.flush()!!;        // final submit at end of forward pass
//
struct RenderGraph {
    DeviceContext* ctx;
    CommandBuffer cmd;
    bool needs_barrier;  // true after the first op in the current frame
}

// Initialise and start a fresh submission frame.
fn void? RenderGraph.init(&self, DeviceContext* ctx) {
    self.ctx = ctx;
    self.cmd = ctx.command_buffer;
    self.needs_barrier = false;
    begin_compute(self.cmd)!!;
}

// Append a single compute dispatch. Inserts a compute_barrier first if any
// prior op has been recorded in the current frame, then builds the
// buffer/size arrays from the tensor pointers and forwards to dispatch_kernel.
fn void RenderGraph.dispatch(
    &self,
    ComputeKernel* kernel,
    Tensor*[] tensors,
    void* push_constants,
    uint groups_x,
    uint groups_y = 1,
    uint groups_z = 1,
) {
    if (self.needs_barrier) compute_barrier(self.cmd);

    vk::Buffer[MAX_BINDINGS] buffers;
    usz[MAX_BINDINGS] sizes;
    for (uint i = 0; i < tensors.len; i++) {
        buffers[i] = tensors[i].gpu_buffer.buffer;
        sizes[i]   = tensors[i].size_bytes;
    }
    dispatch_kernel(
        self.cmd, kernel,
        buffers[0 .. tensors.len - 1],
        sizes[0 .. tensors.len - 1],
        push_constants, groups_x, groups_y, groups_z,
    );
    self.needs_barrier = true;
}

// Issue a buffer-to-buffer copy as a graph op.
// Inserts a barrier before the copy if needed; the copy is treated like any
// other op for barrier purposes.
fn void RenderGraph.copy(&self, Tensor* src, Tensor* dst, ulong size) {
    if (self.needs_barrier) compute_barrier(self.cmd);
    vk::cmdCopyBuffer(self.cmd, src.gpu_buffer.buffer, dst.gpu_buffer.buffer, 1,
        (vk::BufferCopy[]){{ .srcOffset = 0, .dstOffset = 0, .size = size }});
    self.needs_barrier = true;
}

// Submit the recorded work, wait for completion, and start a new frame.
// Use this where the imperative code calls `submit_and_wait + begin_compute`.
fn void? RenderGraph.checkpoint(&self) {
    submit_and_wait(self.ctx)!!;
    self.needs_barrier = false;
    begin_compute(self.cmd)!!;
}

// Final flush at the end of a forward pass — submit and wait without
// re-opening a new frame. Pair with init() at the next forward pass.
fn void? RenderGraph.flush(&self) {
    submit_and_wait(self.ctx)!!;
    self.needs_barrier = false;
}
