module llm;

import vk;
import std::io;
import std::core::mem;
import std::core::mem::alloc;
import std::core::test;

// The allocator's bookkeeping must not live in the temp arena.
//
// A C3 `List` left zero-initialised has a null allocator, and
// `list_ensure_capacity` resolves null to `tmem`. `vk::MemoryAllocator.pages`
// used to be one of those, so the first `MemoryPage` pushed while a `@pool()`
// was open put the page list's backing array in that pool, and the pool handed
// that space back on the way out.
//
// What made this so hard to pin down is that the pool does not destroy
// anything: it rewinds a mark. The page list keeps reading correctly until
// something else allocates over the rewound region, and then `pages` is full
// of whatever that was. So the crash lands arbitrarily far from the mistake —
// in `new_buffer`'s free-block scan, or in `MemoryAllocator.free` — and looks
// data-dependent. Beam search dying only at eight beams, and the SkinTokens
// VAE loader dying only once it was formatting its 176th tensor name, were
// both this.
//
// Each test therefore scribbles over the arena after the pool closes. Without
// that these pass whether the bug is present or not, which is exactly the
// property that let it survive so long.

// Fill the temp arena from the mark the previous pool released, so anything
// left there is overwritten rather than merely abandoned.
fn void clobber_temp_arena(usz bytes = 64 * 1024)
{
    @pool()
    {
        char[] scribble = alloc::new_array(tmem, char, (sz)bytes);
        foreach (&byte : scribble) *byte = (char)0x5A;
    };
}

fn void round_trip(DeviceContext* ctx, Tensor* tensor, usz count)
{
    float[] data = mem::new_array(float, (sz)count);
    defer mem::free(data);
    foreach (usz i, &value : data) *value = (float)i;

    upload_to_tensor(ctx, tensor, data)!!;
    float[] back = download_tensor(mem, ctx, tensor, count)!!;
    defer mem::free(back);

    assert(back[0] == 0.0f && back[count - 1] == (float)(count - 1),
        "the buffer did not survive a round trip");
}

fn void test_page_list_survives_a_temp_scope() @test
{
    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    // A fresh context has an empty page list, so this allocation is the one
    // that creates the first page and makes the first push into it.
    Tensor inside;
    @pool()
    {
        inside = create_f32_tensor(&ctx, { 1024, 0, 0, 0 }, 1)!!;
    };
    defer inside.free();

    clobber_temp_arena();

    // Reading `pages` is the first thing `new_buffer` does.
    Tensor after = create_f32_tensor(&ctx, { 1024, 0, 0, 0 }, 1)!!;
    defer after.free();

    round_trip(&ctx, &after, 1024);
}

// Freeing is the other side of the same list. `Memory.free` only marks, and
// `sync` is what walks `pending_deletes` and pushes a block onto the page's
// own free list — so both of those are first written from wherever a caller
// happens to release a buffer, which is just as likely to be inside a pool.
fn void test_free_lists_survive_a_temp_scope() @test
{
    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    Tensor keep = create_f32_tensor(&ctx, { 4096, 0, 0, 0 }, 1)!!;
    defer keep.free();

    @pool()
    {
        Tensor scratch = create_f32_tensor(&ctx, { 4096, 0, 0, 0 }, 1)!!;
        scratch.free();
        // Turn the mark into an entry on the page's free list while the pool
        // is still open.
        ctx.allocator.sync();
    };

    clobber_temp_arena();

    // Reusing the hole reads the free list the pool would have taken.
    Tensor reused = create_f32_tensor(&ctx, { 4096, 0, 0, 0 }, 1)!!;
    defer reused.free();

    round_trip(&ctx, &reused, 4096);
}

// And the whole thing torn down after a pool touched it: `MemoryAllocator.free`
// walks `pages` to unmap and release each one, which is where the beam search
// actually died.
fn void test_allocator_teardown_survives_a_temp_scope() @test
{
    DeviceContext ctx = createContext()!!;

    Tensor a;
    Tensor b;
    @pool()
    {
        a = create_f32_tensor(&ctx, { 2048, 0, 0, 0 }, 1)!!;
        b = create_f32_tensor(&ctx, { 2048, 0, 0, 0 }, 1)!!;
        b.free();
        ctx.allocator.sync();
    };

    clobber_temp_arena();

    a.free();
    ctx.free();
}
