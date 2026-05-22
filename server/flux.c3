module server::flux_server;

import std::io;
import std::core::mem;
import std::encoding::json;
import std::encoding::base64;
import std::collections::object;
import std::net;
import image;
import image::png;
import llm;
import flux;
import server;
import server::http;

// Single global state — there is only ever one Flux server in a process.
// HandlerFn is a plain function pointer (no closure capture), so we stash
// state here at run_server() time.
flux::FluxServerState* g_state = null;

fn void? run_server(
    flux::FluxConfig* cfg,
    String bind_addr,
    int port,
    bool keep_dit
)
{
    flux::FluxServerState state = flux::init_flux_server(
        cfg.model_path,
        cfg.text_model_path,
        cfg.vae_path,
        keep_dit
    )!!;
    defer state.free();
    g_state = &state;
    defer g_state = null;

    server::serve(bind_addr, (uint)port, &handle_request)!!;
}

fn void respond_error(net::Socket* sock, int status, String message)
{
    char[256] buf;
    String prefix = "{\"error\":\"";
    String suffix = "\"}";
    usz off = 0;
    for (usz i = 0; i < prefix.len; i++) buf[off++] = prefix[i];
    for (usz i = 0; i < message.len && off + suffix.len < buf.len; i++) buf[off++] = message[i];
    for (usz i = 0; i < suffix.len; i++) buf[off++] = suffix[i];
    http::write_response(sock, status, "application/json", buf[0:off]);
}

fn void handle_request(net::Socket* sock, http::HttpRequest* req)
{
    if (req.method != "POST" || req.path != "/sdapi/v1/img2img")
    {
        respond_error(sock, 404, "only POST /sdapi/v1/img2img is supported");
        return;
    }
    if (g_state == null)
    {
        respond_error(sock, 500, "server not initialised");
        return;
    }

    Object* body;
    if (try parsed = json::parse(mem, (String)req.body))
    {
        body = parsed;
    }
    else
    {
        respond_error(sock, 400, "invalid json body");
        return;
    }
    defer body.free();

    String prompt;
    if (try v = body.get_string("prompt"))
    {
        prompt = v;
    }
    else
    {
        respond_error(sock, 400, "missing 'prompt'");
        return;
    }

    flux::FluxGenerateRequest greq = {
        .prompt        = prompt,
        .edit_mode     = "",
        .input_path    = "",
        .input_image   = null,
        .image_size    = 1024,
        .image_width   = 0,
        .image_height  = 0,
        .num_steps     = 4,
        .strength      = 0.6f,
        .seed          = 42,
        .lora_path     = "",
        .lora_strength = 1.0f,
    };
    if (try v = body.get_uint("steps"))  greq.num_steps    = (uint)v;
    if (try v = body.get_uint("seed"))   greq.seed         = (uint)v;
    if (try v = body.get_uint("width"))  greq.image_width  = (uint)v;
    if (try v = body.get_uint("height")) greq.image_height = (uint)v;
    if (try v = body.get_string("lora")) greq.lora_path    = v;
    if (try v = body.get_float("lora_strength")) greq.lora_strength = (float)v;

    image::Image input_img;
    bool have_input = false;
    if (try arr = body.get("init_images"))
    {
        if (arr.is_array() && arr.get_len() > 0)
        {
            String b64;
            if (try s = arr.get_string_at(0))
            {
                b64 = s;
            }
            else
            {
                b64 = "";
            }
            if (b64.len > 0)
            {
                char[] raw;
                if (try decoded = base64::decode(mem, (char[])b64))
                {
                    raw = decoded;
                }
                else
                {
                    respond_error(sock, 400, "init_images[0] is not valid base64");
                    return;
                }
                defer mem::free(raw);

                if (try img = llm::load_image_from_bytes(raw))
                {
                    input_img        = img;
                    have_input       = true;
                    greq.input_image = &input_img;
                    greq.edit_mode   = "kontext";
                }
                else
                {
                    respond_error(sock, 400, "init_images[0] is not a valid PNG/JPEG");
                    return;
                }
            }
        }
    }
    defer if (have_input) input_img.free();

    image::Image result;
    if (try r = flux::flux_generate(g_state, &greq))
    {
        result = r;
    }
    else
    {
        respond_error(sock, 500, "generation failed");
        return;
    }
    defer result.free();

    char[] png_bytes;
    if (try b = png::save_bytes(&result))
    {
        png_bytes = b;
    }
    else
    {
        respond_error(sock, 500, "png encode failed");
        return;
    }
    defer mem::free(png_bytes);

    String b64_str = base64::encode(mem, png_bytes);
    defer mem::free(b64_str);

    // Build {"data":"<...>"} — the base64 alphabet is JSON-safe, no escaping.
    String prefix = "{\"data\":\"";
    String suffix = "\"}";
    usz total     = prefix.len + b64_str.len + suffix.len;
    char[] resp   = mem::new_array(char, (sz)total);
    usz off       = 0;
    for (usz i = 0; i < prefix.len; i++) resp[off++] = prefix[i];
    for (usz i = 0; i < b64_str.len; i++) resp[off++] = b64_str[i];
    for (usz i = 0; i < suffix.len; i++) resp[off++] = suffix[i];

    http::write_response(sock, 200, "application/json", resp);
    mem::free(resp);
}
