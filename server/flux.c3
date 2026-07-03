module server::flux_server;

import std::io;
import std::core::mem;
import std::encoding::json;
import std::encoding::base64;
import std::collections::object;
import std::collections::list;
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
flux::FluxConfig*      g_cfg   = null;

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
    g_cfg   = cfg;
    defer g_state = null;
    defer g_cfg   = null;

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
        .input_paths   = {},
        .input_images  = {},
        .image_size    = 1024,
        .image_width   = 0,
        .image_height  = 0,
        .num_steps     = 4,
        .strength      = 0.6f,
        .seed          = 42,
        .loras         = {},
    };
    if (try v = body.get_uint("steps"))  greq.num_steps    = (uint)v;
    if (try v = body.get_uint("seed"))   greq.seed         = (uint)v;
    if (try v = body.get_uint("width"))  greq.image_width  = (uint)v;
    if (try v = body.get_uint("height")) greq.image_height = (uint)v;

    // LoRA selection. A request that names any adapter replaces the server's
    // configured set entirely; otherwise it inherits the config's loras.
    //   "loras": [ "a.safetensors", { "path": "b...", "strength": 0.7 } ]
    //   "lora": "path" (+ optional "lora_strength")   — single-adapter shorthand
    List{flux::FluxLoraSpec} lora_list;
    defer lora_list.free();
    if (try arr = body.get("loras"))
    {
        if (arr.is_array())
        {
            for (sz idx = 0; idx < arr.get_len(); idx++)
            {
                Object* el = arr.get_at(idx);
                if (el.is_string())
                {
                    lora_list.push({ .path = el.s, .strength = 1.0f });
                }
                else if (try p = el.get_string("path"))
                {
                    float s = 1.0f;
                    if (try sv = el.get_float("strength")) s = (float)sv;
                    lora_list.push({ .path = p, .strength = s });
                }
            }
        }
    }
    else if (try v = body.get_string("lora"))
    {
        float s = 1.0f;
        if (try sv = body.get_float("lora_strength")) s = (float)sv;
        lora_list.push({ .path = v, .strength = s });
    }
    else if (g_cfg != null)
    {
        foreach (sp : g_cfg.loras) lora_list.push(sp);
    }
    greq.loras = lora_list.array_view();

    image::Image[] input_imgs;
    image::Image*[] input_ptrs;
    bool have_input = false;
    if (try arr = body.get("init_images"))
    {
        if (arr.is_array())
        {
            sz n = arr.get_len();
            if (n > (sz)flux::FLUX_MAX_REF_IMAGES)
            {
                respond_error(sock, 400, "init_images exceeds max reference image count");
                return;
            }
            if (n > 0)
            {
                input_imgs = mem::new_array(image::Image, n);
                input_ptrs = mem::new_array(image::Image*, n);
                for (sz idx = 0; idx < n; idx++)
                {
                    String b64;
                    if (try s = arr.get_string_at(idx))
                    {
                        b64 = s;
                    }
                    else
                    {
                        b64 = "";
                    }
                    if (b64.len == 0)
                    {
                        for (sz f = 0; f < idx; f++) input_imgs[f].free();
                        mem::free(input_imgs);
                        mem::free(input_ptrs);
                        respond_error(sock, 400, "init_images entry is not a string");
                        return;
                    }
                    char[] raw;
                    if (try decoded = base64::decode(mem, (char[])b64))
                    {
                        raw = decoded;
                    }
                    else
                    {
                        for (sz f = 0; f < idx; f++) input_imgs[f].free();
                        mem::free(input_imgs);
                        mem::free(input_ptrs);
                        respond_error(sock, 400, "init_images entry is not valid base64");
                        return;
                    }
                    defer mem::free(raw);

                    if (try img = llm::load_image_from_bytes(raw))
                    {
                        input_imgs[idx] = img;
                        input_ptrs[idx] = &input_imgs[idx];
                    }
                    else
                    {
                        for (sz f = 0; f < idx; f++) input_imgs[f].free();
                        mem::free(input_imgs);
                        mem::free(input_ptrs);
                        respond_error(sock, 400, "init_images entry is not a valid PNG/JPEG");
                        return;
                    }
                }
                have_input        = true;
                greq.input_images = input_ptrs;
                greq.edit_mode    = "kontext";
            }
        }
    }
    defer if (have_input)
    {
        for (sz idx = 0; idx < input_imgs.len; idx++) input_imgs[idx].free();
        mem::free(input_imgs);
        mem::free(input_ptrs);
    }

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
