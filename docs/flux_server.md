## Flux server

The Flux plugin can run as an HTTP server. Kernels are compiled once at
startup. With `keep_dit=true`, the 9 GB DiT is loaded once and stays in VRAM,
so a request pays only for text encoding, denoising and the VAE.

### Run

```sh
c3c build llm-runner
./build/llm-runner flux --config flux.json keep_dit=true --server --port 7860
```

The `flux.json` you use for one-shot generation is reused: model paths, LoRAs
and defaults come from it, and each request's fields override them.

| Option               | Default       | Notes |
|----------------------|---------------|-------|
| `--server`           | (off)         | Serve instead of generating once. |
| `--bind`             | `127.0.0.1`   | Bind address. `0.0.0.0` exposes it on the network. |
| `--port`             | `7860`        | TCP port. |
| `keep_dit=true`      | (off)         | Keep the DiT resident across requests. The LoRA set it was merged with is remembered, and a request with a different set reloads it. |

### Endpoint

`POST /sdapi/v1/img2img` or `POST /sdapi/v1/txt2img` with a JSON body:

| Field         | Type     | Notes |
|---------------|----------|-------|
| `prompt`      | string   | |
| `init_images` | string[] | Base64 PNG or JPEG (a `data:` URL prefix is accepted). Present: kontext edit with up to 4 references. |
| `edit_mode`   | string   | `kontext` (default with images) or `img2img`. |
| `steps`, `seed`, `width`, `height`, `strength` | numbers | As in the config. |
| `loras` / `lora` | array or string | Replaces the configured adapters for this request. |

The response is `{"data": "<base64 PNG>", "images": ["<same>"]}`. Errors are a
non-2xx status with `{"error": "..."}`, and the server stays up.

```sh
curl -X POST http://127.0.0.1:7860/sdapi/v1/txt2img \
  -d '{"prompt":"a red cat","seed":42,"width":1024,"height":1024}' \
  | jq -r .data | base64 -d > out.png
```

One request is handled at a time; there is one GPU.
