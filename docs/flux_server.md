## Flux server

The flux example can run as an HTTP server on `/sdapi/v1/img2img`. The DiT
GGUF, VAE safetensors, and text-encoder GGUF are opened and parsed once at
startup; subsequent requests skip Vulkan setup, kernel compilation, and GGUF
parsing.

### Build

```sh
c3c build flux
```

### Run

```sh
./build/flux --server --config flux.json
```

The same `flux.json` you use for one-shot generation is reused — the server
takes its model paths from there. `--prompt` in the config is ignored in
server mode (the prompt arrives per request).

Server-only flags:

| Flag         | Default       | Notes                                                                                          |
|--------------|---------------|------------------------------------------------------------------------------------------------|
| `--server`   | (off)         | Switches the binary into HTTP server mode.                                                     |
| `--bind`     | `127.0.0.1`   | Bind address. Use `0.0.0.0` to expose on the network.                                          |
| `--port`     | `7860`        | TCP port.                                                                                      |
| `--keep-dit` | (off)         | Keep DiT weights (~9 GB) resident across requests. Faster, but borderline OOM on 16 GB at 1024².|

### Endpoint

`POST /sdapi/v1/img2img` with a JSON body. Any other method or path returns
`404`.

Supported fields:

| Field         | Type           | Default | Notes                                                              |
|---------------|----------------|---------|--------------------------------------------------------------------|
| `prompt`      | string         | —       | Required.                                                          |
| `init_images` | string[]       | `[]`    | Optional. If non-empty, only `[0]` is used and mode becomes kontext. |
| `steps`       | uint           | `4`     | Klein is distilled to 4 steps.                                     |
| `seed`        | uint           | `42`    |                                                                    |
| `width`       | uint           | `1024`  | Rounded up to a multiple of 16.                                    |
| `height`      | uint           | `1024`  | Rounded up to a multiple of 16.                                    |

`init_images[0]` is a standard base64-encoded PNG or JPEG (no `data:` URL
prefix needed; both PNG `89 50 4E 47` and JPEG `FF D8` magic bytes are
auto-detected).

### Response

```json
{ "data": "<base64-encoded PNG of the generated image>" }
```

On error (bad JSON, missing prompt, generation failure, etc.) the server
returns a non-2xx status with `{"error": "..."}` and stays alive for the
next request.

### Examples

**txt2img:**

```sh
curl -X POST http://127.0.0.1:7860/sdapi/v1/img2img \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red cat","steps":4,"seed":42,"width":1024,"height":1024}' \
  | jq -r .data | base64 -d > out.png
```

**kontext (image-conditioned edit):**

```sh
IMG=$(base64 -w0 input.png)
curl -X POST http://127.0.0.1:7860/sdapi/v1/img2img \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"make it orange\",\"init_images\":[\"$IMG\"],\"steps\":4,\"seed\":42,\"width\":1024,\"height\":1024}" \
  | jq -r .data | base64 -d > edited.png
```

### Concurrency

The server is single-threaded and processes one request at a time. There's
only one GPU; queueing diffusion requests inside the process gives the same
throughput as parallel handlers and avoids VRAM contention.
