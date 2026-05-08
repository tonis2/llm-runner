module flux;

import vk;
import llm;
import llm::pipelines;
import llm_text;
import zimage;
import image;
import image::png;
import std::io;
import std::io::file;
import std::io::file::mmap;
import std::math;
import std::core::mem;

// Flux 2 Klein 9B pipeline: txt2img, img2img, and Kontext-style image editing.
//
// Flow:
//   1. Encode text with Qwen3-8B → extract 3 hidden states → [seq_len, 12288]
//   2. (img2img/kontext) Encode reference image with the Flux VAE encoder
//   3. Load Flux DiT from GGUF
//   4. Denoise with flow matching Euler steps. For kontext the encoded ref is
//      appended to the image-stream token sequence with a T-axis RoPE offset.
//   5. Pixel shuffle + VAE decode → output image
//   6. Save

struct FluxConfig
{
	String model_path;       // Flux DiT GGUF
	String text_model_path;  // Qwen3-8B GGUF (text encoder)
	String vae_path;         // Flux VAE safetensors
	String prompt;
	String input_path;       // Input image for img2img / kontext
	String output_path;
	String edit_mode;        // "" (txt2img, default) | "img2img" | "kontext"
	uint image_size;         // Square shortcut. If image_width/height are 0, both are derived from this.
	uint image_width;        // Output width  (0 = use image_size)
	uint image_height;       // Output height (0 = use image_size)
	uint num_steps;
	float strength;          // img2img only: noise fraction (0=keep input, 1=full re-roll)
	uint seed;
	String lora_path;        // Optional LoRA safetensors (empty = no LoRA)
	float lora_strength;     // LoRA strength multiplier
}

fn FluxConfig flux_config_defaults()
{
	return {
		.output_path = "output.png",
		.image_size = 1024,
		.image_width = 0,
		.image_height = 0,
		.num_steps = 4,        // Klein is distilled to 4 steps
		.strength = 0.6f,
		.seed = 42,
		.lora_path = "",
		.lora_strength = 1.0f,
	};
}

// --- Pipeline Implementation ---
//
// One-shot CLI entry. Internally goes through the same init_flux_server +
// flux_generate path the server uses, with keep_dit=false so the GPU is
// completely cleaned up by the time this returns.
fn void? run_flux_pipeline(FluxConfig* config)
{
	FluxServerState state = init_flux_server(
		config.model_path,
		config.text_model_path,
		config.vae_path,
		false
	)!!;
	defer state.free();

	FluxGenerateRequest req = {
		.prompt        = config.prompt,
		.edit_mode     = config.edit_mode,
		.input_path    = config.input_path,
		.input_image   = null,
		.image_width   = config.image_width,
		.image_height  = config.image_height,
		.image_size    = config.image_size,
		.num_steps     = config.num_steps,
		.strength      = config.strength,
		.seed          = config.seed,
		.lora_path     = config.lora_path,
		.lora_strength = config.lora_strength,
	};

	image::Image result = flux_generate(&state, &req)!!;
	defer result.free();

	png::save_file(&result, config.output_path)!!;
	io::printfn("  Saved to %s (%dx%d)", config.output_path, result.width, result.height);
}


// --- Pixel shuffle (depth-to-space) on CPU ---
// Input:  [C, H, W] where C = out_ch * r * r
// Output: [out_ch, H*r, W*r]
fn void pixel_shuffle_cpu(float[] input, float[] output, uint in_channels, uint in_h, uint in_w, uint r)
{
	uint out_ch = in_channels / (r * r);
	uint out_h  = in_h * r;
	uint out_w  = in_w * r;

	for (uint c = 0; c < out_ch; c++)
	{
		for (uint oh = 0; oh < out_h; oh++)
		{
			for (uint ow = 0; ow < out_w; ow++)
			{
				// Map output (c, oh, ow) back to input channel
				uint sub_h = oh % r;
				uint sub_w = ow % r;
				uint in_c  = c * r * r + sub_h * r + sub_w;
				uint ih    = oh / r;
				uint iw    = ow / r;

				uint in_idx  = in_c * in_h * in_w + ih * in_w + iw;
				uint out_idx = c * out_h * out_w + oh * out_w + ow;
				output[out_idx] = input[in_idx];
			}
		}
	}
}

// --- Pixel unshuffle (space-to-depth) on CPU ---
// Inverse of pixel_shuffle_cpu. Input [in_ch, H*r, W*r] -> [in_ch * r * r, H, W].
fn void pixel_unshuffle_cpu(float[] input, float[] output, uint in_channels, uint in_h, uint in_w, uint r)
{
	uint out_ch = in_channels * r * r;
	uint out_h  = in_h / r;
	uint out_w  = in_w / r;

	for (uint c = 0; c < in_channels; c++)
	{
		for (uint ih = 0; ih < in_h; ih++)
		{
			for (uint iw = 0; iw < in_w; iw++)
			{
				uint sub_h = ih % r;
				uint sub_w = iw % r;
				uint dst_c = c * r * r + sub_h * r + sub_w;
				uint dst_h = ih / r;
				uint dst_w = iw / r;

				uint src_idx = c * in_h * in_w + ih * in_w + iw;
				uint dst_idx = dst_c * out_h * out_w + dst_h * out_w + dst_w;
				output[dst_idx] = input[src_idx];
			}
		}
	}
}

// --- Encoder CPU post-processing ---
// Takes the raw encoder output [64, enc_h, enc_w] and produces the DiT-input
// latent [128, lat_h, lat_w] in noise space. Mirrors the inverse of the decoder
// pre-processing (BN-inverse + pixel_shuffle + post_quant_conv) at lines 345-413.
//
// Steps:
//   1. quant_conv (1x1, 64 -> 64)
//   2. Split: take first 32 channels (mean), drop logvar
//   3. pixel_unshuffle (32 ch -> 128 ch, halves spatial): [32, enc_h, enc_w] -> [128, lat_h, lat_w]
//   4. BN-forward: x = (x - mean) / sqrt(var + eps)  (in 128-ch space)
fn void? encoder_postprocess_to_dit_latent(
	llm::SafetensorsFile* vae_sf,
	float[] enc_out,         // [64, enc_h, enc_w] from encoder forward
	float[] dit_latent_out,  // [128, lat_h, lat_w] (caller-allocated, lat_h = enc_h / 2)
	uint enc_h, uint enc_w,
	uint lat_h, uint lat_w
)
{
	uint enc_spatial = enc_h * enc_w;

	// Step 1: quant_conv (1x1 [64, 64, 1, 1])
	float[64 * 64] qc_weight;
	float[64] qc_bias;
	if (try info = vae_sf.find_tensor("quant_conv.weight"))
	{
		mem::copy(&qc_weight, (float*)(vae_sf.data_base + (usz)info.data_start), 64 * 64 * 4);
	}
	else
	{
		io::printfn("  ERROR: quant_conv.weight missing from VAE safetensors");
		return llm::COMPUTE_ERROR~;
	}
	if (try info = vae_sf.find_tensor("quant_conv.bias"))
	{
		mem::copy(&qc_bias, (float*)(vae_sf.data_base + (usz)info.data_start), 64 * 4);
	}

	float[] qc_out = mem::new_array(float, (usz)64 * enc_spatial);
	defer mem::free(qc_out);
	float[64] tmp;
	for (uint s = 0; s < enc_spatial; s++)
	{
		for (uint c = 0; c < 64; c++) tmp[c] = enc_out[c * enc_spatial + s];
		for (uint oc = 0; oc < 64; oc++)
		{
			float v = qc_bias[oc];
			for (uint ic = 0; ic < 64; ic++) v += qc_weight[oc * 64 + ic] * tmp[ic];
			qc_out[oc * enc_spatial + s] = v;
		}
	}

	// Step 2: take first 32 channels (mean) — deterministic encode (no logvar sampling)
	float[] mean_only = mem::new_array(float, (usz)32 * enc_spatial);
	defer mem::free(mean_only);
	mem::copy(mean_only.ptr, qc_out.ptr, (usz)32 * enc_spatial * 4);

	// Step 3: pixel_unshuffle 32 ch -> 128 ch (spatial halved): [32, enc_h, enc_w] -> [128, lat_h, lat_w]
	pixel_unshuffle_cpu(mean_only, dit_latent_out, 32, enc_h, enc_w, 2);

	// Step 4: BN-forward (inverse of decoder's BN-inverse at lines 345-368)
	float[128] bn_mean;
	float[128] bn_var;
	if (try info = vae_sf.find_tensor("bn.running_mean"))
	{
		mem::copy(&bn_mean, (float*)(vae_sf.data_base + (usz)info.data_start), 128 * 4);
	}
	if (try info = vae_sf.find_tensor("bn.running_var"))
	{
		mem::copy(&bn_var, (float*)(vae_sf.data_base + (usz)info.data_start), 128 * 4);
	}
	float eps = 1e-5f;
	uint lat_spatial = lat_h * lat_w;
	for (uint c = 0; c < 128; c++)
	{
		float std = math::sqrt(bn_var[c] + eps);
		for (uint j = 0; j < lat_spatial; j++)
		{
			uint idx = c * lat_spatial + j;
			dit_latent_out[idx] = (dit_latent_out[idx] - bn_mean[c]) / std;
		}
	}
}

// --- Run the VAE encoder on an input image and return the DiT-input latent [128, lat_h, lat_w] ---
// Path-based variant: loads the image from disk, then delegates to the Image-based variant below.
// Caller frees the returned array.
fn float[]? encode_image_to_dit_latent(
	llm::DeviceContext* ctx,
	String input_path,
	String vae_path,
	uint img_h, uint img_w,
	uint lat_h, uint lat_w
)
{
	io::printfn("  Loading input: %s", input_path);
	image::Image src_img = llm::load_image(input_path)!!;
	defer src_img.free();
	return encode_image_object_to_dit_latent(ctx, &src_img, vae_path, img_h, img_w, lat_h, lat_w);
}

// --- Image-based variant: encode an in-memory image::Image without disk I/O. ---
// Used by the HTTP server (init_images decoded from base64) so we never touch the filesystem.
fn float[]? encode_image_object_to_dit_latent(
	llm::DeviceContext* ctx,
	image::Image* src_img,
	String vae_path,
	uint img_h, uint img_w,
	uint lat_h, uint lat_w
)
{
	// 1. Preprocess input image (resize + center-crop + normalize to [-1, 1])
	image::Image preprocessed = llm::resize_center_crop(src_img, img_w, img_h)!!;
	defer preprocessed.free();
	float[] input_chw = llm::image_to_float_nchw_signed(&preprocessed)!!;
	defer mem::free(input_chw);
	io::printfn("  Preprocessed: %dx%d, range [-1, 1]", img_w, img_h);

	// 2. Open VAE safetensors + create encoder
	llm::SafetensorsFile enc_sf = llm::safetensors_open(vae_path)!!;
	defer enc_sf.close();

	zimage::VaeKernels enc_kernels = zimage::create_vae_kernels(ctx)!!;
	defer enc_kernels.free(ctx.device);

	zimage::FluxVaeEncoder enc = zimage::load_flux_vae_encoder(ctx, &enc_sf, &enc_kernels)!!;
	defer enc.free();

	uint enc_alloc_size = img_h > img_w ? img_h : img_w;
	zimage::FluxVaeActivations enc_acts = zimage::allocate_flux_vae_activations(ctx, enc_alloc_size)!!;
	defer enc_acts.free();

	// 3. Upload preprocessed image [3, H, W] to enc_acts.buf_a
	{
		usz img_bytes = (usz)3 * img_h * img_w * 4;
		vk::Memory in_staging = vk::new_buffer(
			allocator: &ctx.allocator,
			usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
			properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
			data: input_chw.ptr,
			data_size: img_bytes,
		)!!;
		ctx.device.@single_time_command(ctx.compute_queue ; vk::CommandBuffer up_cmd)
		{
			vk::cmdCopyBuffer(
				up_cmd,
				in_staging.buffer,
				enc_acts.buf_a.gpu_buffer.buffer,
				1,
				(vk::BufferCopy[]) { { .srcOffset = 0, .dstOffset = 0, .size = img_bytes } }
			);
		}!!;
		in_staging.free();
	}

	// 4. Run encoder forward
	io::printfn("  Running VAE encoder...");
	enc.forward(&enc_acts, img_h, img_w)!!;

	// 5. Download encoder output [64, enc_h, enc_w] (enc_h = img_h / 8 = lat_h * 2)
	uint enc_h = img_h / 8;
	uint enc_w = img_w / 8;
	uint enc_elems = 64 * enc_h * enc_w;
	float[] enc_out_cpu = mem::new_array(float, enc_elems);
	defer mem::free(enc_out_cpu);
	{
		usz enc_bytes = (usz)enc_elems * 4;
		vk::Memory dl_staging = vk::new_buffer(
			allocator: &ctx.allocator,
			usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
			properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
			data: null,
			data_size: enc_bytes,
		)!!;
		ctx.device.@single_time_command(ctx.compute_queue ; vk::CommandBuffer dl_cmd)
		{
			vk::cmdCopyBuffer(
				dl_cmd,
				enc_acts.buf_a.gpu_buffer.buffer,
				dl_staging.buffer,
				1,
				(vk::BufferCopy[]) { { .srcOffset = 0, .dstOffset = 0, .size = enc_bytes } }
			);
		}!!;
		mem::copy(enc_out_cpu.ptr, dl_staging.data(), enc_bytes);
		dl_staging.free();
	}

	// 6. CPU post-processing: quant_conv -> split mean -> pixel_unshuffle -> BN-forward
	uint dit_elems = 128 * lat_h * lat_w;
	float[] dit_latent = mem::new_array(float, dit_elems);
	encoder_postprocess_to_dit_latent(&enc_sf, enc_out_cpu, dit_latent, enc_h, enc_w, lat_h, lat_w)!!;

	io::printfn("  Encoded latent: [128, %d, %d]", lat_h, lat_w);
	return dit_latent;
}

// --- Multi-layer text encoding ---
// Encode text through Qwen3, extract hidden states at 3 evenly-spaced layers,
// concatenate them per-token: [seq_len, 3 * dim].
// Diffusers Klein uses text_encoder_out_layers=(9, 18, 27) — 1-indexed hidden_states
// where hidden_states[k] = output after layer k-1. For Qwen3-8B (36 layers), spacing=9
// yields {8, 17, 26} 0-indexed.
fn llm::Tensor? encode_text_multilayer(
	llm::DeviceContext* ctx,
	llm_text::LlmModel* model,
	uint[] tokens,
	uint seq_len,
	uint dim
)
{
	llm::ModelConfig* c = &model.config;

	uint[3] extract_layers;
	if (c.n_layers >= 12)
	{
		uint spacing = c.n_layers / 4;
		extract_layers = { spacing - 1, spacing * 2 - 1, spacing * 3 - 1 };
	}
	else
	{
		extract_layers = { 0, c.n_layers / 2, c.n_layers - 1 };
	}

	// Working hidden states buffer
	llm::Tensor hidden_states = llm::create_f32_tensor(ctx, { (ulong)seq_len * dim, 0, 0, 0 }, 1)!!;

	// Buffers for 3 extracted layer outputs: each [seq_len, dim]
	llm::Tensor[3] layer_outputs;
	for (uint i = 0; i < 3; i++)
	{
		layer_outputs[i] = llm::create_f32_tensor(ctx, { (ulong)seq_len * dim, 0, 0, 0 }, 1)!!;
	}

	// 1. Embed all tokens
	for (uint pos = 0; pos < seq_len; pos++)
	{
		float* staging_ptr = (float*)model.embedding_staging.data();
		llm::dequant_embedding_row(
			model.weights.embedding_data,
			&model.weights.embedding_info,
			tokens[pos],
			staging_ptr,
			dim
		);
		ctx.device.@single_time_command(ctx.compute_queue ; vk::CommandBuffer emb_cmd)
		{
			vk::cmdCopyBuffer(
				emb_cmd,
				model.embedding_staging.buffer,
				hidden_states.gpu_buffer.buffer,
				1,
				(vk::BufferCopy[]) { { .srcOffset = 0, .dstOffset = (ulong)pos * dim * 4, .size = (ulong)dim * 4 } }
			);
		}!!;
	}

	// 2. Process layers, snapshot at extract points
	for (uint l = 0; l < c.n_layers; l++)
	{
		llm_text::LayerWeights lw = llm_text::load_single_layer(ctx, model.gf, model.names, c, l)!!;

		llm::RenderGraph g;
		g.init(ctx)!!;
		for (uint pos = 0; pos < seq_len; pos++)
		{
			if (g.needs_barrier) llm::compute_barrier(g.cmd);
			vk::cmdCopyBuffer(
				g.cmd,
				hidden_states.gpu_buffer.buffer,
				model.acts.hidden.gpu_buffer.buffer,
				1,
				(vk::BufferCopy[]) { { .srcOffset = (ulong)pos * dim * 4, .dstOffset = 0, .size = (ulong)dim * 4 } }
			);
			g.needs_barrier = true;

			model.forward_layer(&g, &lw, l, pos)!!;

			if (g.needs_barrier) llm::compute_barrier(g.cmd);
			vk::cmdCopyBuffer(
				g.cmd,
				model.acts.hidden.gpu_buffer.buffer,
				hidden_states.gpu_buffer.buffer,
				1,
				(vk::BufferCopy[]) { { .srcOffset = 0, .dstOffset = (ulong)pos * dim * 4, .size = (ulong)dim * 4 } }
			);
			g.needs_barrier = true;
		}
		g.flush()!!;
		llm_text::free_layer_weights(&lw, c);

		// Snapshot if this is an extraction layer
		for (uint i = 0; i < 3; i++)
		{
			if (l == extract_layers[i])
			{
				ctx.device.@single_time_command(ctx.compute_queue ; vk::CommandBuffer snap_cmd)
				{
					vk::cmdCopyBuffer(
						snap_cmd,
						hidden_states.gpu_buffer.buffer,
						layer_outputs[i].gpu_buffer.buffer,
						1,
						(vk::BufferCopy[]) { { .srcOffset = 0, .dstOffset = 0, .size = (ulong)seq_len * dim * 4 } }
					);
				}!!;
			}
		}
	}
	hidden_states.free();

	// 3. Concatenate: interleave 3 layer outputs → [seq_len, 3*dim]
	uint out_dim       = dim * 3;
	usz out_size       = (usz)seq_len * out_dim;
	float[] concat_cpu = mem::new_array(float, out_size);

	// Download all 3 layer outputs
	float*[3] layer_data;
	for (uint i = 0; i < 3; i++)
	{
		layer_data[i] = (float*)mem::calloc((usz)seq_len * dim * 4);
		vk::Memory dl = vk::new_buffer(
			allocator: &ctx.allocator,
			usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
			properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
			data: null,
			data_size: (usz)seq_len * dim * 4,
		)!!;
		ctx.device.@single_time_command(ctx.compute_queue ; vk::CommandBuffer dl_cmd)
		{
			vk::cmdCopyBuffer(
				dl_cmd,
				layer_outputs[i].gpu_buffer.buffer,
				dl.buffer,
				1,
				(vk::BufferCopy[]) { { .srcOffset = 0, .dstOffset = 0, .size = (ulong)seq_len * dim * 4 } }
			);
		}!!;
		mem::copy(layer_data[i], (float*)dl.data(), (usz)seq_len * dim * 4);
		dl.free();
		layer_outputs[i].free();
	}

	// Concatenate raw hidden states (official pipeline passes raw, no normalization)
	for (uint t = 0; t < seq_len; t++)
	{
		for (uint i = 0; i < 3; i++)
		{
			for (uint d = 0; d < dim; d++)
			{
				concat_cpu[t * out_dim + i * dim + d] = layer_data[i][t * dim + d];
			}
		}
	}

	for (uint i = 0; i < 3; i++) mem::free(layer_data[i]);

	// Upload concatenated embeddings to GPU
	llm::Tensor result = llm::create_f32_tensor(ctx, { (ulong)out_size, 0, 0, 0 }, 1)!!;

	{
		vk::Memory ul = vk::new_buffer(
			allocator: &ctx.allocator,
			usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
			properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
			data: concat_cpu.ptr,
			data_size: out_size * 4,
		)!!;
		ctx.device.@single_time_command(ctx.compute_queue ; vk::CommandBuffer ul_cmd)
		{
			vk::cmdCopyBuffer(
				ul_cmd,
				ul.buffer,
				result.gpu_buffer.buffer,
				1,
				(vk::BufferCopy[]) { { .srcOffset = 0, .dstOffset = 0, .size = (ulong)out_size * 4 } }
			);
		}!!;
		ul.free();
	}
	mem::free(concat_cpu);

	return result;
}
