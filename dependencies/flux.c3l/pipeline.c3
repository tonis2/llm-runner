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

// Flux 2 Klein Pipeline: txt2img with Flux.2 Klein 9B
// 
// Flow:
//   1. Encode text with Qwen3-8B → extract 3 hidden states → [seq_len, 12288]
//   2. Load Flux DiT from GGUF
//   3. Denoise with flow matching Euler steps
//   4. Pixel shuffle + VAE decode → output image
//   5. Save

struct FluxConfig
{
	String model_path; // Flux DiT GGUF
	String text_model_path; // Qwen3-8B GGUF (text encoder)
	String vae_path; // ae.safetensors (Flux VAE)
	String prompt;
	String input_path; // Input image for img2img / kontext
	String output_path;
	String edit_mode;  // "" (txt2img, default) | "img2img" | "kontext"
	uint image_size;
	uint num_steps;
	float cfg_scale;
	float strength; // img2img strength (0-1): 0=keep input, 1=full re-roll
	uint seed;
	bool offload;
}

fn FluxConfig flux_config_defaults()
{
	return {
		.output_path = "output.png",
		.image_size = 1024,
		.num_steps = 4, // Klein is distilled to 4 steps
		.cfg_scale = 1.0f, // No CFG for distilled Klein
		.strength = 0.75f,
		.seed = 42,
	};
}

// --- Pipeline Implementation ---

fn void? run_flux_pipeline(FluxConfig* config)
{
	io::printfn("\n=== Flux 2 Klein Pipeline ===");
	io::printfn("  DiT model: %s", config.model_path);
	io::printfn("  Text model: %s", config.text_model_path);
	io::printfn("  VAE: %s", config.vae_path);
	io::printfn("  Prompt: %s", config.prompt);
	io::printfn("  Size: %d, Steps: %d, Seed: %d", config.image_size, config.num_steps, config.seed);

	// Setup Vulkan
	llm::DeviceContext ctx = llm::createContext()!!;
	defer ctx.free();

	// --- Phase 1: Text encoding with Qwen3 ---
	io::printfn("\n--- Phase 1: Qwen3 Text Encoding ---");

	mmap::FileMmap text_mm = file::mmap_open(config.text_model_path, "rb")!!;
	char[] text_data       = text_mm.bytes();
	llm::GGUFFile text_gf  = llm::gguf_parse(text_data)!!;

	String arch_name = llm::detect_architecture(&text_gf);
	io::printfn("  Text encoder architecture: %s", arch_name);

	String config_json;
	if (arch_name == "qwen3")
	{
		config_json = (String)&llm_text::QWEN3_CONFIG_JSON;
	}
	else if (arch_name == "qwen2")
	{
		config_json = (String)&llm_text::QWEN2_CONFIG_JSON;
	}
	else
	{
		config_json = (String)&llm_text::QWEN3_CONFIG_JSON;
	}

	llm::ModelConfig text_config = llm::load_arch_config(config_json, &text_gf)!!;
	llm::WeightNames text_names  = llm::load_weight_names(config_json)!!;

	llm::Tokenizer tok = llm::load_tokenizer(&text_gf)!!;

	// Build prompt with chat template (official pipeline uses apply_chat_template with enable_thinking=False)
	char[4096] cond_buf;
	usz cond_len      = 0;
	String cond_parts = "<|im_start|>user\n";
	for (usz i = 0; i < cond_parts.len; i++)
	{
		cond_buf[cond_len] = cond_parts[i];
		cond_len++;
	}
	for (usz i = 0; i < config.prompt.len; i++)
	{
		cond_buf[cond_len] = config.prompt[i];
		cond_len++;
	}
	// Qwen3 chat template with enable_thinking=False appends an empty <think></think> block.
	// Must match diffusers' tokenizer.apply_chat_template(..., enable_thinking=False) exactly.
	String cond_suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
	for (usz i = 0; i < cond_suffix.len; i++)
	{
		cond_buf[cond_len] = cond_suffix[i];
		cond_len++;
	}
	String cond_prompt = (String)cond_buf[0..cond_len - 1];

	uint[] raw_tokens = tok.encode_with_specials(cond_prompt)!!;
	uint raw_len      = (uint)raw_tokens.len;

	// Diffusers Flux2KleinPipeline tokenizes with padding="max_length", max_length=512.
	// The DiT was trained seeing 512 text K/V pairs in joint attention, so we must
	// pad to match. Diffusers does NOT pass an attention mask to the transformer, so
	// padding tokens actively participate in attention.
	const uint FLUX2_TEXT_LEN = 512;
	uint text_len             = FLUX2_TEXT_LEN;
	if (raw_len > FLUX2_TEXT_LEN) raw_len = FLUX2_TEXT_LEN; // Truncate (diffusers truncation=True)
	// Diffusers' Qwen2TokenizerFast pads with "<|endoftext|>". The Qwen3-8B GGUF
	// declares padding_token_id=151654 (<|vision_pad|>), which is wrong for this
	// pipeline, so resolve the diffusers-correct token by name.
	uint flux_pad_id = tok.find_special_token("<|endoftext|>");
	if (flux_pad_id == 0) flux_pad_id = tok.pad_id; // Fallback if vocab lookup fails
	uint[] cond_tokens = mem::new_array(uint, FLUX2_TEXT_LEN);
	for (uint i = 0; i < raw_len; i++) cond_tokens[i] = raw_tokens[i];
	for (uint i = raw_len; i < FLUX2_TEXT_LEN; i++) cond_tokens[i] = flux_pad_id;
	mem::free(raw_tokens);
	io::printfn("  Tokenized: %d real tokens, padded to %d (pad_id=%d)", raw_len, text_len, flux_pad_id);

	// Load LLM for text encoding
	llm_text::LlmModel text_model = llm_text::load_llm_model(
		&ctx,
		&text_gf,
		&text_config,
		&text_names,
		encode_only: true,
		max_seq_len: text_len
	)!!;

	// Encode text with multi-layer extraction: 3 hidden states concatenated → [text_len, 3 * dim]
	llm::Tensor text_embeddings = encode_text_multilayer(&ctx, &text_model, cond_tokens, text_len, text_config.dim)!!;
	io::printfn("  Text embeddings: [%d, %d]", text_len, text_config.dim * 3);

	mem::free(cond_tokens);
	tok.free();
	text_model.free();
	text_gf.free();
	text_mm.destroy();

	// Image dimensions: Flux.2 uses 16x spatial reduction (8x VAE + 2x pixel unshuffle).
	// Computed early because the (optional) img2img encode pass below needs lat_h/lat_w.
	uint img_h        = (config.image_size + 15) / 16 * 16; // Round to multiple of 16
	uint img_w        = img_h;
	uint lat_h        = img_h / 16;                     // 16x reduction (8x VAE * 2x unshuffle)
	uint lat_w        = img_w / 16;
	uint n_img        = lat_h * lat_w;                  // PATCH_SIZE=1, so n_patches = spatial elements
	uint n_txt        = text_len;
	uint n_joint      = n_img + n_txt;
	uint latent_elems = FLUX_LATENT_CH * lat_h * lat_w; // 128 * lat_h * lat_w

	if (img_h != config.image_size)
	{
		io::printfn("  Note: size %d rounded to %d (must be multiple of 16)", config.image_size, img_h);
	}
	io::printfn("  Image: %dx%d → latent: [128, %d, %d] → %d patches", img_h, img_w, lat_h, lat_w, n_img);
	io::printfn("  Sequence: %d img + %d txt = %d joint", n_img, n_txt, n_joint);

	// --- Optional Phase 1.5: img2img / kontext input encoding ---
	// Done before loading the DiT so the encoder weights don't fight with DiT VRAM.
	float[] img2img_ref_latent = (float[]){};
	if (config.edit_mode == "img2img")
	{
		if (config.input_path.len == 0)
		{
			io::printfn("ERROR: --edit-mode img2img requires --input <image>");
			return llm::COMPUTE_ERROR~;
		}
		io::printfn("\n--- Phase 1.5: VAE Encoding (mode=img2img) ---");
		img2img_ref_latent = encode_image_to_dit_latent(&ctx, config.input_path, config.vae_path, img_h, img_w, lat_h, lat_w)!!;
		ctx.allocator.release_empty_pages();
	}
	else if (config.edit_mode == "kontext")
	{
		// Kontext-style editing concatenates the encoded reference latent as
		// extra image-stream tokens to the DiT input, with offset (H, W) position
		// IDs so the model can distinguish noise tokens from reference tokens.
		// This requires plumbing a separate n_ref count through ~30 sites in
		// flux_dit.c3 (mRoPE table, image-stream Q/K/V, joint sequence buffers).
		// The encoder + preprocessing infrastructure (encode_image_to_dit_latent
		// above) is ready for this; the DiT-side splice is not yet wired.
		// Note: Klein 9B is distilled txt2img and is unlikely to produce strong
		// instruction edits even with the wiring in place — an edit-tuned
		// checkpoint (Flux 2 Pro / Kontext-tuned) would be needed for that.
		io::printfn("ERROR: --edit-mode kontext is not yet implemented end-to-end.");
		io::printfn("       The VAE encoder works (validated cos>=0.995 vs PyTorch ref);");
		io::printfn("       what's missing is concatenating the reference latent as extra");
		io::printfn("       DiT image-stream tokens with offset RoPE positions.");
		io::printfn("       See dependencies/flux.c3l/flux_dit.c3 (n_img/n_joint sites).");
		return llm::COMPUTE_ERROR~;
	}

	// --- Phase 2: Load Flux DiT ---
	io::printfn("\n--- Phase 2: Loading Flux DiT ---");

	FluxDiTKernels dit_kernels = create_flux_dit_kernels(&ctx)!!;
	defer dit_kernels.free(ctx.device);

	mmap::FileMmap dit_mm = file::mmap_open(config.model_path, "rb")!!;
	char[] dit_data       = dit_mm.bytes();
	llm::GGUFFile dit_gf  = llm::gguf_parse(dit_data)!!;
	FluxDiTModel dit      = load_flux_dit(&ctx, &dit_gf, &dit_kernels)!!;
	// dit is freed explicitly after the denoise loop to free ~9 GB of VRAM
	// before the VAE allocates its full-resolution scratch buffers.

	// Set model dimensions
	dit.n_img = n_img;
	dit.n_txt = n_txt;
	dit.n_joint = n_joint;
	dit.latent_h = lat_h;
	dit.latent_w = lat_w;

	// Allocate activations
	dit.acts = allocate_flux_activations(&ctx, n_img, n_txt, latent_elems)!!;

	// Precompute mRoPE tables
	precompute_flux_mrope(&ctx, &dit.rope_cos, &dit.rope_sin, n_txt, n_img, lat_w, lat_h)!!;

	// --- Phase 3: Denoising ---
	io::printfn("\n--- Phase 3: Denoising (%d steps) ---", config.num_steps);

	// Scheduler: linear sigma schedule with dynamic shift.
	// For img2img we rebuild the schedule from a lower starting sigma so the
	// reference latent actually contributes — Klein's distilled 4-step shift
	// (~7.6) puts sigma[1] at ~0.96, which would drown out the reference at
	// any practical `strength`. Treating `strength` as the noise fraction and
	// rescheduling all `num_steps` Euler steps from there preserves user intent.
	float[] sigmas = mem::new_array(float, (usz)config.num_steps + 1);

	{
		// Use compute_empirical_mu from diffusers Flux2KleinPipeline
		// https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py
		float a1            = 8.73809524e-05f;
		float b1            = 1.89833333f;
		float a2            = 0.00016927f;
		float b2            = 0.45666666f;
		float image_seq_len = (float)n_img;
		float num_steps_f   = (float)config.num_steps;
		float mu;
		if (image_seq_len > 4300.0f)
		{
			mu = a2 * image_seq_len + b2;
		}
		else
		{
			float m_200 = a2 * image_seq_len + b2;
			float m_10  = a1 * image_seq_len + b1;
			float a     = (m_200 - m_10) / 190.0f;
			float b     = m_200 - 200.0f * a;
			mu = a * num_steps_f + b;
		}
		float flux_shift = math::exp(mu);

		// Pick the schedule's top t. For txt2img/kontext: t_top=1.0 → sigma=1.0
		// (full noise). For img2img: invert the shift to find the t that yields
		// sigma == config.strength, so the schedule starts exactly at the
		// requested noise level.
		float t_top = 1.0f;
		if (config.edit_mode == "img2img" && img2img_ref_latent.len > 0)
		{
			float s = config.strength;
			if (s < 0.0f) s = 0.0f;
			if (s > 1.0f) s = 1.0f;
			// Inverse of time_snr_shift: t = s / (shift*(1-s) + s)
			t_top = s / (flux_shift * (1.0f - s) + s);
		}

		for (uint i = 0; i < config.num_steps; i++)
		{
			float frac = (float)i * (1.0f - 1.0f / (float)config.num_steps)
				/ (float)(config.num_steps > 1 ? config.num_steps - 1 : 1);
			float t = t_top * (1.0f - frac);
			sigmas[i] = pipelines::time_snr_shift(flux_shift, t);
		}
		sigmas[config.num_steps] = 0.0f;
		io::printfn("  Scheduler: mu=%.3f, shift=%.3f, sigma_top=%.4f", mu, flux_shift, sigmas[0]);
	}

	pipelines::FlowMatchingState scheduler = { .num_steps = config.num_steps, .sigmas = sigmas, };
	defer scheduler.free();

	// Generate random latent [128, lat_h, lat_w]
	float[] latent_cpu = mem::new_array(float, latent_elems);
	pipelines::generate_random_latent(latent_cpu, config.seed);

	// img2img: blend noise with the encoded reference latent at sigma[0].
	// Matches diffusers FlowMatchScheduler.scale_noise: latent = sigma*noise + (1-sigma)*ref.
	// We always run the full denoise loop from step 0 (the schedule itself was
	// rebuilt above with a lower top sigma instead of skipping steps).
	if (config.edit_mode == "img2img" && img2img_ref_latent.len > 0)
	{
		float sigma_start = sigmas[0];
		io::printfn("  img2img: strength=%.2f, sigma_start=%.4f, ref_weight=%.4f",
			config.strength, sigma_start, 1.0f - sigma_start);
		for (uint i = 0; i < latent_elems; i++)
		{
			latent_cpu[i] = sigma_start * latent_cpu[i] + (1.0f - sigma_start) * img2img_ref_latent[i];
		}
		mem::free(img2img_ref_latent);
		img2img_ref_latent = (float[]){};
	}

	// Upload latent to GPU
	{
		vk::Memory lat_staging = vk::new_buffer(
			allocator: &ctx.allocator,
			usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
			properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
			data: latent_cpu.ptr,
			data_size: (usz)latent_elems * 4,
		)!!;
		ctx.device.@single_time_command(ctx.compute_queue ; vk::CommandBuffer copy_cmd)
		{
			vk::cmdCopyBuffer(
				copy_cmd,
				lat_staging.buffer,
				dit.acts.latent.gpu_buffer.buffer,
				1,
				(vk::BufferCopy[]) { { .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 } }
			);
		}!!;
		lat_staging.free();
	}

	// Denoising loop — always runs the full num_steps. img2img controls noise
	// retention via the rescheduled top sigma (see schedule build above), not
	// by skipping leading steps.
	for (uint step = 0; step < config.num_steps; step++)
	{
		float sigma = scheduler.get_timestep(step);
		float dt    = scheduler.get_dt(step);
		io::printfn("  Step %d/%d: sigma=%.4f, dt=%.4f", step + 1, config.num_steps, sigma, dt);

		dit.forward(&text_embeddings, sigma)!!;

		// Euler step: latent += dt * velocity
		llm::RenderGraph euler_g;
		euler_g.init(&ctx)!!;
		zimage::FlowEulerPC euler_pc = { .n = latent_elems, .dt = dt };
		euler_g.dispatch(
			&dit_kernels.dit.flow_euler_step,
			{ &dit.acts.latent, &dit.acts.velocity },
			&euler_pc,
			llm::ceil_div(latent_elems, 256)
		);
		euler_g.flush()!!;
	}

	// --- Phase 4: VAE decode ---
	io::printfn("\n--- Phase 4: VAE Decoding ---");

	// Download denoised latent [128, lat_h, lat_w] from GPU

	{
		vk::Memory dl_staging = vk::new_buffer(
			allocator: &ctx.allocator,
			usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
			properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
			data: null,
			data_size: (usz)latent_elems * 4,
		)!!;
		ctx.device.@single_time_command(ctx.compute_queue ; vk::CommandBuffer dl_cmd)
		{
			vk::cmdCopyBuffer(
				dl_cmd,
				dit.acts.latent.gpu_buffer.buffer,
				dl_staging.buffer,
				1,
				(vk::BufferCopy[]) { { .srcOffset = 0, .dstOffset = 0, .size = (ulong)latent_elems * 4 } }
			);
		}!!;
		float* mapped = (float*)dl_staging.data();
		mem::copy(latent_cpu.ptr, mapped, (usz)latent_elems * 4);
		dl_staging.free();
	}

	// Free DiT weights and activations now — the VAE needs the VRAM headroom.
	// release_empty_pages actually returns device memory to the driver; without
	// it, Memory.free only marks regions free within retained Vulkan pages.
	dit.free();
	dit_gf.free();
	dit_mm.destroy();
	ctx.allocator.release_empty_pages();

	// Flux.2 VAE decode:
	// 1. Apply BatchNorm (inv_normalize) in 128-ch space using bn.running_mean/var
	// 2. Pixel shuffle: [128, lat_h, lat_w] → [32, 2*lat_h, 2*lat_w]
	// 3. post_quant_conv: 1x1 conv [32] → [32]
	// 4. Standard VAE decoder

	llm::SafetensorsFile vae_sf = llm::safetensors_open(config.vae_path)!!;
	defer vae_sf.close();

	// Apply BatchNorm inverse normalization: x = x * sqrt(var + eps) + mean

	{
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
		for (uint c = 0; c < 128; c++)
		{
			float std = math::sqrt(bn_var[c] + eps);
			for (uint j = 0; j < lat_h * lat_w; j++)
			{
				uint idx = c * lat_h * lat_w + j;
				latent_cpu[idx] = latent_cpu[idx] * std + bn_mean[c];
			}
		}
	}

	// Pixel shuffle: [128, lat_h, lat_w] → [32, 2*lat_h, 2*lat_w]
	uint vae_h         = lat_h * 2;               // = img_h / 8
	uint vae_w         = lat_w * 2;               // = img_w / 8
	uint vae_elems     = FLUX_VAE_CH * vae_h * vae_w; // 32 * vae_h * vae_w
	float[] vae_latent = mem::new_array(float, vae_elems);
	pixel_shuffle_cpu(latent_cpu, vae_latent, FLUX_LATENT_CH, lat_h, lat_w, 2);
	mem::free(latent_cpu);

	// Apply post_quant_conv (1x1 conv, [32,32,1,1] — per-pixel linear transform)

	{
		float[32 * 32] pqc_weight;
		float[32] pqc_bias;
		if (try info = vae_sf.find_tensor("post_quant_conv.weight"))
		{
			float* data = (float*)(vae_sf.data_base + (usz)info.data_start);
			mem::copy(&pqc_weight, data, 32 * 32 * 4);
		}
		if (try info = vae_sf.find_tensor("post_quant_conv.bias"))
		{
			float* data = (float*)(vae_sf.data_base + (usz)info.data_start);
			mem::copy(&pqc_bias, data, 32 * 4);
		}
		// Apply 1x1 conv per spatial position: out[c] = bias[c] + sum_k(weight[c,k] * in[k])
		uint spatial = vae_h * vae_w;
		float[32] tmp;
		for (uint s = 0; s < spatial; s++)
		{
			// Read input for this spatial position
			for (uint c = 0; c < 32; c++)
			{
				tmp[c] = vae_latent[c * spatial + s];
			}
			// Compute output
			for (uint oc = 0; oc < 32; oc++)
			{
				float val = pqc_bias[oc];
				for (uint ic = 0; ic < 32; ic++)
				{
					val += pqc_weight[oc * 32 + ic] * tmp[ic];
				}
				vae_latent[oc * spatial + s] = val;
			}
		}
	}

	// Load VAE decoder (reuse vae_sf already opened above)
	zimage::VaeKernels vae_kernels = zimage::create_vae_kernels(&ctx)!!;
	defer vae_kernels.free(ctx.device);

	zimage::FluxVaeDecoder vae_dec = zimage::load_flux_vae_decoder(&ctx, &vae_sf, &vae_kernels)!!;
	defer vae_dec.free();

	zimage::FluxVaeActivations vae_acts = zimage::allocate_flux_vae_activations(&ctx, img_h)!!;
	defer vae_acts.free();

	// Upload pixel-shuffled latent to VAE input
	// The VAE conv_in will read however many channels it expects from the buffer

	{
		vk::Memory lat_staging = vk::new_buffer(
			allocator: &ctx.allocator,
			usage: vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
			properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
			data: vae_latent.ptr,
			data_size: (usz)vae_elems * 4,
		)!!;
		ctx.device.@single_time_command(ctx.compute_queue ; vk::CommandBuffer vae_cmd)
		{
			vk::cmdCopyBuffer(
				vae_cmd,
				lat_staging.buffer,
				vae_acts.buf_a.gpu_buffer.buffer,
				1,
				(vk::BufferCopy[]) { { .srcOffset = 0, .dstOffset = 0, .size = (ulong)vae_elems * 4 } }
			);
		}!!;
		lat_staging.free();
	}
	mem::free(vae_latent);

	vae_dec.forward(&vae_acts, vae_h, vae_w)!!;

	// --- Phase 5: Download and save ---
	io::printfn("\n--- Phase 5: Saving output ---");

	usz total_px     = (usz)img_h * img_w * 3;
	float[] img_data = mem::new_array(float, total_px);


	{
		vk::Memory dl_staging = vk::new_buffer(
			allocator: &ctx.allocator,
			usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
			properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
			data: null,
			data_size: total_px * 4,
		)!!;
		ctx.device.@single_time_command(ctx.compute_queue ; vk::CommandBuffer img_cmd)
		{
			vk::cmdCopyBuffer(
				img_cmd,
				vae_acts.buf_a.gpu_buffer.buffer,
				dl_staging.buffer,
				1,
				(vk::BufferCopy[]) { { .srcOffset = 0, .dstOffset = 0, .size = total_px * 4 } }
			);
		}!!;
		float* mapped = (float*)dl_staging.data();
		mem::copy(img_data.ptr, mapped, total_px * 4);
		dl_staging.free();
	}

	image::Image result = pipelines::float_tensor_to_image(img_data, img_w, img_h, 3)!!;
	mem::free(img_data);

	png::save_file(&result, config.output_path)!!;
	io::printfn("  Saved to %s (%dx%d)", config.output_path, img_w, img_h);

	// Cleanup
	text_embeddings.free();
	io::printfn("\n=== Pipeline complete ===");
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
// Loads encoder, encodes, applies CPU post-processing (quant_conv + split + pixel_unshuffle + BN-forward),
// then frees the encoder. Caller frees the returned array.
fn float[]? encode_image_to_dit_latent(
	llm::DeviceContext* ctx,
	String input_path,
	String vae_path,
	uint img_h, uint img_w,
	uint lat_h, uint lat_w
)
{
	// 1. Load + preprocess input image
	io::printfn("  Loading input: %s", input_path);
	image::Image src_img = llm::load_png(input_path)!!;
	defer src_img.free();
	image::Image preprocessed = llm::resize_center_crop(&src_img, img_w, img_h)!!;
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

	zimage::FluxVaeActivations enc_acts = zimage::allocate_flux_vae_activations(ctx, img_h)!!;
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
