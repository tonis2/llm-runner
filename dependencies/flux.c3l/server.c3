module flux;

// Flux server state + per-request generation entry point.
// 
// The CLI (run_flux_pipeline) now goes through this same path with keep_dit=false.
// An HTTP server can hold a FluxServerState across many flux_generate calls,
// optionally with keep_dit=true to keep DiT weights resident on the GPU.

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
import std::time::clock;

// Persistent state held across requests. What lives here is the pre-loaded
// stuff the server pays for once at startup. The text encoder and VAE still
// load + free per request to keep VRAM headroom on 16 GB cards (1024² needs
// ~13 GB if DiT stays resident through the VAE decode — borderline). When
// keep_dit is true the 9 GB DiT weights stay resident, only DiT activations
// + RoPE tables churn per request.
struct FluxServerState
{
	llm::DeviceContext ctx;
	FluxDiTKernels dit_kernels;
	mmap::FileMmap dit_mm; // DiT GGUF mmap (kept open so per-request load is just a VRAM upload)
	llm::GGUFFile dit_gf; // parsed once
	String text_model_path;
	String vae_path;
	bool keep_dit;
	bool dit_loaded; // true if `dit` holds weights
	FluxDiTModel dit; // valid only if dit_loaded
}

// Request-scoped inputs. Mirrors the relevant subset of FluxConfig, plus an
// optional in-memory image (server passes pre-decoded init_images here).
struct FluxGenerateRequest
{
	String prompt;
	String edit_mode; // "" | "img2img" | "kontext"
	String input_path; // used iff input_image == null
	image::Image* input_image; // optional, takes precedence over input_path
	uint image_width;
	uint image_height;
	uint image_size; // fallback when width/height are 0
	uint num_steps;
	float strength;
	uint seed;
	String lora_path;        // empty = no LoRA
	float lora_strength;     // 1.0 if not specified
}

fn FluxServerState? init_flux_server(String dit_path, String text_model_path, String vae_path, bool keep_dit)
{
	io::printfn("\n=== Initialising Flux server state ===");
	io::printfn("  DiT: %s", dit_path);
	io::printfn("  Text: %s", text_model_path);
	io::printfn("  VAE: %s", vae_path);
	io::printfn("  keep_dit: %s", keep_dit ? "true" : "false");

	llm::DeviceContext ctx     = llm::createContext()!!;
	FluxDiTKernels dit_kernels = create_flux_dit_kernels(&ctx)!!;
	mmap::FileMmap dit_mm      = file::mmap_open(dit_path, "rb")!!;
	llm::GGUFFile dit_gf       = llm::gguf_parse(dit_mm.bytes())!!;

	FluxServerState state = {
		.ctx = ctx,
		.dit_kernels = dit_kernels,
		.dit_mm = dit_mm,
		.dit_gf = dit_gf,
		.text_model_path = text_model_path,
		.vae_path = vae_path,
		.keep_dit = keep_dit,
		.dit_loaded = false,
	};

	if (keep_dit)
	{
		io::printfn("  Loading DiT weights into VRAM...");
		state.dit = load_flux_dit(&ctx, &state.dit_gf, &state.dit_kernels)!!;
		state.dit_loaded = true;
	}
	return state;
}

fn void FluxServerState.free(&self)
{
	if (self.dit_loaded)
	{
		self.dit.free();
		self.dit_loaded = false;
	}
	self.dit_kernels.free(self.ctx.device);
	self.dit_gf.free();
	self.dit_mm.destroy();
	self.ctx.free();
}

// Run one full generation. Caller owns the returned image::Image.
fn image::Image? flux_generate(FluxServerState* state, FluxGenerateRequest* req)
{
	llm::DeviceContext* ctx = &state.ctx;

	uint req_w = req.image_width > 0 ? req.image_width : req.image_size;
	uint req_h = req.image_height > 0 ? req.image_height : req.image_size;
	io::printfn("\n=== Flux generate ===");
	io::printfn("  Prompt: %s", req.prompt);
	io::printfn(
		"  Size: %dx%d, Steps: %d, Seed: %d, Mode: %s",
		req_w,
		req_h,
		req.num_steps,
		req.seed,
		req.edit_mode.len > 0 ? req.edit_mode : "txt2img"
	);

	// --- Phase 1: Text encoding with Qwen3 ---
	io::printfn("\n--- Phase 1: Qwen3 Text Encoding ---");

	mmap::FileMmap text_mm = file::mmap_open(state.text_model_path, "rb")!!;
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
	llm::Tokenizer tok           = llm::load_tokenizer(&text_gf)!!;

	char[4096] cond_buf;
	usz cond_len      = 0;
	String cond_parts = "<|im_start|>user\n";
	for (usz i = 0; i < cond_parts.len; i++)
	{
		cond_buf[cond_len] = cond_parts[i];
		cond_len++;
	}
	for (usz i = 0; i < req.prompt.len; i++)
	{
		cond_buf[cond_len] = req.prompt[i];
		cond_len++;
	}
	String cond_suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
	for (usz i = 0; i < cond_suffix.len; i++)
	{
		cond_buf[cond_len] = cond_suffix[i];
		cond_len++;
	}
	String cond_prompt = (String)cond_buf[0..cond_len - 1];

	uint[] raw_tokens = tok.encode_with_specials(cond_prompt)!!;
	uint raw_len      = (uint)raw_tokens.len;

	const uint FLUX2_TEXT_LEN = 512;
	uint text_len             = FLUX2_TEXT_LEN;
	if (raw_len > FLUX2_TEXT_LEN) raw_len = FLUX2_TEXT_LEN;
	uint flux_pad_id = tok.find_special_token("<|endoftext|>");
	if (flux_pad_id == 0) flux_pad_id = tok.pad_id;
	uint[] cond_tokens = mem::new_array(uint, FLUX2_TEXT_LEN);
	for (uint i = 0; i < raw_len; i++) cond_tokens[i] = raw_tokens[i];
	for (uint i = raw_len; i < FLUX2_TEXT_LEN; i++) cond_tokens[i] = flux_pad_id;
	mem::free(raw_tokens);
	io::printfn("  Tokenized: %d real tokens, padded to %d (pad_id=%d)", raw_len, text_len, flux_pad_id);

	llm_text::LlmModel text_model = llm_text::load_llm_model(
		ctx,
		&text_gf,
		&text_config,
		&text_names,
		encode_only: true,
		max_seq_len: text_len
	)!!;

	llm::Tensor text_embeddings = encode_text_multilayer(ctx, &text_model, cond_tokens, text_len, text_config.dim)!!;
	io::printfn("  Text embeddings: [%d, %d]", text_len, text_config.dim * 3);

	mem::free(cond_tokens);
	tok.free();
	text_model.free();
	text_gf.free();
	text_mm.destroy();
	ctx.allocator.release_empty_pages();

	// --- Compute output dims (rounded to multiple of 16) ---
	uint img_h        = (req_h + 15) / 16 * 16;
	uint img_w        = (req_w + 15) / 16 * 16;
	uint lat_h        = img_h / 16;
	uint lat_w        = img_w / 16;
	uint n_img        = lat_h * lat_w;
	uint n_txt        = text_len;
	uint n_joint      = n_img + n_txt;
	uint latent_elems = flux_latent_ch * lat_h * lat_w;

	if (img_h != req_h || img_w != req_w)
	{
		io::printfn("  Note: size %dx%d rounded to %dx%d (must be multiple of 16)", req_w, req_h, img_w, img_h);
	}
	io::printfn("  Image: %dx%d → latent: [128, %d, %d] → %d patches", img_h, img_w, lat_h, lat_w, n_img);
	io::printfn("  Sequence: %d img + %d txt = %d joint", n_img, n_txt, n_joint);

	// --- Optional Phase 1.5: img2img / kontext input encoding ---
	float[] img2img_ref_latent = (float[]) { };
	float[] kontext_ref_packed = (float[]) { };
	uint kontext_n_ref         = 0;
	if (req.edit_mode == "img2img")
	{
		if (req.input_image == null && req.input_path.len == 0)
		{
			io::printfn("ERROR: edit_mode=img2img requires input image");
			text_embeddings.free();
			return llm::COMPUTE_ERROR~;
		}
		io::printfn("\n--- Phase 1.5: VAE Encoding (mode=img2img) ---");
		if (req.input_image != null)
		{
			img2img_ref_latent = encode_image_object_to_dit_latent(
				ctx,
				req.input_image,
				state.vae_path,
				img_h,
				img_w,
				lat_h,
				lat_w
			)!!;
		}
		else
		{
			img2img_ref_latent = encode_image_to_dit_latent(
				ctx,
				req.input_path,
				state.vae_path,
				img_h,
				img_w,
				lat_h,
				lat_w
			)!!;
		}
		ctx.allocator.release_empty_pages();
	}
	else if (req.edit_mode == "kontext")
	{
		if (req.input_image == null && req.input_path.len == 0)
		{
			io::printfn("ERROR: edit_mode=kontext requires input image");
			text_embeddings.free();
			return llm::COMPUTE_ERROR~;
		}
		io::printfn("\n--- Phase 1.5: VAE Encoding (mode=kontext) ---");
		float[] ref_chw;
		if (req.input_image != null)
		{
			ref_chw = encode_image_object_to_dit_latent(
				ctx,
				req.input_image,
				state.vae_path,
				img_h,
				img_w,
				lat_h,
				lat_w
			)!!;
		}
		else
		{
			ref_chw = encode_image_to_dit_latent(ctx, req.input_path, state.vae_path, img_h, img_w, lat_h, lat_w)!!;
		}
		defer mem::free(ref_chw);

		kontext_n_ref = lat_h * lat_w;
		kontext_ref_packed = mem::new_array(float, (usz)kontext_n_ref * flux_latent_ch);
		for (uint sp = 0; sp < kontext_n_ref; sp++)
		{
			for (uint c = 0; c < flux_latent_ch; c++)
			{
				kontext_ref_packed[sp * flux_latent_ch + c] = ref_chw[c * kontext_n_ref + sp];
			}
		}
		ctx.allocator.release_empty_pages();
		io::printfn("  Reference packed: %d tokens x %d ch (T-offset 10)", kontext_n_ref, flux_latent_ch);
	}

	// --- Phase 2: Resolve DiT model ---
	io::printfn("\n--- Phase 2: %s Flux DiT ---", state.dit_loaded ? "Reusing preloaded" : "Loading");

	bool owns_dit = !state.dit_loaded;
	FluxDiTModel local_dit;
	FluxDiTModel* dit;
	if (state.dit_loaded)
	{
		dit = &state.dit;
	}
	else
	{
		local_dit = load_flux_dit(ctx, &state.dit_gf, &state.dit_kernels)!!;
		dit = &local_dit;
	}

	uint n_noise     = n_img;
	uint n_img_total = n_noise + kontext_n_ref;

	dit.n_img = n_img_total;
	dit.n_noise = n_noise;
	dit.n_ref = kontext_n_ref;
	dit.n_txt = n_txt;
	dit.n_joint = n_img_total + n_txt;
	dit.latent_h = lat_h;
	dit.latent_w = lat_w;

	bool use_lora = req.lora_path.len > 0;
	dit.acts = allocate_flux_activations(ctx, n_img_total, n_txt, latent_elems, use_lora)!!;

	// Load LoRA after DiT weights so the file's tensors land into VRAM next to
	// the DiT block weights. We always free per request — this keeps keep_dit
	// servers able to swap the LoRA between requests.
	if (use_lora)
	{
		float strength = req.lora_strength > 0.0f ? req.lora_strength : 1.0f;
		if (try adapter = llm::load_flux_lora_safetensors(
			ctx, req.lora_path, strength, dit.n_dual_blocks, dit.n_single_blocks))
		{
			dit.lora = adapter;
		}
		else
		{
			io::printfn("  Flux LoRA load failed for %s — continuing without LoRA", req.lora_path);
		}
	}

	if (kontext_n_ref > 0)
	{
		usz ref_bytes = (usz)kontext_n_ref * flux_latent_ch * 4;
		vk::Memory ref_buf = vk::new_buffer(
			allocator: &ctx.allocator,
			usage: vk::BUFFER_USAGE_STORAGE_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT
				| vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
			properties: vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			data: kontext_ref_packed.ptr,
			data_size: (sz)ref_bytes,
		)!!;
		dit.ref_patches = {
			.gpu_buffer = ref_buf,
			.dtype = llm::GGML_F32,
			.n_dims = 1,
			.shape = { (ulong)kontext_n_ref * flux_latent_ch, 0, 0, 0 },
			.size_bytes = ref_bytes,
		};
		mem::free(kontext_ref_packed);
	}

	precompute_flux_mrope(
		ctx,
		&dit.rope_cos,
		&dit.rope_sin,
		n_txt,
		n_noise,
		lat_w,
		lat_h,
		kontext_n_ref,
		lat_w,
		lat_h
	)!!;

	// --- Phase 3: Denoising ---
	io::printfn("\n--- Phase 3: Denoising (%d steps) ---", req.num_steps);

	float[] sigmas = mem::new_array(float, (usz)req.num_steps + 1);

	{
		float a1            = 8.73809524e-05f;
		float b1            = 1.89833333f;
		float a2            = 0.00016927f;
		float b2            = 0.45666666f;
		float image_seq_len = (float)n_img;
		float num_steps_f   = (float)req.num_steps;
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

		float t_top = 1.0f;
		if (req.edit_mode == "img2img" && img2img_ref_latent.len > 0)
		{
			float s = req.strength;
			if (s < 0.0f) s = 0.0f;
			if (s > 1.0f) s = 1.0f;
			t_top = s / (flux_shift * (1.0f - s) + s);
		}

		for (uint i = 0; i < req.num_steps; i++)
		{
			float frac = (float)i * (1.0f - 1.0f / (float)req.num_steps)
				/ (float)(req.num_steps > 1 ? req.num_steps - 1 : 1);
			float t = t_top * (1.0f - frac);
			sigmas[i] = pipelines::time_snr_shift(flux_shift, t);
		}
		sigmas[req.num_steps] = 0.0f;
		io::printfn("  Scheduler: mu=%.3f, shift=%.3f, sigma_top=%.4f", mu, flux_shift, sigmas[0]);
	}

	pipelines::FlowMatchingState scheduler = { .num_steps = req.num_steps, .sigmas = sigmas, };
	defer scheduler.free();

	float[] latent_cpu = mem::new_array(float, latent_elems);
	pipelines::generate_random_latent(latent_cpu, req.seed);

	if (req.edit_mode == "img2img" && img2img_ref_latent.len > 0)
	{
		float sigma_start = sigmas[0];
		io::printfn(
			"  img2img: strength=%.2f, sigma_start=%.4f, ref_weight=%.4f",
			req.strength,
			sigma_start,
			1.0f - sigma_start
		);
		for (uint i = 0; i < latent_elems; i++)
		{
			latent_cpu[i] = sigma_start * latent_cpu[i] + (1.0f - sigma_start) * img2img_ref_latent[i];
		}
		mem::free(img2img_ref_latent);
	}


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

	double total_step_ms = 0.0;
	double total_fwd_ms  = 0.0;
	for (uint step = 0; step < req.num_steps; step++)
	{
		float sigma = scheduler.get_timestep(step);
		float dt    = scheduler.get_dt(step);

		Clock step_start = clock::now();
		dit.forward(&text_embeddings, sigma)!!;
		double fwd_ms = step_start.to_now().to_sec() * 1000.0;

		llm::RenderGraph euler_g;
		euler_g.init(ctx)!!;
		zimage::FlowEulerPC euler_pc = { .n = latent_elems, .dt = dt };
		euler_g.dispatch(
			&state.dit_kernels.dit.flow_euler_step,
			{ &dit.acts.latent, &dit.acts.velocity },
			&euler_pc,
			llm::ceil_div(latent_elems, 256)
		);
		euler_g.flush()!!;

		double step_ms = step_start.to_now().to_sec() * 1000.0;
		total_step_ms += step_ms;
		total_fwd_ms  += fwd_ms;
		io::printfn("  Step %d/%d: sigma=%.4f, dt=%.4f  forward=%.1fms  total=%.1fms",
			step + 1, req.num_steps, sigma, dt, fwd_ms, step_ms);
	}
	io::printfn("  --- Step timing: avg fwd=%.1fms, avg total=%.1fms over %d steps ---",
		total_fwd_ms / (double)req.num_steps,
		total_step_ms / (double)req.num_steps,
		req.num_steps);

	// --- Phase 4: VAE decode ---
	io::printfn("\n--- Phase 4: VAE Decoding ---");


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

	// Free per-request DiT resources before allocating the VAE: this is the
	// memory budget that lets a 16 GB card survive 1024². With keep_dit we
	// drop only the activations + RoPE + ref_patches; weights stay.
	dit.acts.free();
	dit.acts = { };
	if (dit.rope_cos.size_bytes > 0)
	{
		dit.rope_cos.free();
		dit.rope_cos = { };
	}
	if (dit.rope_sin.size_bytes > 0)
	{
		dit.rope_sin.free();
		dit.rope_sin = { };
	}
	if (dit.ref_patches.size_bytes > 0)
	{
		dit.ref_patches.free();
		dit.ref_patches = { };
	}
	if (dit.lora.loaded)
	{
		dit.lora.free();
		dit.lora = { };
	}
	if (owns_dit)
	{
		// owns_dit means we loaded weights this request — free them entirely.
		// We've already freed acts/rope/ref above and zeroed them, so dit.free()
		// only frees the weight buffers.
		dit.free();
	}
	ctx.allocator.release_empty_pages();

	llm::SafetensorsFile vae_sf = llm::safetensors_open(state.vae_path)!!;
	defer vae_sf.close();


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

	uint vae_h         = lat_h * 2;
	uint vae_w         = lat_w * 2;
	uint vae_elems     = flux_vae_ch * vae_h * vae_w;
	float[] vae_latent = mem::new_array(float, vae_elems);
	pixel_shuffle_cpu(latent_cpu, vae_latent, flux_latent_ch, lat_h, lat_w, 2);
	mem::free(latent_cpu);


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
		uint spatial = vae_h * vae_w;
		float[32] tmp;
		for (uint s = 0; s < spatial; s++)
		{
			for (uint c = 0; c < 32; c++)
			{
				tmp[c] = vae_latent[c * spatial + s];
			}
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

	zimage::VaeKernels vae_kernels = zimage::create_vae_kernels(ctx)!!;
	defer vae_kernels.free(ctx.device);

	zimage::FluxVaeDecoder vae_dec = zimage::load_flux_vae_decoder(ctx, &vae_sf, &vae_kernels)!!;
	defer vae_dec.free();

	uint vae_alloc_size                 = img_h > img_w ? img_h : img_w;
	zimage::FluxVaeActivations vae_acts = zimage::allocate_flux_vae_activations(ctx, vae_alloc_size)!!;
	defer vae_acts.free();


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

	// --- Phase 5: Download result ---
	io::printfn("\n--- Phase 5: Download result ---");

	usz total_px     = (usz)img_h * img_w * 3;
	float[] img_data = mem::new_array(float, (sz)total_px);


	{
		vk::Memory dl_staging = vk::new_buffer(
			allocator: &ctx.allocator,
			usage: vk::BUFFER_USAGE_TRANSFER_DST_BIT,
			properties: vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
			data: null,
			data_size: (sz)(total_px * 4),
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
		mem::copy(img_data.ptr, mapped, (sz)(total_px * 4));
		dl_staging.free();
	}

	image::Image result = pipelines::float_tensor_to_image(img_data, img_w, img_h, 3)!!;
	mem::free(img_data);

	text_embeddings.free();
	ctx.allocator.release_empty_pages();
	io::printfn("\n=== Generation complete ===");
	return result;
}
