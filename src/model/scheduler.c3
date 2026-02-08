module llm;

import std::io;
import std::math;
import std::math::random;
import std::core::mem;
import vk;

// Diffusion Schedulers: DDIM and Euler
// For SDXS (1-step distilled model), the scheduler is essentially a single pass

// --- Scheduler Types ---

enum SchedulerType : uint {
    DDIM,
    EULER,
}

struct SchedulerState {
    SchedulerType stype;
    uint num_train_timesteps;
    uint num_inference_steps;
    float[] alphas_cumprod;
    float[] timesteps;
    float cfg_scale;
}

// --- Initialize Scheduler ---

fn SchedulerState? init_scheduler(SchedulerConfig* config, SchedulerType stype, uint inference_steps, float cfg_scale) {
    uint train_steps = config.num_train_timesteps;
    float beta_start = config.beta_start;
    float beta_end = config.beta_end;

    // Compute betas using scaled_linear schedule
    // beta_t = (sqrt(beta_start) + t/(T-1) * (sqrt(beta_end) - sqrt(beta_start)))^2
    float[] alphas_cumprod = mem::new_array(float, train_steps);
    float cumprod = 1.0f;

    for (uint t = 0; t < train_steps; t++) {
        float frac = (float)t / (float)(train_steps - 1);
        float sqrt_beta = math::sqrt(beta_start) + frac * (math::sqrt(beta_end) - math::sqrt(beta_start));
        float beta = sqrt_beta * sqrt_beta;
        float alpha = 1.0f - beta;
        cumprod *= alpha;
        alphas_cumprod[t] = cumprod;
    }

    // Compute timestep schedule (evenly spaced, reversed)
    float[] timesteps = mem::new_array(float, inference_steps);
    for (uint i = 0; i < inference_steps; i++) {
        // Linspace from train_steps-1 to 0
        uint denom = inference_steps > 1 ? inference_steps - 1 : 1;
        float t = (float)(train_steps - 1) * (1.0f - (float)i / (float)denom);
        timesteps[i] = t;
    }

    io::printfn("Scheduler: %s, %d inference steps, CFG scale=%.1f",
        stype == DDIM ? "DDIM" : "Euler",
        inference_steps, cfg_scale);
    io::printfn("  alpha_cumprod[0]=%.6f, alpha_cumprod[%d]=%.6f",
        alphas_cumprod[0], train_steps - 1, alphas_cumprod[train_steps - 1]);

    return {
        .stype = stype,
        .num_train_timesteps = train_steps,
        .num_inference_steps = inference_steps,
        .alphas_cumprod = alphas_cumprod,
        .timesteps = timesteps,
        .cfg_scale = cfg_scale,
    };
}

// --- Generate random latent ---

fn void generate_random_latent(float[] latent, uint seed) {
    Sfc64Random rng;
    random::seed(&rng, seed);

    // Box-Muller transform for Gaussian noise
    for (usz i = 0; i + 1 < latent.len; i += 2) {
        float u1 = random::next_float(&rng);
        float u2 = random::next_float(&rng);
        // Clamp to avoid log(0)
        if (u1 < 1e-7f) u1 = 1e-7f;
        float r = math::sqrt(-2.0f * math::ln(u1));
        float theta = 2.0f * 3.14159265f * u2;
        latent[i] = r * math::cos(theta);
        latent[i + 1] = r * math::sin(theta);
    }
    if (latent.len % 2 == 1) {
        float u1 = random::next_float(&rng);
        float u2 = random::next_float(&rng);
        if (u1 < 1e-7f) u1 = 1e-7f;
        latent[latent.len - 1] = math::sqrt(-2.0f * math::ln(u1)) * math::cos(2.0f * 3.14159265f * u2);
    }
}

// --- DDIM Step on GPU ---

fn void dispatch_ddim_step(
    CommandBuffer cmd,
    DiffusionKernels* k,
    Tensor* noisy,           // current noisy latent (read/write)
    Tensor* predicted_noise, // model output
    uint n_elements,
    float alpha_t,
    float alpha_prev
) {
    DdimStepPC pc = {
        .n = n_elements,
        .sqrt_alpha_t = math::sqrt(alpha_t),
        .sqrt_one_minus_alpha_t = math::sqrt(1.0f - alpha_t),
        .sqrt_alpha_prev = math::sqrt(alpha_prev),
        .sqrt_one_minus_alpha_prev = math::sqrt(1.0f - alpha_prev),
    };
    dispatch_kernel(cmd, &k.ddim_step,
        { noisy.gpu_buffer.buffer, predicted_noise.gpu_buffer.buffer },
        { noisy.size_bytes, predicted_noise.size_bytes },
        &pc, ceil_div(n_elements, 256));
}

// --- Euler Step on GPU ---

fn void dispatch_euler_step(
    CommandBuffer cmd,
    DiffusionKernels* k,
    Tensor* noisy,
    Tensor* predicted_noise,
    uint n_elements,
    float dt
) {
    EulerStepPC pc = {
        .n = n_elements,
        .dt = dt,
    };
    dispatch_kernel(cmd, &k.euler_step,
        { noisy.gpu_buffer.buffer, predicted_noise.gpu_buffer.buffer },
        { noisy.size_bytes, predicted_noise.size_bytes },
        &pc, ceil_div(n_elements, 256));
}

// --- Get alpha values for a timestep ---

fn float SchedulerState.get_alpha_cumprod(&self, float timestep) {
    uint t = (uint)timestep;
    if (t >= self.num_train_timesteps) t = self.num_train_timesteps - 1;
    return self.alphas_cumprod[t];
}

fn float SchedulerState.get_prev_alpha_cumprod(&self, uint step_idx) {
    if (step_idx + 1 >= self.num_inference_steps) {
        return 1.0f;  // Final step, alpha_prev = 1 (fully denoised)
    }
    uint t_prev = (uint)self.timesteps[step_idx + 1];
    if (t_prev >= self.num_train_timesteps) t_prev = self.num_train_timesteps - 1;
    return self.alphas_cumprod[t_prev];
}

// --- Free ---

fn void SchedulerState.free(&self) {
    mem::free(self.alphas_cumprod);
    mem::free(self.timesteps);
}
