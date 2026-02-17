module llm::pipelines;

import std::io;
import std::math;
import std::math::random;
import std::core::mem;
import vk;
import llm;

// Diffusion Schedulers: DDIM and Euler
// For SDXS (1-step distilled model), the scheduler is essentially a single pass

// --- Scheduler Types ---

enum SchedulerType : uint {
    DDIM,
    EULER,
    FLOW_MATCHING,
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

// --- Flow Matching Euler Scheduler ---
// For DiT models using DiscreteFlowDenoiser with time_snr_shift
// Reference: KoboldCpp denoiser.hpp DiscreteFlowDenoiser
// sigma = time_snr_shift(shift, (t+1)/1000) where t is linearly spaced 999->0
// Euler step: x = x + dt * velocity, dt = sigma_{i+1} - sigma_i (negative)

struct FlowMatchingState {
    uint num_steps;
    float[] sigmas;       // [num_steps + 1] sigma values (model timestep = sigma * 1000)
}

fn float time_snr_shift(float alpha, float t) {
    if (alpha == 1.0f) return t;
    return alpha * t / (1.0f + (alpha - 1.0f) * t);
}

fn FlowMatchingState init_flow_matching(uint num_steps, uint image_seq_len = 0) {
    float[] sigmas = mem::new_array(float, (usz)num_steps + 1);

    // Shift factor for sigma schedule:
    // - shift=1.0: identity (linear schedule, used by Z-Image Turbo / diffusers default)
    // - shift=3.0: KoboldCpp default for Flux models
    // For image_seq_len > 0, use shift=1.0 (matching diffusers FlowMatchEulerDiscreteScheduler default)
    float shift = image_seq_len > 0 ? 1.0f : 3.0f;
    float t_max = 999.0f;

    if (num_steps == 1) {
        sigmas[0] = time_snr_shift(shift, (t_max + 1.0f) / 1000.0f);
        sigmas[1] = 0.0f;
    } else {
        float step = t_max / (float)(num_steps - 1);
        for (uint i = 0; i < num_steps; i++) {
            float t = t_max - step * (float)i;
            sigmas[i] = time_snr_shift(shift, (t + 1.0f) / 1000.0f);
        }
        sigmas[num_steps] = 0.0f;
    }

    io::printfn("Flow Matching: %d steps (shift=%.1f)", num_steps, shift);
    io::printf("  sigmas:");
    for (uint i = 0; i <= num_steps; i++) {
        io::printf(" %.3f", sigmas[i]);
    }
    io::printn("");

    return {
        .num_steps = num_steps,
        .sigmas = sigmas,
    };
}

fn float FlowMatchingState.get_dt(&self, uint step) {
    // dt = sigma_{step+1} - sigma_{step} (negative, since sigma decreases)
    return self.sigmas[step + 1] - self.sigmas[step];
}

fn float FlowMatchingState.get_timestep(&self, uint step) {
    // Model receives sigma value (scaled by DIT_T_SCALE=1000 in forward pass)
    return self.sigmas[step];
}

fn void FlowMatchingState.free(&self) {
    mem::free(self.sigmas);
}

// --- Free ---

fn void SchedulerState.free(&self) {
    mem::free(self.alphas_cumprod);
    mem::free(self.timesteps);
}
