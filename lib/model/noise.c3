module llm;

import std::math;
import std::math::random;

// Standard normal noise for a seed: Box-Muller over Sfc64. The seed-to-noise
// mapping is what the old C3 pipelines drew, so a seed gives the same image.
fn void generate_random_latent(float[] latent, uint seed) {
    Sfc64Random rng;
    random::seed(&rng, seed);

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
