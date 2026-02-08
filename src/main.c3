module llm;

import std::io;

fn int main(String[] args) {
    if (args.len < 2) {
        io::printfn("Usage: llm <model.gguf>");
        return 1;
    }

    String model_path = args[1];

    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    LlamaModel model = load_model(&ctx, model_path)!!;
    defer model.free();

    io::printfn("\nGenerating tokens (greedy, 64 tokens from BOS)...\n");

    uint token = 1;  // BOS token
    io::printf("Token IDs:");
    for (uint pos = 0; pos < 64; pos++) {
        model.forward(token, pos)!!;
        token = argmax_sample(&ctx, &model.acts.logits, model.config.vocab_size)!!;
        io::printf(" %d", token);
        if (token == 2) break;  // EOS
    }
    io::printfn("");

    return 0;
}
