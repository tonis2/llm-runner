module llm;

import std::io;
import std::io::file;
import std::io::file::mmap;
import std::core::mem;

fn int main(String[] args) {
    if (args.len < 2) {
        io::printfn("Usage: llm <model.gguf> [prompt]");
        return 1;
    }

    String model_path = args[1];
    io::printfn("Loading model from: %s", model_path);

    // Memory map the GGUF file
    mmap::FileMmap mm = file::mmap_open(model_path, "rb")!!;
    char[] data = mm.bytes();
    io::printfn("File mapped: %d bytes", data.len);

    GGUFFile gf = gguf_parse(data)!!;
    defer gf.free();
    gguf_print_info(&gf);

    DeviceContext ctx = createContext()!!;
    defer ctx.free();

    LlamaModel model = load_model(&ctx, &gf)!!;
    defer model.free();

    Tokenizer tok = load_tokenizer(&gf)!!;
    defer tok.free();

    // Encode prompt if provided
    uint[] prompt_tokens;
    usz n_prompt = 0;
    if (args.len >= 3) {
        prompt_tokens = tok.encode(args[2])!!;
        n_prompt = prompt_tokens.len;
        io::printfn("Prompt encoded to %d tokens", n_prompt);
    }

    io::printfn("\nGenerating (greedy, max 256 tokens)...\n");

    // Echo prompt text
    if (args.len >= 3) {
        io::printf("%s", args[2]);
    }

    // Prefill: BOS + prompt tokens
    model.forward(1, 0)!!;  // BOS at position 0
    for (usz i = 0; i < n_prompt; i++) {
        model.forward(prompt_tokens[i], (uint)(i + 1))!!;
    }

    // Decode: sample and generate
    uint next_pos = (uint)(n_prompt + 1);
    char[256] decode_buf;
    for (uint g = 0; g < 256; g++) {
        uint token = argmax_sample(&ctx, &model.acts.logits, model.config.vocab_size)!!;
        if (token == 2) break;  // EOS
        String text = tok.decode_token(token, &decode_buf);
        if (text.len > 0) {
            io::printf("%s", text);
        }
        model.forward(token, next_pos)!!;
        next_pos++;
    }
    io::printfn("");

    if (n_prompt > 0) mem::free(prompt_tokens);

    return 0;
}
