module flux_example;

import std::io;
import flux;

fn int main(String[] args) {
    FluxConfig config = parse_flux_args(args)!!;
    flux::run_flux_pipeline(&config)!!;
    return 0;
}
