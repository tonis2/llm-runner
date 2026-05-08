module flux_example;

import std::io;
import flux;
import server::flux_server;

fn int main(String[] args) {
    ServerOptions server_opts;
    FluxConfig config = parse_flux_args(args, &server_opts)!!;

    if (server_opts.enabled) {
        flux_server::run_server(&config, server_opts.bind_addr, server_opts.port, server_opts.keep_dit)!!;
    } else {
        flux::run_flux_pipeline(&config)!!;
    }
    return 0;
}
