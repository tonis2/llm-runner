module server;

import std::io;
import std::net;
import std::net::tcp;
import server::http;

alias HandlerFn = fn void(net::Socket* sock, http::HttpRequest* req);

// Generic blocking accept loop. Sequential — one request at a time, which is
// what we want for GPU work (no concurrent diffusion). The handler is a
// plain function pointer; callers that need to thread state can stash it in
// a module-level variable.
fn void? serve(String host, uint port, HandlerFn handler)
{
    tcp::TcpServerSocket listen_sock = tcp::listen(host, (int)port, 16, net::SocketOption.REUSEADDR)!;
    defer (void)listen_sock.close();
    io::printfn("[server] listening on %s:%d", host, port);

    while (true)
    {
        tcp::TcpSocket client;
        if (try sock = tcp::accept(&listen_sock))
        {
            client = sock;
        }
        else
        {
            io::printfn("[server] accept() failed, continuing");
            continue;
        }

        if (catch err = process(&client, handler))
        {
            io::printfn("[server] request error: %s", err);
            char[] msg = (char[])"{\"error\":\"internal error\"}";
            http::write_response(&client, 500, "application/json", msg);
        }
        (void)client.close();
    }
}

fn void? process(net::Socket* client, HandlerFn handler)
{
    http::HttpRequest req = http::read_request(client)!;
    defer req.free();
    handler(client, &req);
}
