<*
The runner's HTTP mode: every request goes to the plugin.

 The accept loop and the request parser are `server/`'s, shared with the C3
 Flux server. What is different is who answers: the request is handed to
 `plugin.handle({ method, path, body })` - or, for a plugin with no `handle`,
 to `generate` with the JSON body over the configured settings - and the
 plugin says what goes back with `llm.respond(status, type, body)`. So the
 A1111-style `/sdapi/v1/img2img` a Flux plugin speaks is the plugin's, and a
 text model's plugin can speak something else without this file changing.

 One request at a time, because there is one GPU and a generation holds it.
*>
module runner;

import three;
import server;
import server::http;
import std::io;
import std::net;
import std::core::mem;
import std::encoding::base64;
import qjs;

three::JsRuntime* serving;
http::HttpRequest* current;

// What the plugin answered with, collected by `respond`.
int reply_status;
DString reply_type;
DString reply_body;
bool replied;

fn bool serve(three::JsRuntime* runtime, String bind, uint port)
{
	serving = runtime;
	reply_type.init(mem, 64);
	reply_body.init(mem, 4096);
	defer { reply_type.free(); reply_body.free(); }

	qjs::Value global = runtime.js.global();
	defer runtime.js.release(global);
	qjs::Value host = runtime.js.property(global, "__llm");
	defer runtime.js.release(host);
	runtime.bind(host, "request", &js_request, 0);
	runtime.bind(host, "respond", &js_respond, 3);

	if (catch err = server::serve(bind, port, &handle))
	{
		io::printfn("llm-runner: could not serve on %s:%d (%s)", bind, port, err);
		return false;
	}
	return true;
}

fn void handle(net::Socket* sock, http::HttpRequest* req)
{
	current = req;
	defer current = null;
	replied = false;
	reply_status = 200;
	reply_type.clear();
	reply_body.clear();

	three::RunResult r = serving.run(`
const p = globalThis.__llm_plugin;
const req = __llm.request();
if (p.handle) {
	await p.handle(req);
} else {
	let body = {};
	try { body = req.body.length ? JSON.parse(req.body) : {}; } catch (e) { __llm.respond(400, 'application/json', JSON.stringify({ error: 'invalid json body' })); return; }
	const result = await p.generate({ ...globalThis.__llm_config, ...body });
	__llm.respond(200, 'application/json', JSON.stringify(result ?? {}));
}
`, "request");
	defer r.free();
	if (r.log.len > 0) io::printn(r.log);

	if (!r.ok)
	{
		io::printfn("llm-runner: request failed:\n%s", r.error);
		DString err;
		err.tinit(256);
		err.appendf(`{"error":%s}`, js_quote(r.error));
		http::write_response(sock, 500, "application/json", err.str_view());
		return;
	}
	if (!replied)
	{
		http::write_response(sock, 204, "application/json", {});
		return;
	}
	http::write_response(sock, reply_status, reply_type.str_view(), reply_body.str_view());
}

<*
 `request()` - the request being answered: `{ method, path, body }`, the body
 as text.
*>
fn void js_request(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	if (current == null)
	{
		*out = qjs::null_value();
		return;
	}
	qjs::Value r = js.object();
	js.set_property(r, "method", js.string(current.method));
	js.set_property(r, "path", js.string(current.path));
	js.set_property(r, "body", js.string((String)current.body));
	*out = r;
}

<*
 `respond(status, contentType, body)` - what goes back. `body` is text or bytes.
*>
fn void js_respond(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	qjs::Context js = qjs::wrap(ctx);
	reply_status = three::arg_int(&js, argc, args, 0);
	reply_type.clear();
	reply_type.append(three::arg_text(&js, argc, args, 1, tmem));
	reply_body.clear();
	if (argc >= 3)
	{
		char[] bytes = js.bytes(args[2]);
		if (bytes.len > 0)
		{
			reply_body.append((String)bytes);
		}
		else
		{
			reply_body.append(three::arg_text(&js, argc, args, 2, tmem));
		}
	}
	replied = true;
}
