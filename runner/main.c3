<*
llm-runner: one binary, pipelines as JavaScript plugins.

 A plugin is a directory under the plugins root with a `main.js` that registers
 itself with `llm.plugin({ load, generate })` - see `plugins/lib/llm.js`. This
 program brings up a headless GPU, opens the engine's JS runtime with the
 `__llm` host verbs beside `three.compute`, evaluates the plugin, and calls it:

	llm-runner flux --config flux-t2i.json
	llm-runner flux --config flux-t2i.json prompt="a lighthouse at dusk" seed=7
	llm-runner flux --config flux-t2i.json --server --port 7860
	llm-runner tests/kernels.js

 A name is a plugin; a path ending in `.js` is a script run once as a module,
 which is how the kernel checks run. Paths inside the plugin are resolved
 against the plugins root (`--plugins`, default `plugins`), so a plugin imports
 the shared library as `../lib/llm.js`.
*>
module runner;

import three;
import vk;
import std::io;
import std::io::file;
import std::io::path;
import std::core::mem;
import std::collections::list;
import std::encoding::json;
import std::collections::object;

three::Renderer heap_renderer;

struct Options
{
	String target;          // plugin name, or a script path ending in .js
	String plugins;         // the plugins root
	String config;          // a JSON file of settings
	List{String} overrides; // key=value, applied over the file
	bool server;
	String bind;
	uint port;
	bool validate;
}

fn void usage()
{
	io::printn("usage: llm-runner <plugin | script.js> [--config file.json] [key=value ...]");
	io::printn("                  [--plugins dir] [--server [--bind addr] [--port n]] [--validate]");
}

fn Options? parse_args(String[] args)
{
	Options o = { .plugins = "plugins", .bind = "127.0.0.1", .port = 7860 };
	o.overrides.init(mem, 4);
	for (usz i = 1; i < args.len; i++)
	{
		String a = args[i];
		switch
		{
			case a == "--plugins" && i + 1 < args.len: o.plugins = args[++i];
			case a == "--config" && i + 1 < args.len: o.config = args[++i];
			case a == "--server": o.server = true;
			case a == "--bind" && i + 1 < args.len: o.bind = args[++i];
			case a == "--port" && i + 1 < args.len: o.port = args[++i].to_uint() ?? 7860;
			case a == "--validate": o.validate = true;
			case a == "--help" || a == "-h":
				usage();
				return io::EOF~;
			case a.contains("=") && !a.starts_with("-"): o.overrides.push(a);
			case o.target.len == 0 && !a.starts_with("-"): o.target = a;
			default:
				io::printfn("llm-runner: unexpected argument %s", a);
				usage();
				return io::EOF~;
		}
	}
	if (o.target.len == 0)
	{
		usage();
		return io::EOF~;
	}
	while (o.plugins.len > 1 && o.plugins[^1] == '/') o.plugins = o.plugins[..^2];
	return o;
}

<*
 A JSON string literal for `text`, for splicing a value into a script.
*>
fn String js_quote(String text)
{
	DString out;
	out.tinit(text.len + 16);
	out.append_char('"');
	foreach (c : text)
	{
		switch (c)
		{
			case '"': out.append_string("\\\"");
			case '\\': out.append_string("\\\\");
			case '\n': out.append_string("\\n");
			case '\r': out.append_string("\\r");
			case '\t': out.append_string("\\t");
			default:
				if (c < 0x20)
				{
					out.appendf("\\u%04x", (uint)c);
				}
				else
				{
					out.append_char(c);
				}
		}
	}
	out.append_char('"');
	return out.copy_str(tmem);
}

<*
 The settings a plugin gets: the config file's JSON, then every `key=value`
 on top of it. A value that reads as JSON (a number, `true`, a quoted string,
 an array) is taken as that; anything else is a string.
*>
fn String config_script(Options* o)
{
	DString s;
	s.tinit(4096);
	if (o.config.len > 0)
	{
		char[]? text = file::load(tmem, o.config);
		if (catch err = text)
		{
			io::printfn("llm-runner: could not read %s (%s)", o.config, err);
			s.append("const c = {};\n");
		}
		else
		{
			s.appendf("const c = JSON.parse(%s);\n", js_quote((String)text));
		}
	}
	else
	{
		s.append("const c = {};\n");
	}
	foreach (kv : o.overrides)
	{
		sz eq = kv.index_of("=")!!;
		String key = kv[:eq];
		String value = kv[eq + 1..];
		s.appendf("try { c[%s] = JSON.parse(%s); } catch (e) { c[%s] = %s; }\n",
			js_quote(key), js_quote(value), js_quote(key), js_quote(value));
	}
	s.appendf("globalThis.__llm_config = c;\nglobalThis.__llm_root = %s;\n", js_quote(o.plugins));
	return s.str_view();
}

fn void report(three::RunResult* r, String what)
{
	if (r.log.len > 0) io::printn(r.log);
	if (!r.ok)
	{
		io::printfn("llm-runner: %s failed:\n%s", what, r.error);
	}
}

fn int main(String[] args)
{
	Options? parsed = parse_args(args);
	if (catch parsed) return 1;
	Options o = parsed;

	vk::Instance instance = three::create_instance(o.validate, headless: true)!!;
	defer instance.free();

	three::Renderer* renderer = &heap_renderer;
	renderer.init(instance, (vk::SurfaceKHR)null, 64, 64, validate: o.validate)!!;
	defer renderer.free();
	renderer.pass.init(&renderer.gpu, &renderer.target)!!;
	defer renderer.pass.free(&renderer.gpu);
	renderer.pass.camera = three::default_camera();

	// No heap ceiling and no time budget: a plugin holds token lists and reads
	// back tensors, and a generation runs for minutes.
	three::JsRuntime runtime;
	runtime.open(&renderer.pass, renderer, memory_limit: 0)!!;
	defer runtime.close();
	runtime.script_budget_ns = 0;
	install(&runtime);
	defer shutdown();

	io::printfn("llm-runner: %s", ((ZString)&renderer.gpu.limits.device_name).str_view());

	// A script: run it as a module and stop.
	if (o.target.ends_with(".js"))
	{
		String root = o.plugins;
		String specifier = o.target;
		String prefix = string::tformat("%s/", o.plugins);
		if (o.target.starts_with(prefix))
		{
			specifier = o.target[prefix.len..];
		}
		else if (!file::exists(string::tformat("%s/%s", o.plugins, o.target)))
		{
			// Outside the plugins root: its own directory is the root.
			sz? slash = o.target.rindex_of("/");
			if (try slash)
			{
				root = o.target[:slash];
				specifier = o.target[slash + 1..];
			}
			else
			{
				root = ".";
			}
		}
		runtime.set_assets_root(root);
		three::RunResult setup = runtime.run(config_script(&o), "config");
		defer setup.free();
		three::RunResult result = runtime.run_module(specifier);
		defer result.free();
		report(&result, specifier);
		if (result.value.len > 0) io::printfn("=> %s", result.value);
		return result.ok ? 0 : 1;
	}

	runtime.set_assets_root(o.plugins);
	three::RunResult setup = runtime.run(config_script(&o), "config");
	defer setup.free();
	if (!setup.ok)
	{
		report(&setup, "reading the config");
		return 1;
	}

	String entry = string::tformat("%s/main.js", o.target);
	three::RunResult loaded = runtime.run_module(entry);
	defer loaded.free();
	report(&loaded, entry);
	if (!loaded.ok) return 1;

	three::RunResult prepared = runtime.run(`
const p = globalThis.__llm_plugin;
if (!p) throw new Error('the plugin did not register itself - its main.js calls llm.plugin({ ... })');
if (p.load) await p.load(globalThis.__llm_config);
return p.name || 'plugin';
`, "load");
	defer prepared.free();
	report(&prepared, "load");
	if (!prepared.ok) return 1;

	if (o.server)
	{
		return serve(&runtime, o.bind, o.port) ? 0 : 1;
	}

	three::RunResult generated = runtime.run(`
return await globalThis.__llm_plugin.generate(globalThis.__llm_config);
`, "generate");
	defer generated.free();
	report(&generated, "generate");
	if (generated.value.len > 0) io::printfn("=> %s", generated.value);
	return generated.ok ? 0 : 1;
}
