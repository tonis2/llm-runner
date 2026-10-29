<*
 The llm folder: where a host's pipelines are, read when a job needs one.

 One folder, laid out like a checkout's `llm/`:

     core/      what llm-runner provides - the host API (llm.js, gpu.js, ops.js),
                kernels, weights, VAEs, encoders, graph; `core.json`
     apps/      the studio and the graph runner
     plugins/   one folder per plugin, each with a `plugin.json`

 A release ships `core/` and `apps/` (and whichever plugins it likes) beside
 its executable; a user adds a plugin by linking its folder into `plugins/`.
 Nothing is compiled in, so a plugin edited on disk is what the next job runs.

 **Linking, not copying.** A plugin's folder can live anywhere - a checkout, a
 download - and a symlink in `plugins/` makes it one of the plugins. Plugins
 import core rooted (`/core/llm.js`); the module loader resolves that against
 the llm folder as a string, so it finds core whatever the link points at.

 **Versions.** Plugins and core are not built together. `core/core.json` says
 which core it is and which host verbs it needs (`host`, against
 `llm::host::API`); a plugin may say which core it needs (`requires_core`).
 `check_plugin` refuses what would not run, with a sentence saying why.
*>
module llm::pipeline;

import three;
import llm::host;
import libc;
import std::io;
import std::io::file;
import std::io::path;
import std::core::mem;
import std::core::string;
import std::collections::object;
import std::encoding::json;

faultdef NOT_A_PLUGIN, PLUGIN_EXISTS, PLUGIN_NOT_LINKED, LINK_FAILED;

extern fn CInt symlink(ZString target, ZString link_path) @feat(!WIN32);

<*
 One entry of a plugins folder.
*>
struct PluginInfo
{
	String name;          // plugin.json's "name", else the folder's
	String dir;           // the entry under plugins/: what a job is started with
	String version;
	String description;
	int requires_core;    // 0 when the plugin does not say
	bool linked;          // the entry is a symlink
	String target;        // where a linked entry points
	String error;         // why its manifest could not be read; "" when it could
}

<*
 What `core/core.json` says. `found` is false when the folder has no core.
*>
struct CoreInfo
{
	bool found;
	int version;
	int host;
}

<*
 Whether `root` is an llm folder: it has the host API plugins import.
*>
fn bool has_core(String root) => @pool()
{
	return root.len > 0 && file::exists(string::tformat("%s/core/llm.js", root));
}

<*
 The llm folder a host uses: `explicit` when one is set; else `llm` beside the
 executable, which is where a release puts it; else `fallback`, the host's own
 place (a checkout's `llm`, the copy a program vendors). Owned.

 @param explicit : `The user's setting or a --plugins argument; "" for none.`
 @param fallback : `Where to look last. Returned even when it has no core, so a refusal can name it.`
*>
fn String plugins_root(Allocator allocator, String explicit, String fallback)
{
	if (explicit.len > 0) return trimmed_root(explicit).copy(allocator);
	String beside = plugins_beside_executable(allocator);
	if (beside.len > 0) return beside;
	return trimmed_root(fallback).copy(allocator);
}

// `llm` beside the executable when it is an llm folder, else "". Owned.
fn String plugins_beside_executable(Allocator allocator) @local => @pool()
{
	String? beside = three::executable_dir(tmem);
	if (catch beside) return "";
	String candidate = string::tformat("%s/llm", beside);
	return has_core(candidate) ? candidate.copy(allocator) : "";
}

fn String trimmed_root(String root) @local
{
	String trimmed = root;
	while (trimmed.len > 1 && trimmed[^1] == '/') trimmed = trimmed[..^2];
	return trimmed;
}

<*
 What the folder's `core/core.json` says.
*>
fn CoreInfo core_info(String root) => @pool()
{
	CoreInfo info;
	if (!has_core(root)) return info;
	info.found = true;
	char[]? text = file::load(tmem, string::tformat("%s/core/core.json", root));
	if (catch text) return info;
	Object*? parsed = json::tparse((String)text);
	if (catch parsed) return info;
	info.version = parsed.get_int("version") ?? 0;
	info.host = parsed.get_int("host") ?? 0;
	return info;
}

<*
 Every plugin in `root`'s `plugins/`, by folder name. A folder whose
 `plugin.json` would not read is listed with `error` set; folders without a
 manifest are not plugins. Strings are in `allocator`; `free_plugins` frees them.
*>
fn PluginInfo[] list_plugins(Allocator allocator, String root) => @pool()
{
	if (root.len == 0) return {};
	String folder = plugins_folder(root);
	if (!file::is_dir(folder)) return {};
	PathList? entries = path::ls(tmem, (Path)folder);
	if (catch entries) return {};
	String[] names = mem::temp_array(String, entries.len());
	usz count;
	foreach (entry : entries)
	{
		String name = (String)entry;
		if (name.starts_with(".")) continue;
		if (!file::exists(string::tformat("%s/%s/plugin.json", folder, name))) continue;
		names[count++] = name;
	}
	names = names[:count];
	sort_names(names);
	PluginInfo[] found = alloc::new_array(allocator, PluginInfo, (sz)count);
	foreach (i, name : names) found[i] = read_plugin(allocator, folder, name);
	return found;
}

<*
 Where `root`'s plugins are: `root/plugins`. Temporary.
*>
fn String plugins_folder(String root)
{
	return string::tformat("%s/plugins", trimmed_root(root));
}

fn void sort_names(String[] names) @local
{
	for (usz i = 1; i < names.len; i++)
	{
		for (usz j = i; j > 0 && names[j].compare_to(names[j - 1]) < 0; j--)
		{
			String held = names[j];
			names[j] = names[j - 1];
			names[j - 1] = held;
		}
	}
}

fn PluginInfo read_plugin(Allocator allocator, String folder, String dir) @local => @pool()
{
	String entry = string::tformat("%s/%s", folder, dir);
	PluginInfo info = { .dir = dir.copy(allocator), .linked = path::is_link((Path)entry) };
	info.target = info.linked ? link_target(entry).copy(allocator) : "".copy(allocator);
	String name = dir;
	String version = "";
	String description = "";
	String error = "";
	char[]? text = file::load(tmem, string::tformat("%s/plugin.json", entry));
	if (catch text)
	{
		error = "plugin.json could not be read";
	}
	else
	{
		Object*? parsed = json::tparse((String)text);
		if (catch parsed)
		{
			error = "plugin.json is not JSON";
		}
		else
		{
			name = parsed.get_string("name") ?? dir;
			version = parsed.get_string("version") ?? "";
			description = parsed.get_string("description") ?? "";
			info.requires_core = parsed.get_int("requires_core") ?? 0;
		}
	}
	info.name = name.copy(allocator);
	info.version = version.copy(allocator);
	info.description = description.copy(allocator);
	info.error = error.copy(allocator);
	return info;
}

<*
 Where the symlink at `entry` points, as written; "" when it is not one.
 Temporary.
*>
fn String link_target(String entry)
{
	$if env::WIN32:
		return "";
	$else
		char[4096] buf;
		sz written = libc::readlink(entry.zstr_tcopy(), &buf[0], (int)buf.len - 1);
		if (written <= 0) return "";
		return ((String)buf[:(usz)written]).tcopy();
	$endif
}

fn void free_plugins(Allocator allocator, PluginInfo[] plugins)
{
	foreach (p : plugins)
	{
		alloc::free(allocator, p.name.ptr);
		alloc::free(allocator, p.dir.ptr);
		alloc::free(allocator, p.version.ptr);
		alloc::free(allocator, p.description.ptr);
		alloc::free(allocator, p.target.ptr);
		alloc::free(allocator, p.error.ptr);
	}
	if (plugins.len > 0) alloc::free(allocator, plugins.ptr);
}

<*
 Whether `root` has a plugin folder `plugin` with a main.js to start.
*>
fn bool installed(String root, String plugin) => @pool()
{
	return root.len > 0 && plugin.len > 0 && file::exists(string::tformat("%s/%s/main.js", plugins_folder(root), plugin));
}

<*
 The module a named target starts, root-relative: an app's (`apps/<t>/main.js`)
 when there is one, else the plugin's (`plugins/<t>/main.js`). Temporary.
*>
fn String entry_of(String root, String target)
{
	return is_app(root, target)
		? string::tformat("apps/%s/main.js", target)
		: string::tformat("plugins/%s/main.js", target);
}

<*
 Whether `target` is one of `root`'s apps (the studio, the graph runner)
 rather than a plugin.
*>
fn bool is_app(String root, String target) => @pool()
{
	return target.len > 0 && file::exists(string::tformat("%s/apps/%s/main.js", trimmed_root(root), target));
}

<*
 Why `plugin` in `root` would not run, as a sentence for the user; "" when it
 would. Temporary.
*>
fn String check_plugin(String root, String plugin)
{
	if (!has_core(root))
	{
		return root.len == 0
			? "no llm folder is set"
			: string::tformat("%s is not an llm folder: it has no core/llm.js", root);
	}
	String folder = plugins_folder(root);
	if (!installed(root, plugin))
	{
		if (file::exists(string::tformat("%s/%s/plugin.json", folder, plugin)))
		{
			return string::tformat("the %s plugin has no main.js to run (it may only add graph nodes)", plugin);
		}
		return string::tformat("the %s plugin is not in %s; link its folder there", plugin, folder);
	}
	CoreInfo core = core_info(root);
	if (core.host > host::API)
	{
		return string::tformat("%s/core needs host verbs version %d; this program has %d - update the program or use the core it came with",
			root, core.host, host::API);
	}
	int needs = required_core(folder, plugin);
	if (needs > core.version)
	{
		return string::tformat("the %s plugin needs core version %d; %s/core is version %d", plugin, needs, root, core.version);
	}
	return "";
}

// The core version `plugin`'s manifest asks for; 0 when it asks for none.
fn int required_core(String folder, String plugin) @local => @pool()
{
	char[]? text = file::load(tmem, string::tformat("%s/%s/plugin.json", folder, plugin));
	if (catch text) return 0;
	Object*? parsed = json::tparse((String)text);
	if (catch parsed) return 0;
	return parsed.get_int("requires_core") ?? 0;
}

<*
 Make the plugin at `folder` one of `root`'s: a symlink
 `root/plugins/<folder's name>` pointing at it, `plugins/` made when missing.
 Nothing is copied, so edits to the folder are what runs.

 @param root : `An llm folder.`
 @param folder : `A plugin's folder: it has a plugin.json and a main.js.`
 @return? NOT_A_PLUGIN : `The folder has no plugin.json or no main.js.`
 @return? PLUGIN_EXISTS : `plugins/ already has an entry by that name.`
 @return? LINK_FAILED : `The link could not be made (or the platform has none here).`
 @return `The new entry's name. Temporary.`
*>
fn String? link_plugin(String root, String folder)
{
	String source = trimmed_root(folder);
	if (!file::exists(string::tformat("%s/plugin.json", source))) return NOT_A_PLUGIN~;
	if (!file::exists(string::tformat("%s/main.js", source))) return NOT_A_PLUGIN~;
	usz cut = source.len;
	while (cut > 0 && source[cut - 1] != '/') cut--;
	String name = source[cut..];
	if (name.len == 0 || name.starts_with(".")) return NOT_A_PLUGIN~;
	String plugins = plugins_folder(root);
	String entry = string::tformat("%s/%s", plugins, name);
	if (file::exists(entry) || path::is_link((Path)entry)) return PLUGIN_EXISTS~;
	$if env::WIN32:
		return LINK_FAILED~;
	$else
		if (!file::is_dir(plugins)) (void)path::mkdir(plugins, true);
		Path absolute = source.to_absolute_path(tmem) ?? (Path)source;
		if (symlink(((String)absolute).zstr_tcopy(), entry.zstr_tcopy()) != 0) return LINK_FAILED~;
		return name.tcopy();
	$endif
}

<*
 Take a linked plugin out of `root`'s `plugins/`. Only the link goes: a plugin
 folder that is really there is never deleted, and neither is what a link
 points at.

 @return? PLUGIN_NOT_LINKED : `There is no such entry, or it is a real folder.`
 @return? LINK_FAILED : `The link could not be removed.`
*>
fn void? unlink_plugin(String root, String name)
{
	if (name.len == 0 || name.contains("/")) return PLUGIN_NOT_LINKED~;
	String entry = string::tformat("%s/%s", plugins_folder(root), name);
	if (!path::is_link((Path)entry)) return PLUGIN_NOT_LINKED~;
	if (catch file::delete(entry)) return LINK_FAILED~;
}
