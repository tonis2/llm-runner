<*
 The llm folder (`lib/pipeline/plugins.c3`): what it lists, what linking and
 unlinking do and never do, and the sentences a job is refused with.

     c3c test --test-filter plugins
*>
module llm::pipeline::plugins_test;

import llm::pipeline;
import llm::host;
import std::io;
import std::io::file;
import std::io::path;
import std::core::test;

const String ROOT = "build/test_plugins/root";
const String AWAY = "build/test_plugins/away";

fn void make_plugin(String folder, String manifest, bool main = true)
{
	(void)path::mkdir(folder, true);
	(void)file::save(string::tformat("%s/plugin.json", folder), manifest);
	if (main) (void)file::save(string::tformat("%s/main.js", folder), "llm.plugin({});\n");
}

// A root with core (version 2, host verbs 1), an app, a good plugin, a broken
// one and one that needs a newer core; and plugin folders elsewhere to link in.
fn void make_folders()
{
	(void)path::rmtree((Path)"build/test_plugins");
	(void)path::mkdir(ROOT +++ "/core", true);
	(void)file::save(ROOT +++ "/core/llm.js", "export const llm = {};\n");
	(void)file::save(ROOT +++ "/core/core.json", `{ "version": 2, "host": 1 }`);
	(void)file::save(ROOT +++ "/core/plugin.json", `{ "name": "core", "nodes": "nodes.js" }`);
	(void)path::mkdir(ROOT +++ "/apps/studio", true);
	(void)file::save(ROOT +++ "/apps/studio/main.js", "\n");
	make_plugin(ROOT +++ "/plugins/good", `{ "name": "good", "version": "0.3", "description": "fine" }`);
	make_plugin(ROOT +++ "/plugins/broken", `{ "name": `);
	make_plugin(ROOT +++ "/plugins/future", `{ "name": "future", "requires_core": 99 }`);
	make_plugin(AWAY +++ "/mine", `{ "name": "mine", "version": "1.0" }`);
	make_plugin(AWAY +++ "/nomain", `{ "name": "nomain" }`, main: false);
}

fn void a_plugins_folder_lists_its_plugins_and_their_faults() @test
{
	make_folders();
	defer (void)path::rmtree((Path)"build/test_plugins");
	(void)pipeline::link_plugin(ROOT, AWAY +++ "/mine")!!;

	PluginInfo[] found = pipeline::list_plugins(mem, ROOT);
	defer pipeline::free_plugins(mem, found);
	test::@check(found.len == 4, "good, broken, future and the linked one, not core or the apps (%d)", found.len);
	test::@check(found[0].dir == "broken" && found[0].error.len > 0, "a broken manifest is listed with its fault");
	test::@check(found[2].dir == "good" && found[2].version == "0.3" && !found[2].linked, "a plain plugin, by name order");
	test::@check(found[3].dir == "mine" && found[3].linked && found[3].target.ends_with("away/mine"), "the linked one says where it points (%s)", found[3].target);

	pipeline::CoreInfo core = pipeline::core_info(ROOT);
	test::@check(core.found && core.version == 2 && core.host == 1, "core.json is read");
	test::@check(pipeline::list_plugins(mem, "build/test_plugins/nowhere").len == 0, "no folder, no plugins");
}

fn void a_job_is_refused_with_a_reason() @test
{
	make_folders();
	defer (void)path::rmtree((Path)"build/test_plugins");
	test::@check(pipeline::check_plugin(ROOT, "good") == "", "a plugin that fits is not refused");
	test::@check(pipeline::check_plugin(ROOT, "absent").contains("not in"), "a missing plugin is named");
	make_plugin(ROOT +++ "/plugins/nodes", `{ "name": "nodes", "nodes": "nodes.js" }`, main: false);
	test::@check(pipeline::check_plugin(ROOT, "nodes").contains("no main.js"), "a nodes-only plugin is not called missing");
	test::@check(pipeline::check_plugin(ROOT, "future").contains("needs core version 99"), "a plugin needing a newer core");
	test::@check(pipeline::check_plugin(AWAY, "mine").contains("no core/llm.js"), "a folder without core is not an llm folder");
	test::@check(pipeline::check_plugin("", "good").contains("no llm folder"), "no folder set");
	(void)file::save(ROOT +++ "/core/core.json", string::tformat(`{ "version": 2, "host": %d }`, host::API + 1));
	test::@check(pipeline::check_plugin(ROOT, "good").contains("host verbs"), "a core newer than the program's verbs");
}

fn void linking_refuses_what_is_not_a_plugin_or_is_taken() @test
{
	make_folders();
	defer (void)path::rmtree((Path)"build/test_plugins");
	test::@check(@catch(pipeline::link_plugin(ROOT, AWAY +++ "/nomain")) == pipeline::NOT_A_PLUGIN, "no main.js");
	test::@check(@catch(pipeline::link_plugin(ROOT, AWAY)) == pipeline::NOT_A_PLUGIN, "no plugin.json");
	test::@check(@catch(pipeline::link_plugin(ROOT, ROOT +++ "/plugins/good")) == pipeline::PLUGIN_EXISTS, "a name already there");
	String name = pipeline::link_plugin(ROOT, AWAY +++ "/mine/")!!;
	test::@check(name == "mine", "the folder's name, its trailing slash dropped");
	test::@check(pipeline::installed(ROOT, "mine"), "a linked plugin is installed");
	test::@check(@catch(pipeline::link_plugin(ROOT, AWAY +++ "/mine")) == pipeline::PLUGIN_EXISTS, "linked twice");
}

fn void unlinking_takes_only_the_link() @test
{
	make_folders();
	defer (void)path::rmtree((Path)"build/test_plugins");
	(void)pipeline::link_plugin(ROOT, AWAY +++ "/mine")!!;
	test::@check(@ok(pipeline::unlink_plugin(ROOT, "mine")), "a link goes");
	test::@check(!pipeline::installed(ROOT, "mine"), "and the plugin is no longer there");
	test::@check(file::exists(AWAY +++ "/mine/main.js"), "what it pointed at is untouched");
	test::@check(@catch(pipeline::unlink_plugin(ROOT, "good")) == pipeline::PLUGIN_NOT_LINKED, "a real folder is refused");
	test::@check(file::exists(ROOT +++ "/plugins/good/main.js"), "and kept");
	test::@check(@catch(pipeline::unlink_plugin(ROOT, "../away")) == pipeline::PLUGIN_NOT_LINKED, "no path outside the folder");
}

fn void an_explicit_folder_wins_and_a_fallback_is_named() @test
{
	String chosen = pipeline::plugins_root(mem, "/some/where/", "plugins");
	defer free(chosen.ptr);
	test::@check(chosen == "/some/where", "the setting, without its slash (%s)", chosen);
	String fallback = pipeline::plugins_root(mem, "", "llm");
	defer free(fallback.ptr);
	test::@check(fallback == "llm", "nothing beside the test binary, so the fallback (%s)", fallback);
}

fn void the_library_finds_its_own_llm_folder() @test
{
	String found = pipeline::plugins_root(mem, "", "/nowhere");
	defer free(found.ptr);
	test::@check(found.starts_with("/") && found.ends_with("/llm") && pipeline::has_core(found),
		"the llm folder beside the library's sources (%s)", found);
}

fn void a_name_is_an_app_or_a_plugin() @test
{
	make_folders();
	defer (void)path::rmtree((Path)"build/test_plugins");
	test::@check(pipeline::is_app(ROOT, "studio") && !pipeline::is_app(ROOT, "good"), "apps/ holds the apps");
	test::@check(pipeline::entry_of(ROOT, "studio") == "apps/studio/main.js", "an app starts from apps/");
	test::@check(pipeline::entry_of(ROOT, "good") == "plugins/good/main.js", "a plugin from plugins/");
}

fn void linking_makes_the_plugins_folder() @test
{
	make_folders();
	defer (void)path::rmtree((Path)"build/test_plugins");
	(void)path::rmtree((Path)(ROOT +++ "/plugins"));
	String name = pipeline::link_plugin(ROOT, AWAY +++ "/mine")!!;
	test::@check(name == "mine" && pipeline::installed(ROOT, "mine"), "a root without plugins/ gets one for the link");
	test::@check(path::is_link((Path)(ROOT +++ "/plugins/mine")), "and the entry is a link");
}
