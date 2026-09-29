<*
 Running a pipeline from another C3 program: what the `llm-runner` executable
 does for one generation, as calls a host program makes on its own GPU.

	llm::pipeline::Job* job = mem::new(llm::pipeline::Job);
	job.start(&renderer, "kimodo", config_json, plugins_root)!;
	while (job.pump()) { draw a frame }        // one call a frame
	if (job.state == DONE) use(job.result());  // the plugin's result, as JSON
	job.close();
	mem::free(job);

 **Nothing blocks for the whole run.** A plugin breathes between its blocks of
 GPU work (`plugins/lib/gpu.js`, `breathe`), and here a breath waits for the next
 `pump` - so a program that pumps once a frame keeps drawing while a generation
 runs, the way the studio does. What does block is whatever a plugin does
 between two breaths: opening a model file and uploading its weights is the long
 one.

 **The plugins travel with the library.** The JavaScript and the kernels are
 embedded (`bundle.c3`, written by `tools/bundle_plugins.py`), so a program that
 links `llm.c3l` - a folder or a zip - needs no plugins folder of its own:
 `extract` writes them into one it chooses, and a job runs from there.

 **One job at a time per process.** The `__llm` host verbs keep their open models
 and tokenizers in module globals (`llm::host`), so a second job would share
 them; `start` refuses it.

 A job has a JavaScript runtime of its own on the caller's renderer, separate
 from any the program already runs: its module root is the plugins folder and it
 has no time budget, neither of which a program's own scripting wants.
*>
module llm::pipeline;

import three;
import qjs;
import llm::host;
import std::io;
import std::io::file;
import std::io::path;
import std::core::mem;
import std::core::string;
import std::time::clock;

faultdef JOB_BUSY, JOB_NOT_STARTED;

<*
 One file the library carries: its path under the plugins root, and its bytes.
*>
struct BundledFile
{
	String path;
	char[] bytes;
}

enum JobState
{
	IDLE,
	RUNNING,
	DONE,
	FAILED,
}

alias LineHook = fn void(void* context, String line);

<*
 One generation. Must not move once started - its runtime is addressed by the
 engine - so it lives on the heap or in a place that outlives the run.
*>
struct Job
{
	three::JsRuntime runtime;
	three::Renderer* renderer;
	bool open;
	JobState state;
	// The plugin's result as JSON once DONE; why it failed once FAILED.
	DString outcome;
	// Every `llm.print` line, as it is printed. Optional.
	LineHook on_line;
	void* line_context;
}

// The job running now: the host verbs have one set of tables.
Job* active;

<*
 Write the bundled plugin files under `directory`, leaving any already there
 byte for byte. Cheap enough to call before every job.
*>
fn void? extract(String directory)
{
	foreach (bundled : bundled_files())
	{
		@pool()
		{
			String target = string::tformat("%s/%s", directory, bundled.path);
			char[]? held = file::load(tmem, target);
			if (try held && (String)held == (String)bundled.bytes) continue;
			sz? slash = target.rindex_of_char('/');
			if (try slash) (void)path::mkdir((Path)target[:(usz)slash], recursive: true);
			file::save(target, bundled.bytes)!;
		};
	}
}

<*
 Whether the library carries `plugin`.
*>
fn bool bundled(String plugin)
{
	foreach (name : BUNDLED_PLUGINS)
	{
		if (name == plugin) return true;
	}
	return false;
}

fn String Job.result(&self) => self.state == DONE ? self.outcome.str_view() : "";
fn String Job.error(&self) => self.state == FAILED ? self.outcome.str_view() : "";

<*
 Start `plugin`'s generate on `config` (a JSON object, the settings the runner
 reads from its config file). Returns at once; `pump` does the work.

 @param renderer : `The caller's; its device runs the kernels.`
 @param plugin : `A folder under plugins_root with a main.js.`
 @param config : `A JSON object.`
 @param plugins_root : `Where the plugins are: extract's directory, or a checkout's plugins folder.`
 @return? JOB_BUSY : `Another job is open in this process.`
 @return? JOB_NOT_STARTED : `The runtime would not open or the plugin would not load; error() says why.`
*>
fn void? Job.start(&self, three::Renderer* renderer, String plugin, String config, String plugins_root,
	LineHook on_line = null, void* line_context = null)
{
	if (active != null) return JOB_BUSY~;
	*self = { .renderer = renderer, .on_line = on_line, .line_context = line_context };
	self.outcome.init(mem, 256);

	// No scene: a pipeline dispatches kernels and draws nothing.
	if (catch err = self.runtime.open(null, renderer, memory_limit: 0))
	{
		self.state = FAILED;
		self.outcome.appendf("could not open a script runtime: %s", err);
		return JOB_NOT_STARTED~;
	}
	self.open = true;
	active = self;
	self.runtime.script_budget_ns = 0;
	self.runtime.set_assets_root(plugins_root);
	host::install(&self.runtime);
	host::print_hook = &job_print;
	host::print_context = self;

	qjs::Value global = self.runtime.js.global();
	self.runtime.js.set_property(global, "__llm_job_done", self.runtime.js.function(&js_job_done, "done", 2, &self.runtime));
	self.runtime.js.release(global);

	three::RunResult started = self.runtime.run(kickoff_script(plugin, config, plugins_root), "llm:start", drain_frame: false);
	defer started.free();
	self.state = RUNNING;
	if (!started.ok)
	{
		self.state = FAILED;
		self.outcome.clear();
		self.outcome.append(started.error);
		return JOB_NOT_STARTED~;
	}
}

<*
 The script that loads the plugin and sets its generate going without waiting
 for it: the work starts on the first `pump`, and ends in `__llm_job_done`.
*>
fn String kickoff_script(String plugin, String config, String plugins_root) @local
{
	DString s;
	s.tinit(1024 + config.len);
	s.appendf("globalThis.__llm_root = %s;\n", quote(plugins_root));
	s.appendf("globalThis.__llm_config = JSON.parse(%s);\n", quote(config));
	s.appendf("globalThis.__llm_cancel = false;\n");
	s.append(`const gpu = await import('lib/gpu.js');
gpu.setBreather(() => {
	if (globalThis.__llm_cancel) throw new Error('stopped');
	return three.nextFrame();
});
`);
	s.appendf("await import(%s);\n", quote(string::tformat("%s/main.js", plugin)));
	s.append(`const p = globalThis.__llm_plugin;
if (!p) throw new Error('the plugin did not register itself - its main.js calls llm.plugin({ ... })');
(async () => {
	await three.nextFrame();
	try {
		const c = globalThis.__llm_config;
		if (p.load) await p.load(c);
		const r = await p.generate(c);
		__llm_job_done(true, JSON.stringify(r ?? null));
	} catch (e) {
		__llm_job_done(false, String((e && e.message) || e));
	}
})();
return p.name || 'plugin';
`);
	return s.str_view();
}

<*
 One frame's turn: wake the plugin from its breath and run it until it breathes
 again or `budget_ns` is spent.

 @return `True while the job still runs.`
*>
fn bool Job.pump(&self, long budget_ns = 12_000_000)
{
	if (self.state != RUNNING) return false;
	self.runtime.budget_ns = 0;
	self.runtime.wake_frame_waiters();
	Clock began = clock::now();
	while (self.state == RUNNING && self.runtime.js.jobs_waiting())
	{
		self.runtime.js.step_job();
		if ((long)began.to_now() >= budget_ns) break;
	}
	return self.state == RUNNING;
}

<*
 Ask the plugin to stop at its next breath; keep pumping until it has. `close`
 stops it at once instead.
*>
fn void Job.cancel(&self)
{
	if (self.state != RUNNING) return;
	qjs::Value global = self.runtime.js.global();
	self.runtime.js.set_property(global, "__llm_cancel", qjs::bool_value(true));
	self.runtime.js.release(global);
}

<*
 End the job, running or not: the device is let finish what it was given, then
 the models, the tokenizers and the runtime go.
*>
fn void Job.close(&self)
{
	if (self.open)
	{
		if (self.renderer != null) self.renderer.gpu.wait_idle();
		host::print_hook = null;
		host::print_context = null;
		host::shutdown();
		self.runtime.close();
		self.open = false;
	}
	if (self.state == RUNNING)
	{
		self.state = FAILED;
		self.outcome.clear();
		self.outcome.append("stopped");
	}
	if (active == self) active = null;
}

<*
 Free what `close` leaves: the outcome text.
*>
fn void Job.free(&self)
{
	self.close();
	self.outcome.free();
}

fn void job_print(void* context, String line) @local
{
	Job* job = context;
	if (job.on_line) job.on_line(job.line_context, line);
}

fn void js_job_done(qjs::ContextRef ctx, qjs::Value* this_val, int argc, qjs::Value* args, void* opaque, qjs::Value* out) @local
{
	Job* job = active;
	if (job == null || job.state != RUNNING) return;
	qjs::Context js = qjs::wrap(ctx);
	bool ok = argc > 0 && js.truth(args[0]);
	job.outcome.clear();
	job.outcome.append(three::arg_text(&js, argc, args, 1, tmem));
	job.state = ok ? DONE : FAILED;
}

<*
 A JavaScript string literal for `text`.
*>
fn String quote(String text)
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
	return out.str_view();
}
