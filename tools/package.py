#!/usr/bin/env python3
"""Lays out a release: the llm-runner binary with its llm folder beside it.

    tools/package.py <out> [plugin ...]              # <out>/llm-runner and <out>/llm/{core, apps, plugins/...}
    tools/package.py --llm-only <out> [plugin ...]   # just <out>/llm/{core, plugins/...}

A program that vendors only the packed library (Rigsmith) gets the llm folder
inside llm.c3l instead: `tools/pack_library.py <out> [plugin ...]`.

`llm/core` (the host API every plugin imports, its kernels, weights, VAEs and
encoders) always goes; the plugins named go into `llm/plugins`. The apps (the
studio, the graph runner) go with the binary, which is what runs them. Nothing
is compiled in: a host finds the folder beside its executable
(`lib/pipeline/plugins.c3`, `plugins_root`) and reads a plugin when a job asks
for it, so users add more by linking their folders into `llm/plugins`. Entries
already in <out>/llm/plugins that are not written here - linked plugins
included - are left as they are.
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM = os.path.join(ROOT, "llm")
KEEP = (".js", ".shady", ".json", ".glb", ".txt", ".md")

args = sys.argv[1:]
llm_only = "--llm-only" in args
args = [a for a in args if a != "--llm-only"]
if not args:
    sys.exit(__doc__)
out, names = os.path.abspath(args[0]), args[1:]
target = os.path.join(out, "llm")


def copy(source, dest):
    if os.path.islink(dest):
        sys.exit(f"{dest} is a link to {os.readlink(dest)}; unlink it first")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=lambda d, entries: [
        e for e in entries if not os.path.isdir(os.path.join(d, e)) and not e.endswith(KEEP)])
    print(dest)


copy(os.path.join(LLM, "core"), os.path.join(target, "core"))
if not llm_only:
    copy(os.path.join(LLM, "apps"), os.path.join(target, "apps"))
os.makedirs(os.path.join(target, "plugins"), exist_ok=True)
for name in names:
    source = os.path.join(LLM, "plugins", name)
    if not os.path.isfile(os.path.join(source, "plugin.json")):
        sys.exit(f"{source} is not a plugin folder (no plugin.json)")
    copy(source, os.path.join(target, "plugins", name))

if not llm_only:
    binary = os.path.join(ROOT, "build", "llm-runner")
    if not os.path.isfile(binary):
        sys.exit(f"{binary} is not built: c3c build llm-runner")
    shutil.copy2(binary, os.path.join(out, "llm-runner"))
    print(os.path.join(out, "llm-runner"))
