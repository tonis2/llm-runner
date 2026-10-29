#!/usr/bin/env python3
"""Lays out a release: the llm-runner binary with its plugins folder beside it.

    tools/package.py <out> [plugin ...]                  # <out>/llm-runner and <out>/plugins/{lib, plugin ...}
    tools/package.py --plugins-only <out> [plugin ...]   # just <out>/plugins: what a host such as Rigsmith ships

`plugins/lib` (the host API every plugin imports, and its kernels) always goes;
the plugins named go beside it. Nothing is compiled in: a host finds the folder
beside its executable (`lib/pipeline/plugins.c3`, `plugins_root`) and reads a
plugin when a job asks for it, so users add more by linking their folders in.
Folders already in <out>/plugins that are not written here - linked plugins
included - are left as they are.
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGINS = os.path.join(ROOT, "plugins")
KEEP = (".js", ".shady", ".json", ".glb", ".txt", ".md")

args = sys.argv[1:]
plugins_only = "--plugins-only" in args
args = [a for a in args if a != "--plugins-only"]
if not args:
    sys.exit(__doc__)
out, names = os.path.abspath(args[0]), args[1:]

target = os.path.join(out, "plugins")
os.makedirs(target, exist_ok=True)
for name in ["lib"] + names:
    source = os.path.join(PLUGINS, name)
    if not os.path.isdir(source):
        sys.exit(f"no plugin folder {source}")
    if name != "lib" and not os.path.isfile(os.path.join(source, "plugin.json")):
        sys.exit(f"{source} has no plugin.json")
    dest = os.path.join(target, name)
    if os.path.islink(dest):
        sys.exit(f"{dest} is a link to {os.readlink(dest)}; unlink it first")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=lambda d, entries: [
        e for e in entries if not os.path.isdir(os.path.join(d, e)) and not e.endswith(KEEP)])
    print(f"{dest}")

if not plugins_only:
    binary = os.path.join(ROOT, "build", "llm-runner")
    if not os.path.isfile(binary):
        sys.exit(f"{binary} is not built: c3c build llm-runner")
    shutil.copy2(binary, os.path.join(out, "llm-runner"))
    print(os.path.join(out, "llm-runner"))
