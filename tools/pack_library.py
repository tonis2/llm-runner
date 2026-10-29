#!/usr/bin/env python3
"""Packs the `llm` library into one llm.c3l zip, the form a program vendors.

    tools/pack_library.py [out] [plugin ...]   # default: build/llm.c3l, no plugins
    tools/pack_library.py ../rigsmith/libs/llm.c3l kimodo creature

c3c takes a zipped .c3l wherever it takes a library folder, and unpacks it under
the build directory. The zip holds manifest.json and every source file its
"sources" globs name (and anything those sources `$embed`, at the same paths, so
the embeds still resolve - today nothing). It also carries the llm folder as
data, under `llm/`: `llm/core` always, and `llm/plugins/<name>` for each plugin
named. Nothing of it is compiled in: c3c unpacks the whole zip under the build
directory and compiles the library from there, so the library finds the folder
beside its own sources (`plugins_root`) with no path from the program, and a
release copies it into its bundle - one file is all a program vendors. `version.txt` says which commit it was packed from, and
whether the tree had changes then.
"""
import glob
import json
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "llm.c3l"))
plugins = sys.argv[2:]
os.chdir(ROOT)

# The llm folder's files: core, and the plugins named. Text and the assets
# plugins read (rest skeletons), not caches or test output.
KEEP = (".js", ".shady", ".json", ".glb", ".txt", ".md")
carried = []
for folder in ["llm/core"] + [f"llm/plugins/{name}" for name in plugins]:
    if not os.path.isdir(folder):
        sys.exit(f"no folder {folder}")
    if folder != "llm/core" and not os.path.isfile(os.path.join(folder, "plugin.json")):
        sys.exit(f"{folder} is not a plugin folder (no plugin.json)")
    for dirpath, _, files in os.walk(folder, followlinks=True):
        carried.extend(os.path.join(dirpath, f) for f in files if f.endswith(KEEP))

# Comments are allowed in c3c's JSON; the manifest has none today.
manifest = json.load(open("manifest.json"))
sources = set()
for pattern in manifest["sources"]:
    # c3c's `dir/**` is every .c3 file under dir.
    base = pattern[:-3] if pattern.endswith("/**") else pattern
    if base != pattern:
        sources.update(glob.glob(os.path.join(base, "**", "*.c3"), recursive=True))
    else:
        sources.update(glob.glob(pattern))

embedded = set()
for source in sources:
    for rel in re.findall(r'\$embed\(\s*"([^"]+)"', open(source, encoding="utf-8").read()):
        path = os.path.normpath(os.path.join(os.path.dirname(source), rel))
        if not os.path.isfile(path):
            sys.exit(f"{source} embeds {rel}, which is not there")
        embedded.add(path)

def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

version = git("rev-parse", "--short", "HEAD") or "unknown"
if git("status", "--porcelain"):
    version += " (with uncommitted changes)"

os.makedirs(os.path.dirname(out), exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("manifest.json")
    for path in sorted(sources | embedded):
        z.write(path)
    for path in sorted(carried):
        z.write(path)
    z.writestr("version.txt", version + "\n")
print(f"{out}: {len(sources)} sources, {len(embedded)} embedded files, "
      f"{len(carried)} llm files (core{''.join(', ' + p for p in plugins)}), {version}")
