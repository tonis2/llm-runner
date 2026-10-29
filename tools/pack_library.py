#!/usr/bin/env python3
"""Packs the `llm` library into one llm.c3l zip, the form a program vendors.

    tools/pack_library.py [out]              # default: build/llm.c3l

c3c takes a zipped .c3l wherever it takes a library folder, and unpacks it under
the build directory. The zip holds manifest.json and every source file its
"sources" globs name (and anything those sources `$embed`, at the same paths, so
the embeds still resolve - today nothing). The plugins are not in it: a program
ships the llm folder (`llm/core`, its plugins) beside its executable and reads
plugins from it at run time (`tools/package.py --llm-only`). `version.txt` says which commit it
was packed from, and whether the tree had changes then.
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
os.chdir(ROOT)

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
    z.writestr("version.txt", version + "\n")
print(f"{out}: {len(sources)} sources, {len(embedded)} embedded files, {version}")
