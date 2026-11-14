#!/usr/bin/env python3
"""Extract ATen op schemas from an installed torch into JSON for
`aten-port-gen` (src/bin/aten_port_gen.rs).

    python tools/aten_extract.py [--torch-python PATH] [--out PATH]

Default torch-python is the graphopt dev venv (torch 2.14). The dump
covers every `aten::` schema from `torch._C._jit_get_all_schemas()` —
canonical, `.out`, inplace (`*_`), and private (`_`-prefixed) variants
all land in the JSON; filtering is the generator's job (it has the
`--include`/`--exclude`/port-list machinery).

Each record:

    {"name": "sum", "overload": "dim_IntList",
     "args": [{"name": "self", "type": "Tensor", "default": null,
               "kwonly": false, "mutated": false}, ...],
     "ret": "Tensor"}

`mutated` marks `Tensor(a!)`-style annotated args — the generator uses
it to drop inplace (`foo_`) and `.out` variants wholesale.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

# Runs inside the *target* interpreter (the one that has torch). Emits
# JSON on stdout; diagnostics on stderr.
_DUMP = r'''
import json, re, sys, torch

def split_top(s, sep=","):
    """Split on `sep` at bracket depth 0."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return out

def parse_arg(text, kwonly):
    # Torch prints "Type name" or "Type name=default" — e.g.
    # "Tensor self", "Tensor? bias=None", "int[2] stride=[1, 1]".
    # The type is everything before the last whitespace-separated token;
    # defaults may contain spaces ("[1, 1]") so split on the FIRST "=".
    if "=" in text:
        head, default = text.split("=", 1)
        default = default.strip()
    else:
        head, default = text, None
    toks = head.split()
    if len(toks) < 2:
        return None
    ty, name = " ".join(toks[:-1]), toks[-1]
    return {
        "name": name,
        "type": ty,
        "default": default,
        "kwonly": kwonly,
        "mutated": "!" in ty,
    }

def parse_schema(s):
    # "aten::name.overload(args) -> ret"  (ret may contain spaces/tuples)
    body = str(s)
    assert body.startswith("aten::"), body
    body = body[len("aten::"):]
    m = re.match(r"^(.*?)\((.*)\)\s*->\s*(.*)$", body, re.S)
    if not m:
        return None
    head, argstr, ret = m.group(1).strip(), m.group(2), m.group(3).strip()
    if "." in head:
        name, overload = head.split(".", 1)
    else:
        name, overload = head, ""
    # Split args on top-level commas; a bare "*" flips kwonly.
    args, kwonly = [], False
    for part in split_top(argstr):
        if part == "*":
            kwonly = True
            continue
        a = parse_arg(part, kwonly)
        if a is not None:
            args.append(a)
    return {"name": name, "overload": overload, "args": args, "ret": ret}

schemas = []
for s in torch._C._jit_get_all_schemas():
    if not str(s).startswith("aten::"):
        continue
    rec = parse_schema(s)
    if rec is not None:
        rec["schema"] = str(s)
        schemas.append(rec)
json.dump({"torch": torch.__version__, "schemas": schemas}, sys.stdout)
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--torch-python",
        default="/home/luis/GitHub/handl-lang/graphopt/.venv/bin/python",
        help="interpreter with torch installed (default: graphopt .venv)",
    )
    ap.add_argument("--out", default="tools/aten_schemas.json")
    args = ap.parse_args()

    proc = subprocess.run(
        [args.torch_python, "-c", _DUMP],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode
    data = json.loads(proc.stdout)
    n = len(data["schemas"])
    with open(args.out, "w") as f:
        json.dump(data, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"wrote {args.out}: {n} aten:: schemas (torch {data['torch']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
