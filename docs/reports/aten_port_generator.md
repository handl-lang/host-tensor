# ATen → host-tensor port generator

Status: **landed** — `tools/aten_extract.py` + `tools/aten_schemas.json` +
`tools/aten_port_list.txt` + `src/bin/aten_port_gen.rs` +
`src/ops/aten_kernels.rs` + `src/ops/aten_generated.rs` (generated).

## What it does

Mechanizes the hand-porting workflow: an ATen schema (or a curated pin
of one) goes in; a host-tensor `op`/`cap`/`handle` declaration, a
`Function` impl, a `TENSOR_OPS` entry, and an `OpRegistry` registration
come out — with provenance (`aten::<schema>` in every doc comment) and
an honest seam list for anything the f64-dense wire model can't express.

```
torch._C._jit_get_all_schemas()          cargo run --bin aten_port_gen
        │                                            │
        ▼                                            ▼
tools/aten_extract.py              src/ops/aten_generated.rs   (whole file)
        │                          src/index.hnl              (4 marker regions)
tools/aten_schemas.json (3,757)     src/ops/mod.rs             (2 marker regions)
        ▲
tools/aten_port_list.txt (195 curated pins, `# seam:` = declared stub)
```

## Schema source

Primary source is **runtime introspection**, not torch source checkout:
`tools/aten_extract.py` runs under any torch-enabled interpreter
(`graphopt/.venv/bin/python tools/aten_extract.py`) and dumps every
`aten::` schema from `torch._C._jit_get_all_schemas()` — name, overload,
parsed args (type / default / kwonly / `!`-mutation), return type, and
the original schema string — to `tools/aten_schemas.json`.

Inventory used here: **3,757 schemas, torch `2.14.0+cu130`**.

## Curated selection

`tools/aten_port_list.txt` pins `tensor.<hnl-name> = aten::<name>[.<overload>]`
per line. 195 entries covering the high-value surface — unary/parametric
activations, binary math, comparisons, reductions, normalizations,
sort/select, shape ops, indexing, linalg, constructors, metadata, and
one pooling op. `# seam:` entries declare the op but emit a stub that
fails loudly — the honest "needs handwritten semantics" list
(`einsum`, `tensordot`, `kron`, `cross`, tuple-returning reductions like
`median`/`mode`/`kthvalue`/`unique`, RNG constructors, dtype casts,
`meshgrid`, …).

`tensor.unsqueeze` and `tensor.reshape` are deliberately **not** in the
list — they are already ported by hand in `src/ops/shape_ops.rs`, and the
drift guard requires `index.hnl`/`TENSOR_OPS`/registry to stay in
lockstep.

## Generator (`src/bin/aten_port_gen.rs`)

```
cargo run --bin aten_port_gen
    [--schemas tools/aten_schemas.json] [--list tools/aten_port_list.txt]
    [--include GLOB]... [--exclude GLOB]...
    [--emit-spec [PATH]] [--check]
```

- **`--emit-spec`** prints the resolved spec as JSON (schema chosen,
  decoded bindings, kernel call, signature, arity, status, drop
  reasons) without touching the tree — the dry-run/probe mode, in the
  spirit of ext-sdkgen.
- **`--include/--exclude`** glob filters match the HNL op name or the
  `aten::` query independently.
- **`--check`** verifies every generated surface is up-to-date
  (`aten_generated.rs` byte-equal + all six marker regions contain the
  regenerated bodies) — usable as a CI drift gate.
- Writes are **marker-delimited**: `--- ATEN-PORT-GEN:BEGIN|END
  {caps,ops,handles,export}` regions in `src/index.hnl` and
  `// ATEN-PORT-GEN:BEGIN|END {ops,registry}` in `src/ops/mod.rs`.
  `src/ops/aten_generated.rs` is wholly generated (header says so).
  Handwritten regions are never touched; re-running is idempotent.

### Schema → code mapping

For each entry the generator resolves the pinned schema (preferring
non-out, non-mutated, canonical overloads), feature-gates the args,
and emits one `Function`:

- **Typed decode** — schema args become `require_tensor` /
  `opt_tensor` / `req_i64(s)` / `opt_i64(n)` / `opt_dimlist` /
  `req_f64(s)` / `opt_f64(n)` / `req_bool` / `opt_bool(n)` bindings;
  the `KERNEL_MAP` entry names the `aten_kernels::k_*` call. Emit
  modes: `Tensor` (kernel → `Result<Tensor>`, wrapped in
  `encode_tensor`), `Value` (kernel → `Result<Value>`), `Raw`
  (`args: &[Value]` straight to a driver for merged/scalar-union
  overloads).
- **Signatures** — precise HNL tuples when every decoded param is
  required and ≤4 (`(Tensor * Int) -> Tensor`); `Value... -> T`
  otherwise (the `conv2d` convention: `null`/missing selects ATen
  defaults). `map.arity`/`map.sig` overrides cover merged sibling
  overloads (`where` 1-vs-3 args, `arange`/`eye`/`size` arity variants,
  `pow`/comparisons' scalar-tensor unions).
- **Feature gate** — required args of unsupported types
  (`ScalarType`, `Layout`, `Device`, `MemoryFormat`, `Generator`,
  `Dimname`, `str`, `complex`, varargs `...`, `Tensor?[]`) → `Unsupported`;
  *optional* ones are dropped (recorded in the op's doc comment and the
  spec). `.out`/inplace/`Tensor(a!)`-mutated schemas are never chosen.
- **JIT alias annotations** — `Tensor(a)`, `Tensor(a -> *)`,
  `Tensor(a!)` collapse to `Tensor` for classification; `!` still
  flags mutation upstream. Return aliases normalize the same way
  (`Tensor(a)` → `Tensor`, `Tensor(a)[]` → `Array`).
- **`int[]? dim` / `int[1] dim=[]` / bare `int dim`** positions marked
  per-op decode via `opt_dimlist`/`req_dimlist` — accept int *or* list,
  `None`/absent/`[]` selects the ATen default (usually "all dims").

## Kernel layer (`src/ops/aten_kernels.rs`)

~3,000 lines of handwritten semantics the generated glue calls:
decoders, broadcast helpers, ~120 `k_*` entry points covering
reductions (sum/prod/mean/var/std/amax/amin/argmax/argmin/logsumexp/
all/any/cumsum/cumprod/softmax/log_softmax/layer_norm/rms_norm/glu),
~60 unary math/activation kernels, binary/scalar-union drivers
(`pow`, comparisons, `clamp`, `where`, `lerp`, `fmod` family, `xlogy`,
`remainder`), shape/indexing (`flatten`, `expand`, `slice`, `chunk`,
`split`, `stack`, `unbind`, `repeat`, `flip`, `triu/tril`, `diag`,
`diagonal`, `moveaxis`, `rot90`, `pixel_shuffle/unshuffle`,
`constant_pad_nd`), linalg (`mm`/`bmm`/`mv`/`dot`/`vdot`/`ger`/`outer`/
`addmm`/`addbmm`/`baddbmm`/`addr`/`addmv`), constructors
(`full*`/`zeros_like`/`ones_like`/`arange`/`linspace`/`logspace`/`eye`),
metadata (`size`/`stride`/`numel`/`dim`/`item`/`is_contiguous`),
`avg_pool2d`, and sort/topk/nonzero tuple-or-array returns encoded as
JSON records (`{"values","indices"}`) or arrays.

Multi-return ATen tuples are encoded as JSON objects of wire tensors —
the honest representation given the single-`Value` `Function` boundary.

## Results

| status | count | meaning |
|---|---|---|
| Ported | **174** | generated `Function` calls a real kernel |
| Seam | **21** | declared + registered; stub returns a loud `EvalError` |
| Missing | 0 | — |
| Unsupported | 0 | — (all rejected candidates are seam-listed) |

Ops with dropped optional args (documented per-op): dtype/layout/
device/pin_memory selectors on constructors and reductions (18 ops —
the surface is f64 CPU dense; passing e.g. `dtype="float32"` is
silently f64 — a known limitation, not a hidden code path),
`memory_format` on `contiguous`/`clone`, `sparse_grad` on `gather`.

Special cases handled explicitly rather than faked:

- `gelu(approximate=…)` — raw driver selects exact-erf vs tanh
  surrogate; other strings error.
- `expand(implicit=True)` — errors as a seam (arbitrary-growth
  semantics not implemented).
- `prod(x)` vs `prod(x, dim)` — bare-`int` dim decoded dim-like so both
  forms work.
- `elu` honors `scale`/`input_scale`, not just `alpha`.
- `repeat`/`view`/`expand` accept torch's `*sizes` varargs form as well
  as the list form.
- `avg_pool2d` accepts int-or-pair kernel/stride/padding.

## Verification

- `cargo check -p handl-host-tensor` — clean (0 errors; warnings are
  pre-existing dead-code in kernels not yet reachable plus a few
  unused decoder imports).
- `cargo test -p handl-host-tensor` — **87 lib + 5 bin + 12 integration
  tests, all passing**, including:
  - `ops::tests::index_hnl_interface_matches_tensor_ops` — the drift
    guard now covers all 195+28 ops across `cap`/`op`/`handle`/
    `export tensor`/`TENSOR_OPS`/registry;
  - `ops::tests::aten_generated_ports_smoke` — relu, sum-dim, sort
    tuple, mm, nonzero, arange, and a seam stub erroring;
  - 5 `aten_port_gen` unit tests (type classification, default
    parsing, return-alias normalization, glob filters, list parsing).
- `aten_port_gen --check` — all generated surfaces up-to-date.
- `aten_port_gen --emit-spec` — full resolved spec as JSON.

## Known limitations / seams

- **f64 dense only** — no dtype/device/layout/memory-format selection;
  bool-valued results are `1.0`/`0.0`; index outputs (`argsort`,
  `sort.indices`, `argmax`, `nonzero`) are f64 tensors.
- **No autograd, no RNG, no sparse, no complex, no quantization** —
  `to.dtype`, `rand`/`normal`/`empty`, `unique`-family tuple returns,
  `einsum`/`tensordot`/`kron`/`meshgrid`/`corrcoef` etc. are declared
  seams.
- **Inplace/`.out`/`Tensor(a!)` variants are filtered out**, not
  emulated — the wire model is immutable.
- `Cargo.lock` was regenerated by cargo (pre-existing patch-version
  drift, unrelated to this change).

## Regenerating

```sh
# refresh the schema inventory (needs a torch-enabled python)
../graphopt/.venv/bin/python tools/aten_extract.py   # → tools/aten_schemas.json

# regenerate everything
cargo run --bin aten_port_gen

# verify without writing
cargo run --bin aten_port_gen -- --check
cargo run --bin aten_port_gen -- --emit-spec /tmp/spec.json
```
