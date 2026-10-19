# host-tensor

Host-side tensor functions for handl runtimes — a dense n-d `f64` array
(row-major) behind the `tensor.*` op surface. Runtimes compose this
component to give `.hnl` code tensor ops; catopt is the first consumer.

The surface (18 FFI atoms, `src/index.hnl` is the single source of truth):

| op | signature | semantics |
|---|---|---|
| `tensor.zeros` | `(List Int) -> Tensor` | zero-filled tensor of `shape` |
| `tensor.ones` | `(List Int) -> Tensor` | one-filled tensor of `shape` |
| `tensor.from_list` | `Array -> Tensor` | nested number arrays → tensor (ragged rejected) |
| `tensor.to_list` | `Tensor -> Array` | tensor → nested number arrays |
| `tensor.shape` | `Tensor -> (List Int)` | extent per axis |
| `tensor.add` | `(Tensor * Tensor) -> Tensor` | elementwise `a + b`, numpy broadcasting |
| `tensor.mul` | `(Tensor * Tensor) -> Tensor` | elementwise `a * b`, numpy broadcasting |
| `tensor.sub` | `(Tensor * Tensor) -> Tensor` | elementwise `a − b`, numpy broadcasting |
| `tensor.div` | `(Tensor * Tensor) -> Tensor` | elementwise `a / b`, numpy broadcasting |
| `tensor.matmul` | `(Tensor * Tensor) -> Tensor` | `(m×k)·(k×n)`; rank-3 batches over `B` (broadcast), naive loop |
| `tensor.conv2d` | `Value... -> Tensor` | NCHW cross-correlation `(input, weight, bias?, stride?, padding?, groups?)` |
| `tensor.max_pool` | `Value... -> Tensor` | NCHW max pooling `(input, kernel, stride?)`; stride defaults to kernel |
| `tensor.batch_norm` | `Value... -> Tensor` | eval-mode per-channel affine `(input, gamma, beta, mean, var, eps?)` |
| `tensor.reshape` | `(Tensor * (List Int)) -> Tensor` | re-label under `shape` (product must match) |
| `tensor.transpose` | `(Tensor * Int * Int) -> Tensor` | swap axes `d0`/`d1` |
| `tensor.concat` | `(Array * Int) -> Tensor` | join a list of same-rank tensors along `axis` |
| `tensor.batch_norm_backward` | `Value... -> Value` | adjoint of `batch_norm`: primal args + `grad_out` → `{input, gamma, beta}` |
| `tensor.derive` | `String -> Value` | registered adjoint op name for `name`, or `null` |

**Broadcasting** — the elementwise four (`add`/`mul`/`sub`/`div`)
follow numpy rules: trailing dims align, a `1`-dim stretches, a missing
leading dim counts as `1` (`[2,3] + [3]` → `[2,3]`; `[2,1] + [1,3]` →
`[2,3]`; rank-0 is the scalar). A dim pair that differs with neither
side `1` is an error, not a coercion. `div` keeps `f64` semantics —
`x/0` is `inf`/`NaN`, matching the scalar `/`.

**Positional optionals** — `conv2d`/`max_pool`/`batch_norm` take optional trailing
parameters; a `null` (or a missing argument) selects the default, so
`conv2d(x, w, null, [2, 2])` is "no bias, stride 2×2". `conv2d` consumes
`input [N, C_in, H, W]` × `weight [C_out, C_in/groups, kH, kW]` (`bias?`
rank-1 `[C_out]`; `stride?`/`padding?` an int or `[h, w]` pair;
`groups?` a positive int — `groups = C_in = C_out` is depthwise).
`batch_norm` computes `(x − μ)/√(σ²+ε)·γ+β` per channel with rank-1 `[C]`
parameters and `eps` defaulting to `1e-5`. `max_pool` pools each channel's
`kH × kW` windows (kernel/stride an int or `[h, w]` pair; floor mode —
trailing partial windows drop). All run naive `f64` loops —
the composition seam, not BLAS.

**Shape manipulation** — `reshape(t, shape)` keeps the row-major
payload and requires the new extents' product to equal the element
count (no `-1` wildcard); `transpose(t, d0, d1)` swaps two axes;
`concat(xs, axis)` joins a non-empty list of same-rank tensors that
agree on every axis but `axis`, whose extents sum — `axis 0` stacks,
`axis rank−1` appends along the fastest-varying axis.

**Batched `matmul`** — ranks are {2, 3} per side: `[B, m, k] ×
[B, k, n] → [B, m, n]` for rank-3, with the batch dim broadcasting (a
rank-2 operand or a `B = 1` batch repeats across the other's batches);
rank-2 × rank-2 stays the plain `(m×k)·(k×n)` product.

**Derivatives** — `tensor.derive(name)` returns the qualified op name
of the adjoint registered for `name` in the `DERIV_PAIRS` table
(qualified or bare local name; `null` when no pair exists), and the
returned name resolves through the same `ModuleRegistry` — so a `.hnl`
optimization loop can dispatch it. Adjoint signatures mirror the primal
plus the upstream gradient and return the differentiable inputs' grads
keyed by parameter name: `batch_norm_backward(input, gamma, beta,
mean, var, grad_out, eps?)` → `{"input": dx, "gamma": dγ, "beta": dβ}`
(`mean`/`var` are statistical in eval mode and carry no grad). The only
pair so far is `tensor.batch_norm → tensor.batch_norm_backward`.

**`tensor.derive` is a registered-pairs table, not autodiff**
(investigated this slice). What a true `derive` would take differs by
target: for *host* ops the table is the honest minimal version — an
adjoint is another `Function` in the same registry, so `derive` is a
name lookup and the pairs are maintained by hand (the
`deriv_pairs_are_all_registered` test pins that every pair's impl
exists). For a `.hnl` *function*'s forward+backward, the host boundary
sees only a name string and evaluated argument values — `EvalContext`
can dispatch `ctx.call_function(name, …)` (so a finite-difference
gradient thunk is *possible*, but numerically approximate and O(inputs)
primal calls per grad), while an exact adjoint needs real autodiff
machinery outside this component: a source-to-source AD pass over the
`.hnl` AST (kit/language layer — the AST never crosses the `Function`
boundary) or a reverse-mode tape in the eval engine over
`Tensor`-aware primitives. Deferred; the pairs table is the surface.

**Tensor wire value** — data on the `serde_json::Value` boundary, tagged in
the `__bytes__` style (ADR-0083 §2.1); a plain object to engines with no
`Tensor` scalar type:

```json
{"__tensor__": {"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6], "dtype": "f64"}}
```

## Composition

The Rust crate (`handl-host-tensor`) ships the `.hnl` interface plus the
native impls; a host wires them into its module chain:

```rust
use std::sync::Arc;
use handl_plugin_lib_handl::HandlModule;
use handl_host_tensor::{TENSOR_HNL, TENSOR_OPS, tensor_op_registry};

let module = HandlModule::new("tensor", TENSOR_HNL, &[])
    .with_ops(TENSOR_OPS, Arc::new(tensor_op_registry()));
// register `module` in the runtime's ModuleRegistry / ModuleRuntime;
// `call_host("tensor.matmul")` now reaches the Rust impl.
```

`.hnl` side (a consumer repo declares the dep in its own `.handl.toml`):

```toml
[dependencies]
host-tensor = { path = "../host-tensor/src" }
```

```hnl
from "host-tensor" import * as tensor   -- or `import tensor` (the record)
tensor.matmul(tensor.from_list([[1, 2]]), tensor.from_list([[3], [4]]))
```

## Layout

```text
src/index.hnl          the interface (cap/op/handle declarations)
src/ops/mod.rs         TENSOR_OPS + tensor_op_registry + the interface drift guard
src/ops/codec.rs       the wire codec + boundary ops (zeros/ones/from_list/to_list/shape)
src/ops/elementwise.rs add/mul/sub/div (numpy broadcasting)
src/ops/matmul.rs      matmul (rank-2/3, batch broadcast)
src/ops/conv.rs        conv2d, max_pool (NCHW)
src/ops/norm.rs        batch_norm, batch_norm_backward
src/ops/shape_ops.rs   reshape, transpose, concat
src/ops/derive.rs      derive + DERIV_PAIRS
src/tensor.test.hnl    .hnl spec tests (host-dependent under `hnl test`)
pkg.hnl                { modules: ["tensor"] }
.handl.toml            self-declared dep names for repo-local resolution
ci.sh                  bupha entry: hnl gate + cargo test
```

The `src/ops/` file map mirrors `lib-tensor`'s spec files 1:1
(`codec`/`elementwise`/`matmul`/`conv`/`norm`/`shape_ops`; `derive` splits
out of `norm` since `DERIV_PAIRS` spans domains) — the same op lives in
the same file on both sides of the pair.

## Gate

```bash
HNL=../host-cli/target/debug/hnl ./ci.sh   # hnl typecheck/lint/test + cargo test
cargo test                            # the Rust suite (codec + ops + dispatch)
```

## Follow-up (the catopt-shaped surface)

Deliberately out of scope so far — the composition seam is the
point, not BLAS: ~~`tensor.max_pool` and friends (NCHW layout
params)~~, ~~broadcasting `add`/`mul`/`sub`/`div`~~,
~~`reshape`/`transpose`/`concat`~~, ~~batched `matmul`~~,
~~`tensor.derive`~~ (landed as a registered-pairs table + one adjoint —
`.hnl`-function autodiff is the deferred half; see the finding below),
~~`TaggedValue::Tensor`~~ (landed — see the finding below), and dtype
promotion beyond `f64`.

**`TaggedValue::Tensor` landed in `handl-value`, not in
`value_types.yaml`** (the earlier deferral, resolved). The investigation
had found `value_types.yaml` to be inert — ADR-0078 declarative spec data
that `handl-kit-data-canonical` does not load and `Language` has no
`value_types` field to merge — and the runtime value representation to be
the hardcoded `TaggedValue` enum. The kernel slice landed on that second
path: `TaggedValue::Tensor(Rc<TensorValue>)` in `handl-value` (opaque
`Rc` payload, following the `TaggedValue::Resource` precedent of
ADR-0083 §2.2), with decode/encode in the `TaggedValue`↔`Value` codec —
`call_host` decodes the `{"__tensor__": …}` wire form at the boundary and
engines carry it opaquely (`tests/engine_roundtrip.rs` pins the
round-trip on tree-walk and bytecode). The wire encoding is unchanged —
it stays the `Function`-boundary form, and engines without the variant
still see a plain object. One asymmetry to note: inside an engine the
variant carries non-finite `f64`s, but any `serde_json` crossing still
serializes them as `null` (the codec's documented finite-only hole).
