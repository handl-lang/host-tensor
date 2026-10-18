# host-tensor

Host-side tensor functions for handl runtimes — a dense n-d `f64` array
(row-major) behind the `tensor.*` op surface. Runtimes compose this
component to give `.hnl` code tensor ops; catopt is the first consumer.

The surface (10 FFI atoms, `src/index.hnl` is the single source of truth):

| op | signature | semantics |
|---|---|---|
| `tensor.zeros` | `(List Int) -> Tensor` | zero-filled tensor of `shape` |
| `tensor.ones` | `(List Int) -> Tensor` | one-filled tensor of `shape` |
| `tensor.from_list` | `Array -> Tensor` | nested number arrays → tensor (ragged rejected) |
| `tensor.to_list` | `Tensor -> Array` | tensor → nested number arrays |
| `tensor.shape` | `Tensor -> (List Int)` | extent per axis |
| `tensor.add` | `(Tensor * Tensor) -> Tensor` | elementwise, identical shapes |
| `tensor.matmul` | `(Tensor * Tensor) -> Tensor` | rank-2 `(m×k)·(k×n)`, naive loop |
| `tensor.conv2d` | `Value... -> Tensor` | NCHW cross-correlation `(input, weight, bias?, stride?, padding?, groups?)` |
| `tensor.max_pool` | `Value... -> Tensor` | NCHW max pooling `(input, kernel, stride?)`; stride defaults to kernel |
| `tensor.batch_norm` | `Value... -> Tensor` | eval-mode per-channel affine `(input, gamma, beta, mean, var, eps?)` |

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
src/index.hnl        the interface (cap/op/handle declarations)
src/ops.rs           the Rust atoms + tensor_op_registry
src/tensor.test.hnl  .hnl spec tests (host-dependent under `hnl test`)
pkg.hnl              { modules: ["tensor"] }
.handl.toml          self-declared dep names for repo-local resolution
ci.sh                bupha entry: hnl gate + cargo test
```

## Gate

```bash
HNL=../cli/target/debug/hnl ./ci.sh   # hnl typecheck/lint/test + cargo test
cargo test                            # the Rust suite (codec + ops + dispatch)
```

## Follow-up (the catopt-shaped surface)

Deliberately out of scope so far — the composition seam is the
point, not BLAS: ~~`tensor.max_pool` and friends (NCHW layout
params)~~, broadcasting `add`/`mul`, `reshape`/`transpose`/`concat`, batched
`matmul`, `tensor.derive` (autodiff value: primal + adjoint for the
optimization-loop host functions), a real `Tensor` scalar in
`value_types.yaml` + `TaggedValue::Tensor` (opaque `Rc` payload instead of
the tagged-object encoding — see the finding below), and dtype promotion
beyond `f64`.

**`TaggedValue::Tensor` is a kernel change, not a data change**
(investigated for this slice; deferred). `value_types.yaml` is ADR-0078
declarative spec data: `handl-kit-data-canonical` does not load it and
`handl-language`'s `Language` has no `value_types` field — there is no
`ValueTypeDef` fragment type to merge into, so an entry there would be
inert. The runtime value representation is the hardcoded
`TaggedValue` enum in `handl-value`; a real `Tensor` variant means a new
enum case + eq/hash + the `TaggedValue`↔`Value` codec + every `match`
site in the engines. The opaque-payload precedent already exists —
`TaggedValue::Resource(Rc<ForeignResource>)` (ADR-0083 §2.2, pointer +
finalizer) — and the `Function` boundary is `serde_json::Value`
regardless, so the `__tensor__` tagged-object encoding stays correct and
forward-compatible until that kernel slice lands.
