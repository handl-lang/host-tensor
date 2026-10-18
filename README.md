# host-tensor

Host-side tensor functions for handl runtimes — a dense n-d `f64` array
(row-major) behind the `tensor.*` op surface. Runtimes compose this
component to give `.hnl` code tensor ops; catopt is the first consumer.

The surface (7 FFI atoms, `src/index.hnl` is the single source of truth):

| op | signature | semantics |
|---|---|---|
| `tensor.zeros` | `(List Int) -> Tensor` | zero-filled tensor of `shape` |
| `tensor.ones` | `(List Int) -> Tensor` | one-filled tensor of `shape` |
| `tensor.from_list` | `Array -> Tensor` | nested number arrays → tensor (ragged rejected) |
| `tensor.to_list` | `Tensor -> Array` | tensor → nested number arrays |
| `tensor.shape` | `Tensor -> (List Int)` | extent per axis |
| `tensor.add` | `(Tensor * Tensor) -> Tensor` | elementwise, identical shapes |
| `tensor.matmul` | `(Tensor * Tensor) -> Tensor` | rank-2 `(m×k)·(k×n)`, naive loop |

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

Deliberately out of scope for the first slice — the composition seam is the
point, not BLAS: `tensor.conv2d`/`max_pool`/`batch_norm` (NCHW layout
params), broadcasting `add`/`mul`, `reshape`/`transpose`/`concat`, batched
`matmul`, `tensor.derive` (autodiff value: primal + adjoint for the
optimization-loop host functions), a real `Tensor` scalar in
`value_types.yaml` + `TaggedValue::Tensor` (opaque `Rc` payload instead of
the tagged-object encoding), and dtype promotion beyond `f64`.
