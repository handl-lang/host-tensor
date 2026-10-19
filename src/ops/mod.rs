//! Rust `Function` impls for the `tensor` module — the handls (handlers)
//! for the `op`/`handle` declarations in `src/index.hnl`.
//!
//! All 24 functions are FFI atoms: a tensor has no literal syntax and no
//! eval-vocabulary constructor, so construction, projection, and algebra
//! cannot be expressed as pure handl arrows. The impls here are the
//! host-side half runtimes compose: a consumer registers them in an
//! [`OpRegistry`](handl_plugin_lib_handl::OpRegistry) (via
//! [`tensor_op_registry`]) and wires them onto a `HandlModule` with
//! `with_ops(TENSOR_OPS, …)`, after which `.hnl` `tensor.*` calls dispatch
//! through the module chain into these impls.
//!
//! ## Layout (the module map mirrors lib-tensor's spec files 1:1)
//!
//! | file | ops | spec |
//! |---|---|---|
//! | [`codec`] | the wire codec + `zeros`/`ones`/`from_list`/`to_list`/`shape` | `codec.hnl` |
//! | [`elementwise`] | `add`/`mul`/`sub`/`div` + their `*_backward` adjoints | `elementwise.hnl` |
//! | [`matmul`] | `matmul` (rank-2/3, batch broadcast) + `matmul_backward` | `matmul.hnl` |
//! | [`conv`] | `conv2d`, `max_pool`, `conv2d_backward` (NCHW) | `conv.hnl` |
//! | [`norm`] | `batch_norm`, `batch_norm_backward` | `norm.hnl` |
//! | [`shape_ops`] | `reshape`, `transpose`, `concat` | `shape_ops.hnl` |
//! | [`derive`] | `derive` + `DERIV_PAIRS` (7 pairs) | `norm.hnl`'s `derive` |
//!
//! ## The tensor value
//!
//! Tensors cross the `Function` boundary as
//! `{"__tensor__": {"shape": [d, ...], "data": [x, ...], "dtype": "f64"}}`
//! — a tagged plain object in the style of `{"__bytes__": [n, ...]}`
//! (ADR-0083 §2.1). `shape` is the extent per axis (rank = `shape.len()`),
//! `data` the flat row-major payload (`data.len() == product(shape)`),
//! `dtype` the element type — optional on decode, `"f64"` the only value
//! accepted for now. Engines with no `Tensor` scalar see an ordinary
//! object; engines with the kernel `TaggedValue::Tensor` decode the tag
//! at the `call_host` boundary and carry the tensor opaquely.

use std::sync::Arc;

use handl_plugin_lib_handl::OpRegistry;

pub mod codec;
pub mod conv;
pub mod derive;
pub mod elementwise;
pub mod matmul;
pub mod norm;
pub mod shape_ops;

pub use codec::{FromList, Ones, Shape, ToList, Zeros};
pub use conv::{Conv2d, Conv2dBackward, MaxPool};
pub use derive::{DERIV_PAIRS, Derive};
pub use elementwise::{Add, AddBackward, Div, DivBackward, Mul, MulBackward, Sub, SubBackward};
pub use matmul::{Matmul, MatmulBackward};
pub use norm::{BatchNorm, BatchNormBackward};
pub use shape_ops::{Concat, Reshape, Transpose};

/// The op names declared in `src/index.hnl`, in declaration order.
pub const TENSOR_OPS: &[&str] = &[
    "tensor.zeros",
    "tensor.ones",
    "tensor.from_list",
    "tensor.to_list",
    "tensor.shape",
    "tensor.add",
    "tensor.add_backward",
    "tensor.mul",
    "tensor.mul_backward",
    "tensor.sub",
    "tensor.sub_backward",
    "tensor.div",
    "tensor.div_backward",
    "tensor.matmul",
    "tensor.matmul_backward",
    "tensor.conv2d",
    "tensor.conv2d_backward",
    "tensor.max_pool",
    "tensor.batch_norm",
    "tensor.reshape",
    "tensor.transpose",
    "tensor.concat",
    "tensor.batch_norm_backward",
    "tensor.derive",
];

// ---------------------------------------------------------------------------
// Op Registry
// ---------------------------------------------------------------------------

/// Build the [`OpRegistry`] for tensor, registering the Rust handls under
/// their op names (no runtime checkers; axioms trusted).
pub fn tensor_op_registry() -> OpRegistry {
    let mut registry = OpRegistry::new();
    registry.register("tensor.zeros", Arc::new(Zeros));
    registry.register("tensor.ones", Arc::new(Ones));
    registry.register("tensor.from_list", Arc::new(FromList));
    registry.register("tensor.to_list", Arc::new(ToList));
    registry.register("tensor.shape", Arc::new(Shape));
    registry.register("tensor.add", Arc::new(Add));
    registry.register("tensor.add_backward", Arc::new(AddBackward));
    registry.register("tensor.mul", Arc::new(Mul));
    registry.register("tensor.mul_backward", Arc::new(MulBackward));
    registry.register("tensor.sub", Arc::new(Sub));
    registry.register("tensor.sub_backward", Arc::new(SubBackward));
    registry.register("tensor.div", Arc::new(Div));
    registry.register("tensor.div_backward", Arc::new(DivBackward));
    registry.register("tensor.matmul", Arc::new(Matmul));
    registry.register("tensor.matmul_backward", Arc::new(MatmulBackward));
    registry.register("tensor.conv2d", Arc::new(Conv2d));
    registry.register("tensor.conv2d_backward", Arc::new(Conv2dBackward));
    registry.register("tensor.max_pool", Arc::new(MaxPool));
    registry.register("tensor.batch_norm", Arc::new(BatchNorm));
    registry.register("tensor.reshape", Arc::new(Reshape));
    registry.register("tensor.transpose", Arc::new(Transpose));
    registry.register("tensor.concat", Arc::new(Concat));
    registry.register("tensor.batch_norm_backward", Arc::new(BatchNormBackward));
    registry.register("tensor.derive", Arc::new(Derive));
    registry
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod testutil {
    use handl_platform::{EvalContext, EvalError, Function};
    use serde_json::{Value, json};

    /// Helper: call a [`Function`] with a fresh [`EvalContext`].
    pub(crate) fn call_fn(f: &dyn Function, args: &[Value]) -> Result<Value, EvalError> {
        let mut ctx = EvalContext::new();
        f.call(args, &mut ctx)
    }

    /// The canonical `__tensor__` wire value for tests.
    pub(crate) fn tensor(shape: &[usize], data: &[f64]) -> Value {
        json!({
            "__tensor__": {
                "shape": shape,
                "data": data,
                "dtype": "f64",
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Tests — the registry, the index.hnl drift guard, and the module seam
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TENSOR_HNL;
    use crate::ops::testutil::tensor;
    use handl_platform::{EvalContext, Module};

    /// Drift guard: `src/index.hnl` is the interface source of truth — its
    /// `cap` / `op` / `handle` decls and the `export tensor` record must
    /// all name exactly [`TENSOR_OPS`] (the decls in declaration order,
    /// the record as a set), and every `DERIV_PAIRS` endpoint must be a
    /// declared op.
    #[test]
    fn index_hnl_interface_matches_tensor_ops() {
        let decls = |prefix: &str| -> Vec<&str> {
            TENSOR_HNL
                .lines()
                .filter_map(|l| l.trim_start().strip_prefix(prefix))
                .filter_map(|r| r.split('"').next())
                .collect()
        };
        assert_eq!(decls("cap \""), TENSOR_OPS, "cap decls vs TENSOR_OPS");
        assert_eq!(decls("op \""), TENSOR_OPS, "op decls vs TENSOR_OPS");
        assert_eq!(
            decls("handle \""),
            TENSOR_OPS,
            "handle decls vs TENSOR_OPS"
        );

        // `export tensor := { name: "tensor.<op>", ... }` — the record's
        // string values must be exactly the op set.
        let record: Vec<&str> = TENSOR_HNL
            .lines()
            .skip_while(|l| !l.contains("export tensor"))
            .skip(1)
            .filter_map(|l| l.split(": \"").nth(1))
            .filter_map(|r| r.split('"').next())
            .collect();
        assert_eq!(record.len(), TENSOR_OPS.len());
        for name in &record {
            assert!(
                TENSOR_OPS.contains(name),
                "export record names undeclared {name}"
            );
        }

        for (forward, backward) in DERIV_PAIRS {
            assert!(TENSOR_OPS.contains(forward), "pair forward {forward}");
            assert!(TENSOR_OPS.contains(backward), "pair adjoint {backward}");
        }
    }

    #[test]
    fn deriv_pairs_are_all_registered() {
        let registry = tensor_op_registry();
        for (forward, backward) in DERIV_PAIRS {
            assert!(registry.contains(forward), "unregistered forward {forward}");
            assert!(
                registry.contains(backward),
                "unregistered adjoint {backward}"
            );
        }
    }

    #[test]
    fn registry_registers_every_declared_op() {
        let registry = tensor_op_registry();
        assert_eq!(registry.len(), TENSOR_OPS.len());
        for op in TENSOR_OPS {
            assert!(registry.contains(op), "missing {op}");
            assert!(registry.get(op).is_some());
        }
    }

    /// The composition seam a runtime uses: attach the registry's impls to
    /// a `HandlModule` via `with_ops`, then dispatch by local name — this is
    /// what `call_host("tensor.matmul")` reaches through the module chain.
    #[test]
    fn handl_module_with_ops_dispatches() {
        let module = handl_plugin_lib_handl::HandlModule::new("tensor", crate::TENSOR_HNL, &[])
            .with_ops(TENSOR_OPS, Arc::new(tensor_op_registry()));
        let f = module
            .function("matmul")
            .expect("matmul should be an op function on the module");
        let mut ctx = EvalContext::new();
        let v = f
            .call(
                &[tensor(&[1, 2], &[3.0, 4.0]), tensor(&[2, 1], &[5.0, 6.0])],
                &mut ctx,
            )
            .unwrap();
        assert_eq!(v, tensor(&[1, 1], &[39.0]));
    }
}
