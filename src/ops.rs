//! Rust `Function` impls for the `tensor` module — the handls (handlers)
//! for the `op`/`handle` declarations in `src/index.hnl`.
//!
//! All 7 functions are FFI atoms: a tensor has no literal syntax and no
//! eval-vocabulary constructor, so construction, projection, and algebra
//! cannot be expressed as pure handl arrows. The impls here are the
//! host-side half runtimes compose: a consumer registers them in an
//! [`OpRegistry`](handl_plugin_lib_handl::OpRegistry) (via
//! [`tensor_op_registry`]) and wires them onto a `HandlModule` with
//! `with_ops(TENSOR_OPS, …)`, after which `.hnl` `tensor.*` calls dispatch
//! through the module chain into these impls.
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
//! object; the tag keeps it unambiguous next to user records.

use std::sync::Arc;

use handl_platform::{EvalContext, EvalError, Function};
use handl_plugin_lib_handl::OpRegistry;
use serde_json::{Value, json};

/// The op names declared in `src/index.hnl`, in declaration order.
pub const TENSOR_OPS: &[&str] = &[
    "tensor.zeros",
    "tensor.ones",
    "tensor.from_list",
    "tensor.to_list",
    "tensor.shape",
    "tensor.add",
    "tensor.matmul",
];

// ---------------------------------------------------------------------------
// The tensor codec
// ---------------------------------------------------------------------------

/// A decoded tensor: dense, row-major, `f64`.
#[derive(Debug, Clone, PartialEq)]
struct Tensor {
    /// Extent per axis; `shape.len()` is the rank.
    shape: Vec<usize>,
    /// Flat row-major payload; `data.len() == product(shape)`.
    data: Vec<f64>,
}

fn type_mismatch(expected: &str, got: &str) -> EvalError {
    EvalError::TypeMismatch(format!("expected {expected}, got {got}"))
}

/// Decode a tensor argument. Accepts `{"__tensor__": {"shape": [...],
/// "data": [...], "dtype"?: "f64"}}`; anything else — a missing tag, a
/// non-array shape/data, a negative or non-integer extent, a data length
/// that is not `product(shape)`, or a `dtype` other than `"f64"` — is a
/// `TypeMismatch` (malformed wire value) rather than a silent coercion.
fn decode_tensor(arg: &Value) -> Result<Tensor, EvalError> {
    let inner = arg
        .get("__tensor__")
        .and_then(Value::as_object)
        .ok_or_else(|| type_mismatch("Tensor", "non-Tensor"))?;
    if let Some(dtype) = inner.get("dtype") {
        if dtype.as_str() != Some("f64") {
            return Err(type_mismatch("dtype \"f64\"", "other dtype"));
        }
    }
    let shape_val = inner
        .get("shape")
        .ok_or_else(|| type_mismatch("tensor.shape field", "missing"))?;
    let shape_arr = shape_val
        .as_array()
        .ok_or_else(|| type_mismatch("shape: array", "non-array"))?;
    let mut shape = Vec::with_capacity(shape_arr.len());
    for d in shape_arr {
        let n = d
            .as_u64()
            .ok_or_else(|| type_mismatch("shape: non-negative integers", "non-integer"))?;
        shape.push(
            usize::try_from(n).map_err(|_| type_mismatch("shape: usize extents", "overflow"))?,
        );
    }
    let data_val = inner
        .get("data")
        .ok_or_else(|| type_mismatch("tensor.data field", "missing"))?;
    let data_arr = data_val
        .as_array()
        .ok_or_else(|| type_mismatch("data: array", "non-array"))?;
    let mut data = Vec::with_capacity(data_arr.len());
    for x in data_arr {
        data.push(
            x.as_f64()
                .ok_or_else(|| type_mismatch("data: numbers", "non-number"))?,
        );
    }
    let expected = shape.iter().product::<usize>();
    if data.len() != expected {
        return Err(EvalError::Other(format!(
            "tensor: data length {} does not match product of shape {shape:?} ({expected})",
            data.len()
        )));
    }
    Ok(Tensor { shape, data })
}

/// Decode the tensor at argument position `pos`.
fn require_tensor(args: &[Value], pos: usize) -> Result<Tensor, EvalError> {
    let arg = args
        .get(pos)
        .ok_or_else(|| type_mismatch("tensor argument", "missing"))?;
    decode_tensor(arg)
}

/// Encode a tensor as the canonical `{"__tensor__": {...}}` wire value.
fn encode_tensor(t: Tensor) -> Value {
    json!({
        "__tensor__": {
            "shape": t.shape,
            "data": t.data,
            "dtype": "f64",
        }
    })
}

/// Decode a non-negative integer extent (integral floats count — upstream
/// ops produce `f64` values, matching the `require_int` convention in
/// `plugin-lib-char-native` / `lib-bytes`).
fn require_extent(arg: &Value) -> Result<usize, EvalError> {
    let n = arg.as_u64().or_else(|| {
        arg.as_f64()
            .filter(|n| *n >= 0.0 && n.fract() == 0.0 && n.is_finite())
            .map(|n| n as u64)
    });
    n.and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| type_mismatch("non-negative integer extent", "other"))
}

/// Decode a `shape` argument (a list of non-negative integer extents).
fn require_shape(args: &[Value], pos: usize) -> Result<Vec<usize>, EvalError> {
    let arg = args
        .get(pos)
        .ok_or_else(|| type_mismatch("shape argument", "missing"))?;
    let arr = arg
        .as_array()
        .ok_or_else(|| type_mismatch("shape: array of extents", "non-array"))?;
    arr.iter().map(require_extent).collect()
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// The Rust handl for `op "tensor.zeros" : (List Int) -> Tensor`.
#[derive(Debug, Clone, Copy)]
pub struct Zeros;
impl Function for Zeros {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 1)?;
        let shape = require_shape(args, 0)?;
        let len = shape.iter().product();
        Ok(encode_tensor(Tensor {
            shape,
            data: vec![0.0; len],
        }))
    }
}

/// The Rust handl for `op "tensor.ones" : (List Int) -> Tensor`.
#[derive(Debug, Clone, Copy)]
pub struct Ones;
impl Function for Ones {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 1)?;
        let shape = require_shape(args, 0)?;
        let len = shape.iter().product();
        Ok(encode_tensor(Tensor {
            shape,
            data: vec![1.0; len],
        }))
    }
}

/// The Rust handl for `op "tensor.from_list" : Array -> Tensor`.
///
/// Infers the shape from the nesting of number arrays and requires the
/// input to be rectangular: at every level all siblings must have the same
/// inferred sub-shape, so `[[1, 2], [3]]` and `[[1], 2]` are errors. A
/// scalar argument is not a list — `TypeMismatch`. `[]` is the rank-1
/// empty tensor (shape `[0]`); `[[], []]` is shape `[2, 0]`.
#[derive(Debug, Clone, Copy)]
pub struct FromList;
impl Function for FromList {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 1)?;
        let arg = &args[0];
        if !arg.is_array() {
            return Err(type_mismatch("nested number arrays", "non-array"));
        }
        let shape = infer_shape(arg)?;
        let mut data = Vec::with_capacity(shape.iter().product());
        flatten(arg, &mut data);
        Ok(encode_tensor(Tensor { shape, data }))
    }
}

/// Infer `xs`' shape: `[]` prepends its length to the (uniform) sub-shape
/// of its elements; a number contributes `[]`; anything else errors.
fn infer_shape(xs: &Value) -> Result<Vec<usize>, EvalError> {
    match xs {
        Value::Array(arr) => {
            let mut shape = vec![arr.len()];
            if let Some(first) = arr.first() {
                let sub = infer_shape(first)?;
                for el in &arr[1..] {
                    let el_shape = infer_shape(el)?;
                    if el_shape != sub {
                        return Err(EvalError::Other(format!(
                            "tensor.from_list: ragged input — element shapes {sub:?} and {el_shape:?} differ"
                        )));
                    }
                }
                shape.extend(sub);
            }
            Ok(shape)
        }
        Value::Number(_) => Ok(Vec::new()),
        _ => Err(type_mismatch("nested number arrays", "non-number leaf")),
    }
}

/// Append `xs`' numbers to `out` in row-major order. Only reachable after
/// [`infer_shape`] accepted `xs`, so every leaf is a number.
fn flatten(xs: &Value, out: &mut Vec<f64>) {
    match xs {
        Value::Array(arr) => {
            for el in arr {
                flatten(el, out);
            }
        }
        Value::Number(n) => out.push(n.as_f64().unwrap_or(0.0)),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

/// The Rust handl for `op "tensor.to_list" : Tensor -> Array`.
///
/// The inverse of `from_list`: re-nests the flat row-major payload into
/// `shape`'s structure.
#[derive(Debug, Clone, Copy)]
pub struct ToList;
impl Function for ToList {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 1)?;
        let t = require_tensor(args, 0)?;
        let mut it = t.data.iter().copied();
        Ok(nest(&t.shape, &mut it))
    }
}

/// Re-nest `data` (row-major) under `shape`.
fn nest(shape: &[usize], data: &mut impl Iterator<Item = f64>) -> Value {
    match shape.split_first() {
        None => json!(data.next().unwrap_or(0.0)),
        Some((&d, rest)) => Value::Array((0..d).map(|_| nest(rest, data)).collect()),
    }
}

/// The Rust handl for `op "tensor.shape" : Tensor -> (List Int)`.
#[derive(Debug, Clone, Copy)]
pub struct Shape;
impl Function for Shape {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 1)?;
        let t = require_tensor(args, 0)?;
        Ok(json!(t.shape))
    }
}

// ---------------------------------------------------------------------------
// Algebra
// ---------------------------------------------------------------------------

/// The Rust handl for `op "tensor.add" : (Tensor * Tensor) -> Tensor`.
///
/// Elementwise addition; the shapes must be identical (broadcasting is
/// follow-up — a mismatch is an error, not a coercion).
#[derive(Debug, Clone, Copy)]
pub struct Add;
impl Function for Add {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 2)?;
        let a = require_tensor(args, 0)?;
        let b = require_tensor(args, 1)?;
        if a.shape != b.shape {
            return Err(EvalError::Other(format!(
                "tensor.add: shape mismatch {:?} vs {:?} (broadcasting is not supported yet)",
                a.shape, b.shape
            )));
        }
        let data = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| x + y)
            .collect();
        Ok(encode_tensor(Tensor {
            shape: a.shape,
            data,
        }))
    }
}

/// The Rust handl for `op "tensor.matmul" : (Tensor * Tensor) -> Tensor`.
///
/// Rank-2 × rank-2 dense product `(m×k)·(k×n) -> m×n`, naive triple loop —
/// the point of this component is the composition seam, not BLAS. Other
/// ranks are an error (batched matmul is follow-up).
#[derive(Debug, Clone, Copy)]
pub struct Matmul;
impl Function for Matmul {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 2)?;
        let a = require_tensor(args, 0)?;
        let b = require_tensor(args, 1)?;
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(EvalError::Other(format!(
                "tensor.matmul: expected two rank-2 tensors, got ranks {} and {}",
                a.shape.len(),
                b.shape.len()
            )));
        }
        let (m, k) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);
        if k != k2 {
            return Err(EvalError::Other(format!(
                "tensor.matmul: inner dimension mismatch {k} vs {k2} ({m}x{k})·({k2}x{n})"
            )));
        }
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for l in 0..k {
                let a_il = a.data[i * k + l];
                if a_il == 0.0 {
                    continue;
                }
                for j in 0..n {
                    data[i * n + j] += a_il * b.data[l * n + j];
                }
            }
        }
        Ok(encode_tensor(Tensor {
            shape: vec![m, n],
            data,
        }))
    }
}

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
    registry.register("tensor.matmul", Arc::new(Matmul));
    registry
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use handl_platform::Module;
    use serde_json::json;

    /// Helper: call a [`Function`] with a fresh [`EvalContext`].
    fn call_fn(f: &dyn Function, args: &[Value]) -> Result<Value, EvalError> {
        let mut ctx = EvalContext::new();
        f.call(args, &mut ctx)
    }

    fn tensor(shape: &[usize], data: &[f64]) -> Value {
        json!({
            "__tensor__": {
                "shape": shape,
                "data": data,
                "dtype": "f64",
            }
        })
    }

    // ── codec ─────────────────────────────────────────────────────────

    #[test]
    fn decode_round_trip() {
        let v = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = decode_tensor(&v).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(encode_tensor(t.clone()), v);
    }

    #[test]
    fn decode_dtype_optional_on_input() {
        let v = json!({"__tensor__": {"shape": [1], "data": [7]}});
        let t = decode_tensor(&v).unwrap();
        assert_eq!(t.data, vec![7.0]);
    }

    #[test]
    fn decode_rejects_wrong_dtype() {
        let v = json!({"__tensor__": {"shape": [1], "data": [7], "dtype": "i32"}});
        assert!(decode_tensor(&v).is_err());
    }

    #[test]
    fn decode_rejects_untagged_and_malformed() {
        assert!(decode_tensor(&json!({"shape": [1], "data": [1]})).is_err());
        assert!(decode_tensor(&json!({"__tensor__": {"shape": [2], "data": [1]}})).is_err());
        assert!(decode_tensor(&json!({"__tensor__": {"shape": [-1], "data": []}})).is_err());
        assert!(decode_tensor(&json!(42)).is_err());
    }

    // ── construction ──────────────────────────────────────────────────

    #[test]
    fn zeros_fills() {
        let v = call_fn(&Zeros, &[json!([2, 2])]).unwrap();
        assert_eq!(v, tensor(&[2, 2], &[0.0; 4]));
    }

    #[test]
    fn ones_fills() {
        let v = call_fn(&Ones, &[json!([3])]).unwrap();
        assert_eq!(v, tensor(&[3], &[1.0; 3]));
    }

    #[test]
    fn zeros_scalar_shape_is_rank_zero() {
        let v = call_fn(&Zeros, &[json!([])]).unwrap();
        assert_eq!(v, tensor(&[], &[0.0]));
    }

    #[test]
    fn zeros_rejects_negative_extent() {
        assert!(call_fn(&Zeros, &[json!([2, -1])]).is_err());
    }

    #[test]
    fn from_list_infers_shape() {
        let v = call_fn(&FromList, &[json!([[1, 2, 3], [4, 5, 6]])]).unwrap();
        assert_eq!(v, tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    }

    #[test]
    fn from_list_rank_one() {
        let v = call_fn(&FromList, &[json!([1, 2])]).unwrap();
        assert_eq!(v, tensor(&[2], &[1.0, 2.0]));
    }

    #[test]
    fn from_list_rejects_ragged() {
        assert!(call_fn(&FromList, &[json!([[1, 2], [3]])]).is_err());
        assert!(call_fn(&FromList, &[json!([[1], 2])]).is_err());
        assert!(call_fn(&FromList, &[json!([1, "x"])]).is_err());
        assert!(call_fn(&FromList, &[json!(5)]).is_err());
    }

    // ── projection ────────────────────────────────────────────────────

    #[test]
    fn to_list_nests() {
        let v = call_fn(&ToList, &[tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]).unwrap();
        assert_eq!(v, json!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    }

    #[test]
    fn from_list_to_list_round_trip() {
        let input = json!([[1, 2], [3, 4], [5, 6]]);
        let t = call_fn(&FromList, &[input.clone()]).unwrap();
        let back = call_fn(&ToList, &[t]).unwrap();
        // `to_list` emits `f64`s — semantically equal to the integer input
        // (handl `=` is Int/Float cross-comparable) though the JSON
        // `Number` representation differs.
        assert_eq!(back, json!([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]));
    }

    #[test]
    fn shape_projects() {
        let v = call_fn(&Shape, &[tensor(&[2, 3], &[0.0; 6])]).unwrap();
        assert_eq!(v, json!([2, 3]));
    }

    // ── algebra ───────────────────────────────────────────────────────

    #[test]
    fn add_elementwise() {
        let v = call_fn(
            &Add,
            &[
                tensor(&[2], &[1.0, 2.0]),
                tensor(&[2], &[10.0, 20.0]),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[2], &[11.0, 22.0]));
    }

    #[test]
    fn add_rejects_shape_mismatch() {
        assert!(
            call_fn(
                &Add,
                &[tensor(&[2], &[1.0, 2.0]), tensor(&[3], &[1.0, 2.0, 3.0])]
            )
            .is_err()
        );
    }

    #[test]
    fn matmul_2x2() {
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        let v = call_fn(
            &Matmul,
            &[
                tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[2, 2], &[19.0, 22.0, 43.0, 50.0]));
    }

    #[test]
    fn matmul_rectangular() {
        // [[1,2,3],[4,5,6]] (2x3) · [[7,8],[9,10],[11,12]] (3x2) = [[58,64],[139,154]]
        let v = call_fn(
            &Matmul,
            &[
                tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                tensor(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[2, 2], &[58.0, 64.0, 139.0, 154.0]));
    }

    #[test]
    fn matmul_rejects_inner_dim_mismatch_and_bad_rank() {
        assert!(
            call_fn(
                &Matmul,
                &[tensor(&[2, 3], &[0.0; 6]), tensor(&[2, 3], &[0.0; 6])]
            )
            .is_err()
        );
        assert!(
            call_fn(
                &Matmul,
                &[tensor(&[4], &[0.0; 4]), tensor(&[4], &[0.0; 4])]
            )
            .is_err()
        );
    }

    // ── registry ──────────────────────────────────────────────────────

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
                &[
                    tensor(&[1, 2], &[3.0, 4.0]),
                    tensor(&[2, 1], &[5.0, 6.0]),
                ],
                &mut ctx,
            )
            .unwrap();
        assert_eq!(v, tensor(&[1, 1], &[39.0]));
    }
}
