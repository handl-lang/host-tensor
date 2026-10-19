//! The `__tensor__` wire codec and the boundary ops — the impl mirrored by
//! lib-tensor's `codec.hnl`.
//!
//! [`Tensor`] is the decoded representation (dense, row-major, `f64`);
//! [`decode_tensor`] / [`encode_tensor`] cross the `serde_json::Value`
//! `Function` boundary in the `{"__tensor__": {"shape": ..., "data": ...,
//! "dtype": "f64"}}` tagged-object style (ADR-0083 §2.1). The `require_*`
//! / `opt_*` helpers decode positional arguments for every op — including
//! the optional trailing parameters of `conv2d` / `max_pool` /
//! `batch_norm`, where a `null` or absent argument selects the default.
//!
//! The boundary ops live here too: [`Zeros`], [`Ones`], [`FromList`]
//! construct wire tensors; [`ToList`], [`Shape`] project them.

use handl_platform::{EvalContext, EvalError, Function};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// The tensor codec
// ---------------------------------------------------------------------------

/// A decoded tensor: dense, row-major, `f64`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Tensor {
    /// Extent per axis; `shape.len()` is the rank.
    pub(crate) shape: Vec<usize>,
    /// Flat row-major payload; `data.len() == product(shape)`.
    pub(crate) data: Vec<f64>,
}

pub(crate) fn type_mismatch(expected: &str, got: &str) -> EvalError {
    EvalError::TypeMismatch(format!("expected {expected}, got {got}"))
}

/// Decode a tensor argument. Accepts `{"__tensor__": {"shape": [...],
/// "data": [...], "dtype"?: "f64"}}`; anything else — a missing tag, a
/// non-array shape/data, a negative or non-integer extent, a data length
/// that is not `product(shape)`, or a `dtype` other than `"f64"` — is a
/// `TypeMismatch` (malformed wire value) rather than a silent coercion.
pub(crate) fn decode_tensor(arg: &Value) -> Result<Tensor, EvalError> {
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
pub(crate) fn require_tensor(args: &[Value], pos: usize) -> Result<Tensor, EvalError> {
    let arg = args
        .get(pos)
        .ok_or_else(|| type_mismatch("tensor argument", "missing"))?;
    decode_tensor(arg)
}

/// Encode a tensor as the canonical `{"__tensor__": {...}}` wire value.
///
/// Non-finite `f64` elements (`inf`, `-inf`, `NaN` — e.g. a `div` by
/// zero or an `add`/`mul` overflow) have no `serde_json` representation:
/// `Number::from_f64` rejects them and they serialize as `null` in
/// `data`, which [`decode_tensor`] rejects on the way back in. The wire
/// is therefore closed over *finite* `f64` only — the kernel
/// `TaggedValue::Tensor` carries them inside an engine, but any
/// `serde_json` crossing still degrades them to `null`.
pub(crate) fn encode_tensor(t: Tensor) -> Value {
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
/// `lib-char-native`).
pub(crate) fn require_extent(arg: &Value) -> Result<usize, EvalError> {
    let n = arg.as_u64().or_else(|| {
        arg.as_f64()
            .filter(|n| *n >= 0.0 && n.fract() == 0.0 && n.is_finite())
            .map(|n| n as u64)
    });
    n.and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| type_mismatch("non-negative integer extent", "other"))
}

/// Decode a `shape` argument (a list of non-negative integer extents).
pub(crate) fn require_shape(args: &[Value], pos: usize) -> Result<Vec<usize>, EvalError> {
    let arg = args
        .get(pos)
        .ok_or_else(|| type_mismatch("shape argument", "missing"))?;
    let arr = arg
        .as_array()
        .ok_or_else(|| type_mismatch("shape: array of extents", "non-array"))?;
    arr.iter().map(require_extent).collect()
}

// ---------------------------------------------------------------------------
// Optional-argument decoding (conv2d / batch_norm)
//
// Optional trailing parameters are positional; a `null` (or an absent
// argument) selects the default, so callers can set a later parameter
// without committing to the earlier ones — `conv2d(x, w, null, [2, 2])`
// is "no bias, stride 2×2, no padding".
// ---------------------------------------------------------------------------

/// Check the argument count is in `min..=max` (variadic ops cannot use
/// `require_arg_count`).
pub(crate) fn require_arg_range(args: &[Value], min: usize, max: usize, op: &str) -> Result<(), EvalError> {
    if args.len() < min || args.len() > max {
        return Err(EvalError::TypeMismatch(format!(
            "{op}: expected {min}..={max} args, got {}",
            args.len()
        )));
    }
    Ok(())
}

/// Whether the optional argument at `pos` is present and non-null.
pub(crate) fn has_opt_arg(args: &[Value], pos: usize) -> bool {
    args.get(pos).is_some_and(|a| !a.is_null())
}

/// Decode an optional tensor argument (`null`/absent → `None`).
pub(crate) fn opt_tensor(args: &[Value], pos: usize) -> Result<Option<Tensor>, EvalError> {
    match args.get(pos) {
        Some(arg) if !arg.is_null() => Ok(Some(decode_tensor(arg)?)),
        _ => Ok(None),
    }
}

/// Decode an optional spatial parameter (`stride` / `padding`): a single
/// non-negative integer `n` means `[n, n]`; a two-element list `[h, w]`
/// gives the per-axis extents; `null`/absent means `default`.
pub(crate) fn opt_extent_pair(
    args: &[Value],
    pos: usize,
    default: [usize; 2],
    name: &str,
) -> Result<[usize; 2], EvalError> {
    let Some(arg) = args.get(pos) else {
        return Ok(default);
    };
    if arg.is_null() {
        return Ok(default);
    }
    if let Some(arr) = arg.as_array() {
        if arr.len() != 2 {
            return Err(type_mismatch(
                &format!("{name}: int or [h, w] pair"),
                &format!("array of len {}", arr.len()),
            ));
        }
        return Ok([require_extent(&arr[0])?, require_extent(&arr[1])?]);
    }
    let n = require_extent(arg)?;
    Ok([n, n])
}

/// Decode an optional strictly-positive integer (`groups`): `null`/absent
/// means `default`; zero or negative is an error (a non-positive group
/// count has no meaning, unlike a zero stride/padding which is simply
/// rejected downstream or degenerate).
pub(crate) fn opt_positive_int(
    args: &[Value],
    pos: usize,
    default: usize,
    name: &str,
) -> Result<usize, EvalError> {
    if !has_opt_arg(args, pos) {
        return Ok(default);
    }
    let n = require_extent(&args[pos])?;
    if n == 0 {
        return Err(EvalError::Other(format!(
            "{name}: must be a positive integer, got 0"
        )));
    }
    Ok(n)
}

/// Decode an optional non-negative finite float (`eps`): `null`/absent
/// means `default`.
pub(crate) fn opt_nonneg_float(
    args: &[Value],
    pos: usize,
    default: f64,
    name: &str,
) -> Result<f64, EvalError> {
    if !has_opt_arg(args, pos) {
        return Ok(default);
    }
    let x = args[pos]
        .as_f64()
        .ok_or_else(|| type_mismatch(&format!("{name}: number"), "non-number"))?;
    if !x.is_finite() || x < 0.0 {
        return Err(EvalError::Other(format!(
            "{name}: expected a non-negative finite number, got {x}"
        )));
    }
    Ok(x)
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::testutil::{call_fn, tensor};
    use serde_json::json;

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
}
