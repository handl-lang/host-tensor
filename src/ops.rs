//! Rust `Function` impls for the `tensor` module — the handls (handlers)
//! for the `op`/`handle` declarations in `src/index.hnl`.
//!
//! All 13 functions are FFI atoms: a tensor has no literal syntax and no
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
    "tensor.mul",
    "tensor.sub",
    "tensor.div",
    "tensor.matmul",
    "tensor.conv2d",
    "tensor.max_pool",
    "tensor.batch_norm",
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
///
/// Non-finite `f64` elements (`inf`, `-inf`, `NaN` — e.g. a `div` by
/// zero or an `add`/`mul` overflow) have no `serde_json` representation:
/// `Number::from_f64` rejects them and they serialize as `null` in
/// `data`, which [`decode_tensor`] rejects on the way back in. The wire
/// is therefore closed over *finite* `f64` only — a `TaggedValue::Tensor`
/// kernel payload would carry them faithfully (README follow-up).
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
// Optional-argument decoding (conv2d / batch_norm)
//
// Optional trailing parameters are positional; a `null` (or an absent
// argument) selects the default, so callers can set a later parameter
// without committing to the earlier ones — `conv2d(x, w, null, [2, 2])`
// is "no bias, stride 2×2, no padding".
// ---------------------------------------------------------------------------

/// Check the argument count is in `min..=max` (variadic ops cannot use
/// `require_arg_count`).
fn require_arg_range(args: &[Value], min: usize, max: usize, op: &str) -> Result<(), EvalError> {
    if args.len() < min || args.len() > max {
        return Err(EvalError::TypeMismatch(format!(
            "{op}: expected {min}..={max} args, got {}",
            args.len()
        )));
    }
    Ok(())
}

/// Whether the optional argument at `pos` is present and non-null.
fn has_opt_arg(args: &[Value], pos: usize) -> bool {
    args.get(pos).is_some_and(|a| !a.is_null())
}

/// Decode an optional tensor argument (`null`/absent → `None`).
fn opt_tensor(args: &[Value], pos: usize) -> Result<Option<Tensor>, EvalError> {
    match args.get(pos) {
        Some(arg) if !arg.is_null() => Ok(Some(decode_tensor(arg)?)),
        _ => Ok(None),
    }
}

/// Decode an optional spatial parameter (`stride` / `padding`): a single
/// non-negative integer `n` means `[n, n]`; a two-element list `[h, w]`
/// gives the per-axis extents; `null`/absent means `default`.
fn opt_extent_pair(
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
fn opt_positive_int(
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
fn opt_nonneg_float(
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
// Algebra
// ---------------------------------------------------------------------------

/// The broadcast of two shapes (numpy rules): align trailing dims, and
/// each dim pair must be equal or contain a `1` — which stretches to
/// the other extent. A missing leading dim counts as `1`, so a rank-1
/// `[k]` vector broadcasts across the last axis of an `[m, k]` matrix.
fn broadcast_shape(a: &[usize], b: &[usize], op: &str) -> Result<Vec<usize>, EvalError> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for (i, o) in out.iter_mut().enumerate() {
        let da = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let db = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
        *o = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => {
                return Err(EvalError::Other(format!(
                    "{op}: shapes {a:?} and {b:?} are not broadcast-compatible \
                     (differ at axis {i}: {da} vs {db})"
                )));
            }
        };
    }
    Ok(out)
}

/// Row-major strides for `shape` indexed under the broadcast `out_shape`:
/// a dim of `1` (or a missing leading dim) contributes stride 0, so its
/// single index is reused across the stretch. `out_shape` must be a
/// broadcast of `shape` — the caller checks with [`broadcast_shape`].
fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    // Contiguous row-major strides within `shape` itself.
    let mut inner = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        inner[i] = acc;
        acc *= shape[i];
    }
    let off = out_shape.len() - shape.len();
    let mut strides = vec![0usize; out_shape.len()];
    for (i, &s) in inner.iter().enumerate() {
        strides[off + i] = if shape[i] == 1 { 0 } else { s };
    }
    strides
}

/// The driver behind `tensor.add` / `tensor.mul` / `tensor.sub` /
/// `tensor.div`: elementwise `f` over the broadcast of `a` and `b`.
fn broadcast_elementwise(
    a: &Tensor,
    b: &Tensor,
    op: &str,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Tensor, EvalError> {
    let out_shape = broadcast_shape(&a.shape, &b.shape, op)?;
    let sa = broadcast_strides(&a.shape, &out_shape);
    let sb = broadcast_strides(&b.shape, &out_shape);
    let rank = out_shape.len();
    let mut data = Vec::with_capacity(out_shape.iter().product());
    // Odometer over the output index space; strides map it back into
    // each input's flat row-major data.
    let mut idx = vec![0usize; rank];
    for _ in 0..out_shape.iter().product::<usize>() {
        let (mut oa, mut ob) = (0usize, 0usize);
        for d in 0..rank {
            oa += idx[d] * sa[d];
            ob += idx[d] * sb[d];
        }
        data.push(f(a.data[oa], b.data[ob]));
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(Tensor {
        shape: out_shape,
        data,
    })
}

/// The Rust handl for `op "tensor.add" : (Tensor * Tensor) -> Tensor`.
///
/// Elementwise addition with numpy-style broadcasting: trailing dims
/// align, a dim of `1` stretches, and a missing leading dim counts as
/// `1` — so `add(ones([2, 3]), ones([3]))` is `[2, 3]`. An incompatible
/// pair (dims that differ and neither is `1`) is an error.
#[derive(Debug, Clone, Copy)]
pub struct Add;
impl Function for Add {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 2)?;
        let a = require_tensor(args, 0)?;
        let b = require_tensor(args, 1)?;
        Ok(encode_tensor(broadcast_elementwise(
            &a,
            &b,
            "tensor.add",
            |x, y| x + y,
        )?))
    }
}

/// The Rust handl for `op "tensor.mul" : (Tensor * Tensor) -> Tensor`.
///
/// Elementwise multiplication, broadcasting like [`Add`].
#[derive(Debug, Clone, Copy)]
pub struct Mul;
impl Function for Mul {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 2)?;
        let a = require_tensor(args, 0)?;
        let b = require_tensor(args, 1)?;
        Ok(encode_tensor(broadcast_elementwise(
            &a,
            &b,
            "tensor.mul",
            |x, y| x * y,
        )?))
    }
}

/// The Rust handl for `op "tensor.sub" : (Tensor * Tensor) -> Tensor`.
///
/// Elementwise subtraction `a − b`, broadcasting like [`Add`].
#[derive(Debug, Clone, Copy)]
pub struct Sub;
impl Function for Sub {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 2)?;
        let a = require_tensor(args, 0)?;
        let b = require_tensor(args, 1)?;
        Ok(encode_tensor(broadcast_elementwise(
            &a,
            &b,
            "tensor.sub",
            |x, y| x - y,
        )?))
    }
}

/// The Rust handl for `op "tensor.div" : (Tensor * Tensor) -> Tensor`.
///
/// Elementwise division `a / b`, broadcasting like [`Add`]. Division by
/// zero produces `f64` `inf`/`NaN` — the same semantics `.hnl`
/// arithmetic gives the scalar `/`, not a host-side error. Note the
/// `serde_json` wire cannot carry non-finite `f64`s: they serialize as
/// `null` in `data`, which [`decode_tensor`] rejects on the way back in
/// (see the codec doc comment).
#[derive(Debug, Clone, Copy)]
pub struct Div;
impl Function for Div {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 2)?;
        let a = require_tensor(args, 0)?;
        let b = require_tensor(args, 1)?;
        Ok(encode_tensor(broadcast_elementwise(
            &a,
            &b,
            "tensor.div",
            |x, y| x / y,
        )?))
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

/// The Rust handl for `op "tensor.conv2d" : Value... -> Tensor`.
///
/// NCHW cross-correlation (the "convolution" of the deep-learning
/// libraries — no kernel flip): `conv2d(input, weight, bias?, stride?,
/// padding?, groups?)` with
///
/// - `input` — rank-4 `[N, C_in, H, W]`;
/// - `weight` — rank-4 `[C_out, C_in/groups, kH, kW]`;
/// - `bias` — `null`/absent or a rank-1 `[C_out]` tensor added per output
///   channel;
/// - `stride` — `null`/absent (→ `[1, 1]`), a positive integer `s`
///   (→ `[s, s]`), or `[sH, sW]`;
/// - `padding` — `null`/absent (→ `[0, 0]`), a non-negative integer `p`
///   (→ `[p, p]`), or `[pH, pW]`; implicit zero-padding;
/// - `groups` — `null`/absent (→ `1`) or a positive integer. `groups`
///   must divide both `C_in` and `C_out`; `groups = C_in = C_out` is
///   depthwise convolution.
///
/// The output is `[N, C_out, H_out, W_out]` with
/// `H_out = (H + 2·pH − kH)/sH + 1` (likewise `W_out`); a kernel that
/// does not fit the padded input is an error. Naive `f64` loops — the
/// composition seam, not BLAS.
#[derive(Debug, Clone, Copy)]
pub struct Conv2d;
impl Function for Conv2d {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        require_arg_range(args, 2, 6, "tensor.conv2d")?;
        let input = require_tensor(args, 0)?;
        let weight = require_tensor(args, 1)?;
        let bias = opt_tensor(args, 2)?;
        let [s_h, s_w] = opt_extent_pair(args, 3, [1, 1], "tensor.conv2d: stride")?;
        let [p_h, p_w] = opt_extent_pair(args, 4, [0, 0], "tensor.conv2d: padding")?;
        let groups = opt_positive_int(args, 5, 1, "tensor.conv2d: groups")?;
        if s_h == 0 || s_w == 0 {
            return Err(EvalError::Other(format!(
                "tensor.conv2d: stride must be positive, got [{s_h}, {s_w}]"
            )));
        }
        if input.shape.len() != 4 || weight.shape.len() != 4 {
            return Err(EvalError::Other(format!(
                "tensor.conv2d: expected rank-4 input and weight (NCHW × OIHW), got ranks {} and {}",
                input.shape.len(),
                weight.shape.len()
            )));
        }
        let [n, c_in, h, w] = [
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        ];
        let [c_out, c_in_w, k_h, k_w] = [
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        ];
        if c_in % groups != 0 || c_out % groups != 0 {
            return Err(EvalError::Other(format!(
                "tensor.conv2d: groups {groups} must divide both C_in {c_in} and C_out {c_out}"
            )));
        }
        let c_in_g = c_in / groups;
        let c_out_g = c_out / groups;
        if c_in_w != c_in_g {
            return Err(EvalError::Other(format!(
                "tensor.conv2d: weight channels {c_in_w} != C_in/groups {c_in_g} \
                 (input [N, {c_in}, {h}, {w}], weight [{c_out}, {c_in_w}, {k_h}, {k_w}], groups {groups})"
            )));
        }
        if let Some(b) = &bias {
            if b.shape != [c_out] {
                return Err(EvalError::Other(format!(
                    "tensor.conv2d: bias shape {:?} != [{c_out}]",
                    b.shape
                )));
            }
        }
        let Some(h_out_num) = (h + 2 * p_h).checked_sub(k_h) else {
            return Err(EvalError::Other(format!(
                "tensor.conv2d: kernel height {k_h} exceeds padded input height {}",
                h + 2 * p_h
            )));
        };
        let Some(w_out_num) = (w + 2 * p_w).checked_sub(k_w) else {
            return Err(EvalError::Other(format!(
                "tensor.conv2d: kernel width {k_w} exceeds padded input width {}",
                w + 2 * p_w
            )));
        };
        let h_out = h_out_num / s_h + 1;
        let w_out = w_out_num / s_w + 1;
        let mut data = vec![0.0; n * c_out * h_out * w_out];
        for bn in 0..n {
            for g in 0..groups {
                for oc in 0..c_out_g {
                    let co = g * c_out_g + oc;
                    let b_co = bias.as_ref().map_or(0.0, |b| b.data[co]);
                    for oy in 0..h_out {
                        for ox in 0..w_out {
                            let mut acc = b_co;
                            for ic in 0..c_in_g {
                                let ci = g * c_in_g + ic;
                                for ky in 0..k_h {
                                    // Input row = oy*sH + ky − pH; out of
                                    // range is the zero padding.
                                    let iy = oy * s_h + ky;
                                    if iy < p_h {
                                        continue;
                                    }
                                    let iy = iy - p_h;
                                    if iy >= h {
                                        continue;
                                    }
                                    for kx in 0..k_w {
                                        let ix = ox * s_w + kx;
                                        if ix < p_w {
                                            continue;
                                        }
                                        let ix = ix - p_w;
                                        if ix >= w {
                                            continue;
                                        }
                                        acc += input.data[((bn * c_in + ci) * h + iy) * w + ix]
                                            * weight.data
                                                [((co * c_in_g + ic) * k_h + ky) * k_w + kx];
                                    }
                                }
                            }
                            data[((bn * c_out + co) * h_out + oy) * w_out + ox] = acc;
                        }
                    }
                }
            }
        }
        Ok(encode_tensor(Tensor {
            shape: vec![n, c_out, h_out, w_out],
            data,
        }))
    }
}

/// The Rust handl for `op "tensor.max_pool" : Value... -> Tensor`.
///
/// NCHW max pooling: `max_pool(input, kernel, stride?)` with `input`
/// rank-4 `[N, C, H, W]`; `kernel` an int `k` (→ `[k, k]`) or a
/// `[kH, kW]` pair; `stride` `null`/absent (→ `kernel`), an int `s`
/// (→ `[s, s]`), or `[sH, sW]`. Pooling applies per channel —
/// `out[n, c, oy, ox]` is the max of the `kH × kW` window at
/// `(oy·sH, ox·sW)`. The output is `[N, C, H_out, W_out]` with
/// `H_out = (H − kH)/sH + 1` (likewise `W_out`, floor mode — a trailing
/// partial window is dropped); kernel and stride must be positive and
/// the kernel must fit the input. Naive `f64` loops.
#[derive(Debug, Clone, Copy)]
pub struct MaxPool;
impl Function for MaxPool {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        require_arg_range(args, 2, 3, "tensor.max_pool")?;
        let input = require_tensor(args, 0)?;
        if input.shape.len() != 4 {
            return Err(EvalError::Other(format!(
                "tensor.max_pool: expected a rank-4 input [N, C, H, W], got rank {} ({:?})",
                input.shape.len(),
                input.shape
            )));
        }
        if !has_opt_arg(args, 1) {
            return Err(type_mismatch("tensor.max_pool: kernel", "missing"));
        }
        let [k_h, k_w] = opt_extent_pair(args, 1, [0, 0], "tensor.max_pool: kernel")?;
        let [s_h, s_w] = opt_extent_pair(args, 2, [k_h, k_w], "tensor.max_pool: stride")?;
        if k_h == 0 || k_w == 0 {
            return Err(EvalError::Other(format!(
                "tensor.max_pool: kernel must be positive, got [{k_h}, {k_w}]"
            )));
        }
        if s_h == 0 || s_w == 0 {
            return Err(EvalError::Other(format!(
                "tensor.max_pool: stride must be positive, got [{s_h}, {s_w}]"
            )));
        }
        let [n, c, h, w] = [
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        ];
        if k_h > h || k_w > w {
            return Err(EvalError::Other(format!(
                "tensor.max_pool: kernel [{k_h}, {k_w}] exceeds input spatial dims [{h}, {w}]"
            )));
        }
        let h_out = (h - k_h) / s_h + 1;
        let w_out = (w - k_w) / s_w + 1;
        let hw = h * w;
        let mut data = vec![0.0; n * c * h_out * w_out];
        for bn in 0..n {
            for ch in 0..c {
                let in_base = (bn * c + ch) * hw;
                for oy in 0..h_out {
                    for ox in 0..w_out {
                        let mut acc = f64::NEG_INFINITY;
                        for ky in 0..k_h {
                            let iy = oy * s_h + ky;
                            for kx in 0..k_w {
                                let ix = ox * s_w + kx;
                                acc = acc.max(input.data[in_base + iy * w + ix]);
                            }
                        }
                        data[((bn * c + ch) * h_out + oy) * w_out + ox] = acc;
                    }
                }
            }
        }
        Ok(encode_tensor(Tensor {
            shape: vec![n, c, h_out, w_out],
            data,
        }))
    }
}

/// The Rust handl for `op "tensor.batch_norm" : Value... -> Tensor`.
///
/// Eval-mode (inference) batch normalization:
/// `batch_norm(input, gamma, beta, mean, var, eps?)` with `input` rank-4
/// `[N, C, H, W]` and `gamma` / `beta` / `mean` / `var` rank-1 `[C]`
/// learned/statistical parameters. Per channel `c`,
/// `out = (x − mean[c]) / sqrt(var[c] + eps) · gamma[c] + beta[c]`.
/// `eps` is `null`/absent (→ `1e-5`) or a non-negative finite number.
#[derive(Debug, Clone, Copy)]
pub struct BatchNorm;
impl Function for BatchNorm {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        require_arg_range(args, 5, 6, "tensor.batch_norm")?;
        let input = require_tensor(args, 0)?;
        if input.shape.len() != 4 {
            return Err(EvalError::Other(format!(
                "tensor.batch_norm: expected a rank-4 input [N, C, H, W], got rank {} ({:?})",
                input.shape.len(),
                input.shape
            )));
        }
        let c = input.shape[1];
        let params: Vec<Tensor> = (1..5)
            .map(|pos| require_tensor(args, pos))
            .collect::<Result<_, _>>()?;
        let names = ["gamma", "beta", "mean", "var"];
        for (name, p) in names.iter().zip(params.iter()) {
            if p.shape != [c] {
                return Err(EvalError::Other(format!(
                    "tensor.batch_norm: {name} shape {:?} != [{c}]",
                    p.shape
                )));
            }
        }
        let [gamma, beta, mean, var] = [&params[0], &params[1], &params[2], &params[3]];
        let eps = opt_nonneg_float(args, 5, 1e-5, "tensor.batch_norm: eps")?;
        let hw = input.shape[2] * input.shape[3];
        let mut data = vec![0.0; input.data.len()];
        for bn in 0..input.shape[0] {
            for ch in 0..c {
                let inv_std = (var.data[ch] + eps).sqrt().recip();
                let g = gamma.data[ch] * inv_std;
                let shift = beta.data[ch] - mean.data[ch] * g;
                let base = (bn * c + ch) * hw;
                for i in 0..hw {
                    data[base + i] = input.data[base + i] * g + shift;
                }
            }
        }
        Ok(encode_tensor(Tensor {
            shape: input.shape,
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
    registry.register("tensor.mul", Arc::new(Mul));
    registry.register("tensor.sub", Arc::new(Sub));
    registry.register("tensor.div", Arc::new(Div));
    registry.register("tensor.matmul", Arc::new(Matmul));
    registry.register("tensor.conv2d", Arc::new(Conv2d));
    registry.register("tensor.max_pool", Arc::new(MaxPool));
    registry.register("tensor.batch_norm", Arc::new(BatchNorm));
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
            &[tensor(&[2], &[1.0, 2.0]), tensor(&[2], &[10.0, 20.0])],
        )
        .unwrap();
        assert_eq!(v, tensor(&[2], &[11.0, 22.0]));
    }

    #[test]
    fn add_broadcasts_trailing_dim() {
        // [2,3] + [3] → each row plus the vector.
        let v = call_fn(
            &Add,
            &[
                tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                tensor(&[3], &[10.0, 20.0, 30.0]),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[2, 3], &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]));
    }

    #[test]
    fn add_broadcasts_one_dim_and_scalar() {
        // [2,1] + [1,3] → [2,3]: the column vector stretches across
        // columns, the row vector across rows — outer-sum semantics.
        let v = call_fn(
            &Add,
            &[
                tensor(&[2, 1], &[1.0, 10.0]),
                tensor(&[1, 3], &[100.0, 200.0, 300.0]),
            ],
        )
        .unwrap();
        assert_eq!(
            v,
            tensor(&[2, 3], &[101.0, 201.0, 301.0, 110.0, 210.0, 310.0])
        );
        // Rank-0 (shape []) is the scalar case: stretches everywhere.
        let v = call_fn(&Add, &[tensor(&[], &[5.0]), tensor(&[2], &[1.0, 2.0])]).unwrap();
        assert_eq!(v, tensor(&[2], &[6.0, 7.0]));
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
        // [2,3] vs [2]: trailing dims 3 vs 2 differ, neither is 1.
        assert!(
            call_fn(
                &Add,
                &[tensor(&[2, 3], &[0.0; 6]), tensor(&[2], &[0.0, 0.0])]
            )
            .is_err()
        );
    }

    #[test]
    fn mul_broadcasts() {
        let v = call_fn(
            &Mul,
            &[
                tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                tensor(&[3], &[2.0, 3.0, 4.0]),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[2, 3], &[2.0, 6.0, 12.0, 8.0, 15.0, 24.0]));
    }

    #[test]
    fn sub_broadcasts() {
        let v = call_fn(
            &Sub,
            &[
                tensor(&[2, 2], &[10.0, 20.0, 30.0, 40.0]),
                tensor(&[2, 1], &[1.0, 2.0]),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[2, 2], &[9.0, 19.0, 28.0, 38.0]));
    }

    #[test]
    fn div_broadcasts_and_div_by_zero_is_f64_semantics() {
        let v = call_fn(
            &Div,
            &[
                tensor(&[2, 2], &[8.0, 9.0, 10.0, 12.0]),
                tensor(&[1, 2], &[2.0, 3.0]),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[2, 2], &[4.0, 3.0, 5.0, 4.0]));
        // x/0 is inf — the f64 semantics the scalar `/` gives, not an
        // error. The inf crosses the JSON boundary as `null` (no
        // non-finite f64 in serde_json) — the codec's documented hole.
        let v = call_fn(&Div, &[tensor(&[1], &[1.0]), tensor(&[1], &[0.0])]).unwrap();
        assert_eq!(v["__tensor__"]["data"][0], Value::Null);
    }

    #[test]
    fn elementwise_rejects_incompatible_broadcast() {
        assert!(call_fn(&Mul, &[tensor(&[2, 2], &[0.0; 4]), tensor(&[3], &[0.0; 3])]).is_err());
        assert!(call_fn(&Sub, &[tensor(&[4], &[0.0; 4]), tensor(&[2, 3], &[0.0; 6])]).is_err());
        assert!(
            call_fn(
                &Div,
                &[tensor(&[3, 1], &[0.0; 3]), tensor(&[2, 2], &[0.0; 4])]
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
        assert!(call_fn(&Matmul, &[tensor(&[4], &[0.0; 4]), tensor(&[4], &[0.0; 4])]).is_err());
    }

    // ── conv2d ────────────────────────────────────────────────────────

    #[test]
    fn conv2d_1x1_kernel_is_per_pixel_scale() {
        // input [1,2,1,2] = [[1,2],[3,4]] per channel; weight [2,2,1,1]
        // picks out channels: out0 = 1·x0 + 2·x1, out1 = -x0 + 0·x1.
        let input = tensor(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let weight = tensor(&[2, 2, 1, 1], &[1.0, 2.0, -1.0, 0.0]);
        let v = call_fn(&Conv2d, &[input, weight]).unwrap();
        assert_eq!(v, tensor(&[1, 2, 1, 2], &[7.0, 10.0, -1.0, -2.0]));
    }

    #[test]
    fn conv2d_3x3_with_padding() {
        // input [1,1,2,2] = [[1,2],[3,4]]; all-ones 3x3 kernel, padding 1 →
        // each output is the sum of the (zero-padded) 3x3 neighborhood.
        let input = tensor(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let weight = tensor(&[1, 1, 3, 3], &[1.0; 9]);
        // args: (input, weight, bias=null, stride=null→1, padding=1).
        let v = call_fn(
            &Conv2d,
            &[input, weight, Value::Null, Value::Null, json!(1)],
        )
        .unwrap();
        // Pad 1 turns the 2x2 input into 4x4; a 3x3 window at output
        // position (oy, ox) covers input rows/cols {oy−1, oy, oy+1} —
        // which for oy ∈ {0, 1} always contains both real rows {0, 1}
        // (the third row is padding). So every output sums all four
        // input cells: 1+2+3+4 = 10.
        assert_eq!(v, tensor(&[1, 1, 2, 2], &[10.0, 10.0, 10.0, 10.0]));
    }

    #[test]
    fn conv2d_no_padding_valid_region() {
        // input [1,1,3,3] = 1..9; 2x2 kernel [1,0;0,1] picks the diagonal.
        let input = tensor(
            &[1, 1, 3, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let weight = tensor(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let v = call_fn(&Conv2d, &[input, weight]).unwrap();
        // out(0,0)=1+5=6, (0,1)=2+6=8, (1,0)=4+8=12, (1,1)=5+9=14.
        assert_eq!(v, tensor(&[1, 1, 2, 2], &[6.0, 8.0, 12.0, 14.0]));
    }

    #[test]
    fn conv2d_stride_and_bias() {
        // input [1,1,3,3] = 1..9; 2x2 all-ones kernel, stride 2, bias 10.
        let input = tensor(
            &[1, 1, 3, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let weight = tensor(&[1, 1, 2, 2], &[1.0; 4]);
        let bias = tensor(&[1], &[10.0]);
        let v = call_fn(&Conv2d, &[input, weight, bias, json!([2, 2])]).unwrap();
        // h_out = (3−2)/2+1 = 1 (floor); only position (0,0): 1+2+4+5+10 = 22.
        assert_eq!(v, tensor(&[1, 1, 1, 1], &[22.0]));
    }

    #[test]
    fn conv2d_groups_depthwise() {
        // Depthwise: groups = C_in = C_out = 2, per-channel 1x1 scales.
        let input = tensor(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let weight = tensor(&[2, 1, 1, 1], &[10.0, 100.0]);
        let v = call_fn(
            &Conv2d,
            &[
                input,
                weight,
                Value::Null,
                Value::Null,
                Value::Null,
                json!(2),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[1, 2, 1, 2], &[10.0, 20.0, 300.0, 400.0]));
    }

    #[test]
    fn conv2d_rejects_bad_shapes_and_params() {
        let input = tensor(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let weight = tensor(&[1, 1, 1, 1], &[1.0]);
        // rank mismatch.
        assert!(call_fn(&Conv2d, &[tensor(&[4], &[0.0; 4]), weight.clone()]).is_err());
        // channel mismatch (weight has 2 in-channels, input 1).
        assert!(
            call_fn(
                &Conv2d,
                &[input.clone(), tensor(&[1, 2, 1, 1], &[1.0, 1.0])]
            )
            .is_err()
        );
        // bias wrong length.
        assert!(
            call_fn(
                &Conv2d,
                &[input.clone(), weight.clone(), tensor(&[2], &[1.0, 2.0])]
            )
            .is_err()
        );
        // kernel larger than input.
        assert!(call_fn(&Conv2d, &[input.clone(), tensor(&[1, 1, 3, 3], &[1.0; 9])]).is_err());
        // stride 0.
        assert!(
            call_fn(
                &Conv2d,
                &[input.clone(), weight.clone(), Value::Null, json!(0)]
            )
            .is_err()
        );
        // groups not dividing C_out.
        assert!(
            call_fn(
                &Conv2d,
                &[
                    input.clone(),
                    tensor(&[3, 1, 1, 1], &[1.0; 3]),
                    Value::Null,
                    Value::Null,
                    Value::Null,
                    json!(2)
                ]
            )
            .is_err()
        );
        // arity: too few / too many.
        assert!(call_fn(&Conv2d, &[input.clone()]).is_err());
        assert!(
            call_fn(
                &Conv2d,
                &[
                    input,
                    weight,
                    Value::Null,
                    Value::Null,
                    Value::Null,
                    Value::Null,
                    Value::Null,
                ]
            )
            .is_err()
        );
    }

    // ── max_pool ──────────────────────────────────────────────────────

    #[test]
    fn max_pool_2x2_default_stride() {
        // input [1,1,4,4] = 1..16; 2x2 kernel, stride defaults to kernel.
        let input = tensor(
            &[1, 1, 4, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );
        let v = call_fn(&MaxPool, &[input, json!(2)]).unwrap();
        assert_eq!(v, tensor(&[1, 1, 2, 2], &[6.0, 8.0, 14.0, 16.0]));
    }

    #[test]
    fn max_pool_per_channel_and_batch() {
        // [1,2,2,2]: channel maxima are independent.
        let input = tensor(&[1, 2, 2, 2], &[1.0, 9.0, 3.0, 4.0, 8.0, 2.0, 6.0, 5.0]);
        let v = call_fn(&MaxPool, &[input, json!([2, 2])]).unwrap();
        assert_eq!(v, tensor(&[1, 2, 1, 1], &[9.0, 8.0]));
    }

    #[test]
    fn max_pool_stride_1_overlapping_windows() {
        // input [1,1,2,3] = [[1,5,2],[3,4,6]]; 2x2 kernel, stride 1 →
        // windows (0,0): max(1,5,3,4)=5, (0,1): max(5,2,4,6)=6.
        let input = tensor(&[1, 1, 2, 3], &[1.0, 5.0, 2.0, 3.0, 4.0, 6.0]);
        let v = call_fn(&MaxPool, &[input, json!([2, 2]), json!([1, 1])]).unwrap();
        assert_eq!(v, tensor(&[1, 1, 1, 2], &[5.0, 6.0]));
    }

    #[test]
    fn max_pool_stride_larger_than_window_drops_tail() {
        // input [1,1,2,5] with kernel [1,3], stride [1,2] → floor: only
        // windows at ox=0 and ox=2 count (ox=4 is a dropped tail).
        let input = tensor(
            &[1, 1, 2, 5],
            &[1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0, 0.0],
        );
        let v = call_fn(&MaxPool, &[input, json!([1, 3]), json!([1, 2])]).unwrap();
        // w_out = (5−3)/2+1 = 2; row 0 windows {1,9,2}→9 / {2,8,3}→8;
        // row 1 windows {7,4,6}→7 / {6,5,0}→6.
        assert_eq!(v, tensor(&[1, 1, 2, 2], &[9.0, 8.0, 7.0, 6.0]));
    }

    #[test]
    fn max_pool_rejects_bad_params() {
        let input = tensor(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        // non-rank-4 input.
        assert!(call_fn(&MaxPool, &[tensor(&[4], &[0.0; 4]), json!(2)]).is_err());
        // kernel larger than the input.
        assert!(call_fn(&MaxPool, &[input.clone(), json!(3)]).is_err());
        // zero kernel / stride.
        assert!(call_fn(&MaxPool, &[input.clone(), json!(0)]).is_err());
        assert!(call_fn(&MaxPool, &[input.clone(), json!(1), json!(0)]).is_err());
        // missing kernel / extra arg.
        assert!(call_fn(&MaxPool, &[input.clone()]).is_err());
        assert!(call_fn(&MaxPool, &[input.clone(), json!(2), json!(2), json!(2)]).is_err());
    }

    // ── batch_norm ────────────────────────────────────────────────────

    #[test]
    fn batch_norm_normalizes_per_channel() {
        // input [1,2,1,2] = [[1,2],[3,4]]; eps = 0 for exact math.
        // c0: (x−0)/√1 ·1 +0 = x; c1: (x−3)/√4 ·2 +1 = x−2.
        let input = tensor(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let args = [
            input,
            tensor(&[2], &[1.0, 2.0]), // gamma
            tensor(&[2], &[0.0, 1.0]), // beta
            tensor(&[2], &[0.0, 3.0]), // mean
            tensor(&[2], &[1.0, 4.0]), // var
            json!(0),
        ];
        let v = call_fn(&BatchNorm, &args).unwrap();
        assert_eq!(v, tensor(&[1, 2, 1, 2], &[1.0, 2.0, 1.0, 2.0]));
    }

    #[test]
    fn batch_norm_default_eps_and_batch_dim() {
        // N=2, C=1: x = (x − 5)/√(4 + 1e-5).
        let input = tensor(&[2, 1, 1, 1], &[3.0, 7.0]);
        let v = call_fn(
            &BatchNorm,
            &[
                input,
                tensor(&[1], &[1.0]),
                tensor(&[1], &[0.0]),
                tensor(&[1], &[5.0]),
                tensor(&[1], &[4.0]),
            ],
        )
        .unwrap();
        let t = decode_tensor(&v).unwrap();
        let inv = (4.0f64 + 1e-5).sqrt().recip();
        assert!((t.data[0] - (3.0 - 5.0) * inv).abs() < 1e-12);
        assert!((t.data[1] - (7.0 - 5.0) * inv).abs() < 1e-12);
    }

    #[test]
    fn batch_norm_rejects_bad_shapes_and_eps() {
        let input = tensor(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let good = |p: &[f64]| tensor(&[2], p);
        // non-rank-4 input.
        assert!(
            call_fn(
                &BatchNorm,
                &[
                    tensor(&[4], &[0.0; 4]),
                    good(&[1.0; 2]),
                    good(&[0.0; 2]),
                    good(&[0.0; 2]),
                    good(&[1.0; 2]),
                ]
            )
            .is_err()
        );
        // param wrong length (gamma [3] vs C=2).
        assert!(
            call_fn(
                &BatchNorm,
                &[
                    input.clone(),
                    tensor(&[3], &[1.0; 3]),
                    good(&[0.0; 2]),
                    good(&[0.0; 2]),
                    good(&[1.0; 2]),
                ]
            )
            .is_err()
        );
        // negative eps.
        assert!(
            call_fn(
                &BatchNorm,
                &[
                    input.clone(),
                    good(&[1.0; 2]),
                    good(&[0.0; 2]),
                    good(&[0.0; 2]),
                    good(&[1.0; 2]),
                    json!(-1),
                ]
            )
            .is_err()
        );
        // arity: missing var / extra arg.
        assert!(
            call_fn(
                &BatchNorm,
                &[
                    input.clone(),
                    good(&[1.0; 2]),
                    good(&[0.0; 2]),
                    good(&[0.0; 2])
                ]
            )
            .is_err()
        );
        assert!(
            call_fn(
                &BatchNorm,
                &[
                    input,
                    good(&[1.0; 2]),
                    good(&[0.0; 2]),
                    good(&[0.0; 2]),
                    good(&[1.0; 2]),
                    json!(0),
                    json!(0),
                ]
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
                &[tensor(&[1, 2], &[3.0, 4.0]), tensor(&[2, 1], &[5.0, 6.0])],
                &mut ctx,
            )
            .unwrap();
        assert_eq!(v, tensor(&[1, 1], &[39.0]));
    }
}
