//! Kernel cores for the `aten-port-gen` generated ops — the semantic
//! half of the ATen port. `src/ops/aten_generated.rs` (generated, do not
//! edit) holds one thin `Function` per op: it decodes the positional
//! arguments per the ATen schema and calls the entry point here.
//!
//! ## Conventions
//!
//! * Same tensor wire value as the rest of `ops/` — dense row-major
//!   `f64`, `{"__tensor__": {"shape", "data", "dtype"}}`. No bool dtype:
//!   comparisons and predicates yield `1.0`/`0.0`, `logical_*` the same.
//! * Entry points take *decoded* arguments (`Tensor`, `i64`, `f64`,
//!   `bool`, `Vec<i64>`, `Option<…>`) and return `Result<Value,
//!   EvalError>` — the encoded wire value, or a JSON object/array for
//!   multi-return ops (`{"values", "indices"}` for `sort`/`topk`/
//!   `max.dim`/`min.dim`, an array for `unbind`/`chunk`/`split`/`stack`).
//! * Optional ATen args arrive as `Option`/defaulted values; `None`
//!   selects the ATen default documented per function.
//! * Everything is naive `f64` loops — the composition seam, not BLAS,
//!   matching `ops/matmul.rs`.
//!
//! ## Schema features deliberately not represented
//!
//! * `ScalarType`/`Layout`/`Device`/`MemoryFormat`/`Generator`/`Dimname`
//!   args — dtype/device/layout selection; the surface is f64 CPU dense.
//!   Trailing *optional* args of these kinds are dropped by the
//!   generator (they can only select unsupported features); a *required*
//!   one makes the op a seam.
//! * Inplace (`foo_`), `.out`, `Tensor(a!)`-mutated, and `_`-private
//!   schemas — filtered before port selection.
//! * `int[]? dim=None` reductions: `None` reduces ALL dims; an empty
//!   list reduces NONE (identity), matching torch's empty-tuple rule.

// Kernel catalog: entries are wired by `aten_generated.rs` (generated)
// or reserved for sibling overloads — not every `k_*` is reachable
// from the current port list.
#![allow(dead_code)]

use handl_platform::EvalError;
use serde_json::{Value, json};

use super::codec::{Tensor, decode_tensor, encode_tensor, row_major_strides};
use super::elementwise::broadcast_elementwise;
use super::shape_ops::transpose_dims;

pub(crate) fn op_err(op: &str, msg: impl std::fmt::Display) -> EvalError {
    EvalError::Other(format!("tensor.{op}: {msg}"))
}

fn type_mismatch(what: &str) -> EvalError {
    EvalError::TypeMismatch(format!("expected {what}"))
}

// ---------------------------------------------------------------------------
// Argument decoders (generated code calls these positionally)
// ---------------------------------------------------------------------------

/// A required signed integer arg (`int`/`SymInt`) — dims may be negative.
pub(crate) fn req_i64(arg: &Value) -> Result<i64, EvalError> {
    arg.as_i64()
        .or_else(|| arg.as_u64().and_then(|n| i64::try_from(n).ok()))
        .or_else(|| {
            arg.as_f64()
                .filter(|n| n.fract() == 0.0 && n.is_finite())
                .map(|n| n as i64)
        })
        .ok_or_else(|| type_mismatch("integer"))
}

/// An optional signed integer with an ATen default (`int dim=-1`).
pub(crate) fn opt_i64(arg: Option<&Value>, default: i64) -> Result<i64, EvalError> {
    match arg {
        Some(v) if !v.is_null() => req_i64(v),
        _ => Ok(default),
    }
}

/// A nullable signed integer (`int? dim=None`) — `None`/absent/`null` → `None`.
pub(crate) fn opt_i64n(arg: Option<&Value>) -> Result<Option<i64>, EvalError> {
    match arg {
        Some(v) if !v.is_null() => Ok(Some(req_i64(v)?)),
        _ => Ok(None),
    }
}

/// A required finite float (`float`/`Scalar`).
pub(crate) fn req_f64(arg: &Value) -> Result<f64, EvalError> {
    arg.as_f64()
        .ok_or_else(|| type_mismatch("number"))
}

/// An optional float with an ATen default (`Scalar beta=1`).
pub(crate) fn opt_f64(arg: Option<&Value>, default: f64) -> Result<f64, EvalError> {
    match arg {
        Some(v) if !v.is_null() => req_f64(v),
        _ => Ok(default),
    }
}

/// A nullable float (`float? eps=None`, `Scalar? p=None`).
pub(crate) fn opt_f64n(arg: Option<&Value>) -> Result<Option<f64>, EvalError> {
    match arg {
        Some(v) if !v.is_null() => Ok(Some(req_f64(v)?)),
        _ => Ok(None),
    }
}

/// A required bool.
pub(crate) fn req_bool(arg: &Value) -> Result<bool, EvalError> {
    arg.as_bool().ok_or_else(|| type_mismatch("boolean"))
}

/// An optional bool with an ATen default (`bool keepdim=False`).
pub(crate) fn opt_bool(arg: Option<&Value>, default: bool) -> Result<bool, EvalError> {
    match arg {
        Some(v) if !v.is_null() => req_bool(v),
        _ => Ok(default),
    }
}

/// A nullable bool (`bool? stable=None`).
pub(crate) fn opt_booln(arg: Option<&Value>) -> Result<Option<bool>, EvalError> {
    match arg {
        Some(v) if !v.is_null() => Ok(Some(req_bool(v)?)),
        _ => Ok(None),
    }
}

/// A required int list (`int[]`/`SymInt[]`/`int[1]`) — elements may be
/// negative (dims); fixed-width `[N]` types are validated by the kernel.
pub(crate) fn req_i64s(arg: &Value) -> Result<Vec<i64>, EvalError> {
    let arr = arg.as_array().ok_or_else(|| type_mismatch("int list"))?;
    arr.iter().map(req_i64).collect()
}

/// An optional int list with an ATen default (`int[]? dim=None` →
/// `default: Option<&[i64]>`; `int[1] dim=[]` → `Some(&[])`).
pub(crate) fn opt_i64s(
    arg: Option<&Value>,
    default: Option<&[i64]>,
) -> Result<Option<Vec<i64>>, EvalError> {
    match arg {
        Some(v) if !v.is_null() => Ok(Some(req_i64s(v)?)),
        _ => Ok(default.map(|d| d.to_vec())),
    }
}

/// A required float list (`Scalar[]`).
pub(crate) fn req_f64s(arg: &Value) -> Result<Vec<f64>, EvalError> {
    let arr = arg.as_array().ok_or_else(|| type_mismatch("number list"))?;
    arr.iter().map(req_f64).collect()
}

/// A required tensor list (`Tensor[]`).
pub(crate) fn req_tensors(arg: &Value) -> Result<Vec<Tensor>, EvalError> {
    let arr = arg
        .as_array()
        .ok_or_else(|| type_mismatch("tensor list"))?;
    arr.iter().map(decode_tensor).collect()
}

/// A tensor-or-number argument (`Scalar | Tensor` rhs of `pow`,
/// comparisons, `clamp`, `where`, `xlogy`, `remainder`, `lerp`).
/// Returns `Ok(None)` for a number, `Ok(Some(tensor))` for a tensor.
pub(crate) fn opt_tensor_or_num(arg: &Value) -> Result<Option<Tensor>, EvalError> {
    if arg.is_number() {
        return Ok(None);
    }
    Ok(Some(decode_tensor(arg)?))
}

/// Fetch `args[pos]` or a type error — generated code uses this so a
/// missing required argument fails cleanly instead of panicking.
pub(crate) fn arg_at<'a>(args: &'a [Value], pos: usize) -> Result<&'a Value, EvalError> {
    args.get(pos)
        .ok_or_else(|| type_mismatch("argument"))
}

/// A dim spec (`int`, `int[]`, `int[1]`, nullable variants): accepts a
/// bare int, a list, or null → `None`. `default` applies when the arg
/// is absent/null (pass `None` for "absent means reduce-all").
pub(crate) fn opt_dimlist(
    arg: Option<&Value>,
    default: Option<&[i64]>,
) -> Result<Option<Vec<i64>>, EvalError> {
    match arg {
        Some(v) if !v.is_null() => {
            if v.is_array() {
                Ok(Some(req_i64s(v)?))
            } else {
                Ok(Some(vec![req_i64(v)?]))
            }
        }
        _ => Ok(default.map(|d| d.to_vec())),
    }
}

/// Required dim spec: bare int or list → `Vec<i64>`.
pub(crate) fn req_dimlist(arg: &Value) -> Result<Vec<i64>, EvalError> {
    if arg.is_array() {
        req_i64s(arg)
    } else {
        Ok(vec![req_i64(arg)?])
    }
}

// ---------------------------------------------------------------------------
// Shared tensor machinery
// ---------------------------------------------------------------------------

/// Elementwise `f` over `t` (no broadcasting — unary ops).
pub(crate) fn map_unary(t: &Tensor, f: impl Fn(f64) -> f64) -> Tensor {
    Tensor {
        shape: t.shape.clone(),
        data: t.data.iter().map(|&x| f(x)).collect(),
    }
}

/// `t` relabeled under `shape` (the `reshape`/`view` rule: product must
/// match, row-major payload unchanged).
fn reshape_like(t: &Tensor, shape: &[usize], op: &str) -> Result<Tensor, EvalError> {
    let expected: usize = shape.iter().product();
    if expected != t.data.len() {
        return Err(op_err(
            op,
            format!(
                "shape {shape:?} has {expected} elements, tensor {:?} has {}",
                t.shape,
                t.data.len()
            ),
        ));
    }
    Ok(Tensor {
        shape: shape.to_vec(),
        data: t.data.clone(),
    })
}

/// Wrap a possibly-negative dim into `[0, rank)`; error when `rank == 0`
/// or the wrapped value is out of range.
pub(crate) fn wrap_dim(dim: i64, rank: usize, op: &str) -> Result<usize, EvalError> {
    if rank == 0 {
        return Err(op_err(op, "cannot index an axis of a rank-0 tensor"));
    }
    let d = dim.rem_euclid(rank as i64);
    Ok(d as usize)
}

/// Normalize a reduction `dim` spec: `None` → all axes; `Some([])` →
/// no axes (identity); otherwise wrap negatives, dedupe-check.
fn norm_reduce_dims(
    t: &Tensor,
    dims: Option<&[i64]>,
    op: &str,
) -> Result<Option<Vec<usize>>, EvalError> {
    let rank = t.shape.len();
    match dims {
        None => Ok(Some((0..rank).collect())),
        Some(d) if d.is_empty() => Ok(None),
        Some(d) => {
            let mut out = Vec::with_capacity(d.len());
            for &x in d {
                let w = wrap_dim(x, rank, op)?;
                if out.contains(&w) {
                    return Err(op_err(op, format!("duplicate dim {x}")));
                }
                out.push(w);
            }
            Ok(Some(out))
        }
    }
}

/// Generic reduction: `f(acc, x)` folded over the reduced axes.
/// `init` seeds each output element. `keepdim` retains size-1 axes.
fn reduce_fold(
    t: &Tensor,
    dims: Option<&[i64]>,
    keepdim: bool,
    init: f64,
    f: impl Fn(f64, f64) -> f64,
    op: &str,
) -> Result<Tensor, EvalError> {
    let Some(red) = norm_reduce_dims(t, dims, op)? else {
        // Empty dim list → identity (torch's empty-tuple rule).
        return Ok(t.clone());
    };
    let rank = t.shape.len();
    let mut out_shape: Vec<usize> = (0..rank).map(|d| t.shape[d]).collect();
    for &d in &red {
        out_shape[d] = if keepdim { 1 } else { out_shape[d] };
    }
    if !keepdim {
        out_shape = out_shape
            .iter()
            .enumerate()
            .filter(|(d, _)| !red.contains(d))
            .map(|(_, &s)| s)
            .collect();
    }
    let in_strides = row_major_strides(&t.shape);
    let count: usize = out_shape.iter().product();
    let mut data = Vec::with_capacity(count);
    // Odometer over the *kept* output axes; inner loop walks the
    // reduced axes' index space.
    let kept: Vec<usize> = (0..rank).filter(|d| !red.contains(d)).collect();
    let red_ext: Vec<usize> = red.iter().map(|&d| t.shape[d]).collect();
    let red_count: usize = red_ext.iter().product();
    let mut idx = vec![0usize; kept.len()];
    let mut ridx = vec![0usize; red.len()];
    for _ in 0..count {
        let mut acc = init;
        for _ in 0..red_count {
            let mut off = 0usize;
            for (j, &d) in kept.iter().enumerate() {
                off += idx[j] * in_strides[d];
            }
            for (j, &d) in red.iter().enumerate() {
                off += ridx[j] * in_strides[d];
            }
            acc = f(acc, t.data[off]);
            for j in (0..red.len()).rev() {
                ridx[j] += 1;
                if ridx[j] < red_ext[j] {
                    break;
                }
                ridx[j] = 0;
            }
        }
        data.push(acc);
        for j in (0..kept.len()).rev() {
            idx[j] += 1;
            if idx[j] < t.shape[kept[j]] {
                break;
            }
            idx[j] = 0;
        }
    }
    Ok(Tensor {
        shape: out_shape,
        data,
    })
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

fn enc(t: Tensor) -> Value {
    encode_tensor(t)
}

/// `sum`/`prod`/`mean`/`amax`/`amin` share this driver shape:
/// `(self, dim?, keepdim?)`.
pub(crate) fn reduce_sum(t: &Tensor, dims: Option<Vec<i64>>, keepdim: bool) -> Result<Tensor, EvalError> {
    reduce_fold(t, dims.as_deref(), keepdim, 0.0, |a, b| a + b, "sum")
}

pub(crate) fn reduce_prod(t: &Tensor, dims: Option<Vec<i64>>, keepdim: bool) -> Result<Tensor, EvalError> {
    reduce_fold(t, dims.as_deref(), keepdim, 1.0, |a, b| a * b, "prod")
}

pub(crate) fn reduce_mean(t: &Tensor, dims: Option<Vec<i64>>, keepdim: bool) -> Result<Tensor, EvalError> {
    let s = reduce_sum(t, dims.clone(), keepdim)?;
    // Count of reduced elements per output — uniform across outputs.
    let Some(red) = norm_reduce_dims(t, dims.as_deref(), "mean")? else {
        return Ok(t.clone());
    };
    let n: usize = red.iter().map(|&d| t.shape[d]).product();
    if n == 0 {
        return Ok(s);
    }
    Ok(Tensor {
        shape: s.shape,
        data: s.data.iter().map(|&x| x / n as f64).collect(),
    })
}

pub(crate) fn reduce_amax(t: &Tensor, dims: Option<Vec<i64>>, keepdim: bool) -> Result<Tensor, EvalError> {
    reduce_fold(t, dims.as_deref(), keepdim, f64::NEG_INFINITY, f64::max, "amax")
}

pub(crate) fn reduce_amin(t: &Tensor, dims: Option<Vec<i64>>, keepdim: bool) -> Result<Tensor, EvalError> {
    reduce_fold(t, dims.as_deref(), keepdim, f64::INFINITY, f64::min, "amin")
}

/// `logsumexp(self, dim, keepdim)` — dim is a required `int[1]`.
/// Computed with keepdim internally (broadcast-friendly), then the
/// reduced axes are squeezed back out when `keepdim` is false.
pub(crate) fn reduce_logsumexp(t: &Tensor, dims: Vec<i64>, keepdim: bool) -> Result<Tensor, EvalError> {
    // max-subtract for stability, then log(sum(exp(x - max))) + max.
    let mx = reduce_amax(t, Some(dims.clone()), true)?;
    let full = broadcast_to(&mx, &t.shape, "logsumexp")?;
    let shifted = Tensor {
        shape: t.shape.clone(),
        data: t
            .data
            .iter()
            .zip(&full.data)
            .map(|(&x, &m)| (x - m).exp())
            .collect(),
    };
    let s = reduce_sum(&shifted, Some(dims.clone()), true)?;
    let out = Tensor {
        shape: s.shape.clone(),
        data: s
            .data
            .iter()
            .zip(&mx.data)
            .map(|(&a, &m)| a.ln() + m)
            .collect(),
    };
    if keepdim {
        return Ok(out);
    }
    // Squeeze the reduced axes (they are size-1 in `out`).
    let red = norm_reduce_dims(t, Some(&dims), "logsumexp")?.unwrap_or_default();
    let mut shape = out.shape.clone();
    for &d in red.iter().rev() {
        if d < shape.len() && shape[d] == 1 {
            shape.remove(d);
        }
    }
    Ok(Tensor { shape, data: out.data })
}

/// Broadcast `src` (a reduction output) back to `dst` shape: dims that
/// were kept-as-1 or dropped re-stretch. `src` must be broadcastable
/// to `dst` under numpy rules.
fn broadcast_to(src: &Tensor, dst: &[usize], op: &str) -> Result<Tensor, EvalError> {
    // Manual broadcast: align trailing dims.
    let rank = dst.len();
    let off = rank - src.shape.len();
    let mut strides = vec![0usize; rank];
    let inner = row_major_strides(&src.shape);
    for (i, &s) in inner.iter().enumerate() {
        let d = off + i;
        strides[d] = if src.shape[i] == 1 { 0 } else { s };
    }
    // Validate broadcastability.
    for i in 0..rank {
        let sd = if i < off { 1 } else { src.shape[i - off] };
        if sd != 1 && sd != dst[i] {
            return Err(op_err(
                op,
                format!("cannot broadcast {:?} to {dst:?}", src.shape),
            ));
        }
    }
    let count: usize = dst.iter().product();
    let mut data = Vec::with_capacity(count);
    let mut idx = vec![0usize; rank];
    for _ in 0..count {
        let mut o = 0usize;
        for d in 0..rank {
            o += idx[d] * strides[d];
        }
        data.push(src.data[o]);
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < dst[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(Tensor {
        shape: dst.to_vec(),
        data,
    })
}

/// `argmax`/`argmin(self, dim?, keepdim?)` — returns f64 indices
/// (no int dtype on the wire; documented seam).
pub(crate) fn reduce_arg(
    t: &Tensor,
    dims: Option<i64>,
    keepdim: bool,
    want_max: bool,
    op: &str,
) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    let d = match dims {
        Some(d) => wrap_dim(d, rank, op)?,
        None => {
            // Flatten: argmax over all elements → rank-0 (or all-ones
            // keepdim) index tensor.
            let mut best = 0usize;
            for (i, &x) in t.data.iter().enumerate() {
                let better = if want_max {
                    x > t.data[best]
                } else {
                    x < t.data[best]
                };
                if better {
                    best = i;
                }
            }
            let shape = if keepdim {
                vec![1usize; rank]
            } else {
                Vec::new()
            };
            return Ok(Tensor {
                shape,
                data: vec![best as f64],
            });
        }
    };
    // Kept axes in original order; the reduced axis's extent is `width`.
    let width = t.shape[d];
    let kept: Vec<usize> = (0..rank).filter(|&k| k != d).collect();
    let in_strides = row_major_strides(&t.shape);
    let kept_count: usize = kept.iter().map(|&k| t.shape[k]).product();
    let mut idx_data = Vec::with_capacity(kept_count);
    let mut idx = vec![0usize; kept.len()];
    for _ in 0..kept_count {
        // Base offset of this kept-position, plus the reduced-axis
        // sweep offset (in_strides[d] per step).
        let mut base = 0usize;
        for (j, &k) in kept.iter().enumerate() {
            base += idx[j] * in_strides[k];
        }
        let mut best = 0usize;
        for j in 1..width {
            let x = t.data[base + j * in_strides[d]];
            let b = t.data[base + best * in_strides[d]];
            let better = if want_max { x > b } else { x < b };
            if better {
                best = j;
            }
        }
        idx_data.push(best as f64);
        for j in (0..kept.len()).rev() {
            idx[j] += 1;
            if idx[j] < t.shape[kept[j]] {
                break;
            }
            idx[j] = 0;
        }
    }
    // Output shape: kept dims, with a size-1 axis reinserted at d when
    // keepdim (the data order is unaffected — a 1-extent axis).
    let mut shape: Vec<usize> = kept.iter().map(|&k| t.shape[k]).collect();
    if keepdim {
        let pos = kept.iter().filter(|&&k| k < d).count();
        shape.insert(pos, 1);
    }
    Ok(Tensor {
        shape,
        data: idx_data,
    })
}

/// Permute `t` by `perm` (output axis i reads input axis perm[i]).
pub(crate) fn permute_dims(t: &Tensor, perm: &[usize], op: &str) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    if perm.len() != rank {
        return Err(op_err(op, format!("perm {perm:?} length != rank {rank}")));
    }
    let mut seen = vec![false; rank];
    for &p in perm {
        if p >= rank || seen[p] {
            return Err(op_err(op, format!("invalid perm {perm:?} for rank {rank}")));
        }
        seen[p] = true;
    }
    let out_shape: Vec<usize> = perm.iter().map(|&d| t.shape[d]).collect();
    let in_strides = row_major_strides(&t.shape);
    let count: usize = out_shape.iter().product();
    let mut data = Vec::with_capacity(count);
    let mut idx = vec![0usize; rank];
    for _ in 0..count {
        let off: usize = idx
            .iter()
            .zip(perm)
            .map(|(&i, &d)| i * in_strides[d])
            .sum();
        data.push(t.data[off]);
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

/// `cumsum`/`cumprod` along `dim` (required int). Row-major layout
/// splits flat offsets into (outer, i, tail) blocks.
pub(crate) fn cum_along(t: &Tensor, dim: i64, prod: bool, op: &str) -> Result<Tensor, EvalError> {
    let d = wrap_dim(dim, t.shape.len(), op)?;
    let width = t.shape[d];
    let tail: usize = t.shape[d + 1..].iter().product();
    let outer: usize = t.shape[..d].iter().product();
    let mut data = vec![0.0; t.data.len()];
    for o in 0..outer {
        for i in 0..width {
            for j in 0..tail {
                let at = (o * width + i) * tail + j;
                let prev = if i == 0 {
                    if prod { 1.0 } else { 0.0 }
                } else {
                    data[(o * width + i - 1) * tail + j]
                };
                data[at] = if prod { prev * t.data[at] } else { prev + t.data[at] };
            }
        }
    }
    Ok(Tensor {
        shape: t.shape.clone(),
        data,
    })
}

/// `softmax`/`log_softmax` along required `dim`.
pub(crate) fn softmax_dim(t: &Tensor, dim: i64, log: bool, op: &str) -> Result<Tensor, EvalError> {
    let d = wrap_dim(dim, t.shape.len(), op)?;
    let rank = t.shape.len();
    let width = t.shape[d];
    let tail: usize = t.shape[d + 1..].iter().product();
    let outer: usize = t.shape[..d].iter().product();
    let mut data = vec![0.0; t.data.len()];
    for o in 0..outer {
        for j in 0..tail {
            let base = (o * width) * tail + j;
            // Slice elements are strided by `tail`.
            let mut mx = f64::NEG_INFINITY;
            for i in 0..width {
                mx = mx.max(t.data[base + i * tail]);
            }
            let mut sum = 0.0;
            for i in 0..width {
                sum += (t.data[base + i * tail] - mx).exp();
            }
            let lse = mx + sum.ln();
            for i in 0..width {
                let x = t.data[base + i * tail];
                data[base + i * tail] = if log {
                    x - lse
                } else {
                    (x - lse).exp()
                };
            }
        }
    }
    let _ = rank;
    Ok(Tensor {
        shape: t.shape.clone(),
        data,
    })
}

// ---------------------------------------------------------------------------
// Unary kernels
// ---------------------------------------------------------------------------

fn erf_approx(x: f64) -> f64 {
    // Abramowitz–Stegun 7.1.26 (|ε| ≤ 1.5e-7).
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

macro_rules! unary_op {
    ($name:ident, $f:expr) => {
        pub(crate) fn $name(t: &Tensor) -> Result<Tensor, EvalError> {
            Ok(map_unary(t, $f))
        }
    };
}

unary_op!(k_abs, f64::abs);
unary_op!(k_neg, |x| -x);
unary_op!(k_exp, f64::exp);
unary_op!(k_exp2, f64::exp2);
unary_op!(k_expm1, f64::exp_m1);
unary_op!(k_log, f64::ln);
unary_op!(k_log1p, f64::ln_1p);
unary_op!(k_log2, f64::log2);
unary_op!(k_log10, f64::log10);
unary_op!(k_sqrt, f64::sqrt);
unary_op!(k_rsqrt, |x| 1.0 / x.sqrt());
unary_op!(k_reciprocal, |x| 1.0 / x);
unary_op!(k_square, |x| x * x);
unary_op!(k_sin, f64::sin);
unary_op!(k_cos, f64::cos);
unary_op!(k_tan, f64::tan);
unary_op!(k_asin, f64::asin);
unary_op!(k_acos, f64::acos);
unary_op!(k_atan, f64::atan);
unary_op!(k_sinh, f64::sinh);
unary_op!(k_cosh, f64::cosh);
unary_op!(k_tanh, f64::tanh);
unary_op!(k_asinh, f64::asinh);
unary_op!(k_acosh, f64::acosh);
unary_op!(k_atanh, f64::atanh);
unary_op!(k_floor, f64::floor);
unary_op!(k_ceil, f64::ceil);
unary_op!(k_round, f64::round);
unary_op!(k_trunc, f64::trunc);
unary_op!(k_frac, |x| x - x.trunc());
unary_op!(k_sign, |x| {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x // preserves ±0.0 and NaN
    }
});
unary_op!(k_sigmoid, |x| 1.0 / (1.0 + (-x).exp()));
unary_op!(k_silu, |x| x / (1.0 + (-x).exp()));
unary_op!(k_mish, |x| x * f64::tanh(f64::ln_1p(x.exp())));
unary_op!(k_gelu, |x| 0.5 * x * (1.0 + erf_approx(x / std::f64::consts::SQRT_2)));
/// `gelu(approximate="tanh")` — the tanh surrogate; `"none"`/absent is
/// the exact erf form above.
pub(crate) fn k_gelu_tanh(t: &Tensor) -> Result<Tensor, EvalError> {
    Ok(map_unary(t, |x| {
        let u = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + f64::tanh(u))
    }))
}
unary_op!(k_relu, |x| x.max(0.0));
unary_op!(k_selu, |x| {
    const ALPHA: f64 = 1.6732632423543772848170429916717;
    const SCALE: f64 = 1.0507009873554804934193349852946;
    if x > 0.0 { SCALE * x } else { SCALE * ALPHA * (x.exp() - 1.0) }
});
unary_op!(k_hardswish, |x| {
    let x = x.clamp(0.0, 6.0);
    x * (x + 3.0) / 6.0
});
unary_op!(k_hardsigmoid, |x| (x + 3.0).clamp(0.0, 6.0) / 6.0);
unary_op!(k_softsign, |x| x / (1.0 + x.abs()));
unary_op!(k_logsigmoid, |x| -f64::ln_1p((-x).exp()));
unary_op!(k_tanhshrink, |x| x - f64::tanh(x));
unary_op!(k_erf, erf_approx);
unary_op!(k_erfc, |x| 1.0 - erf_approx(x));
unary_op!(k_sinc, |x| {
    if x == 0.0 {
        1.0
    } else {
        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
    }
});
unary_op!(k_degrees_to_radians, |x| x * std::f64::consts::PI / 180.0);
unary_op!(k_radians_to_degrees, |x| x * 180.0 / std::f64::consts::PI);

/// Parametric unary ops: `(self, scalar…)` with ATen defaults.
pub(crate) fn k_leaky_relu(t: &Tensor, slope: f64) -> Result<Tensor, EvalError> {
    Ok(map_unary(t, |x| if x > 0.0 { x } else { slope * x }))
}
/// `elu` — ATen signature `(self, alpha=1, scale=1, input_scale=1)`:
/// `y = scale * (x' > 0 ? x' : alpha*(exp(x')-1))` with `x' = input_scale*x`.
pub(crate) fn k_elu(t: &Tensor, alpha: f64, scale: f64, input_scale: f64) -> Result<Tensor, EvalError> {
    Ok(map_unary(t, |x| {
        let x = input_scale * x;
        if x > 0.0 {
            scale * x
        } else {
            scale * alpha * (x.exp() - 1.0)
        }
    }))
}
pub(crate) fn k_celu(t: &Tensor, alpha: f64) -> Result<Tensor, EvalError> {
    // CELU: x>0 → x; else alpha*(exp(x/alpha) - 1).
    Ok(map_unary(t, |x| {
        if x > 0.0 {
            x
        } else {
            alpha * ((x / alpha).exp() - 1.0)
        }
    }))
}
pub(crate) fn k_softplus(t: &Tensor, beta: f64, threshold: f64) -> Result<Tensor, EvalError> {
    Ok(map_unary(t, |x| {
        if beta * x > threshold {
            x
        } else {
            f64::ln_1p((beta * x).exp()) / beta
        }
    }))
}
pub(crate) fn k_softshrink(t: &Tensor, lambd: f64) -> Result<Tensor, EvalError> {
    Ok(map_unary(t, |x| {
        if x > lambd {
            x - lambd
        } else if x < -lambd {
            x + lambd
        } else {
            0.0
        }
    }))
}
pub(crate) fn k_hardshrink(t: &Tensor, lambd: f64) -> Result<Tensor, EvalError> {
    Ok(map_unary(t, |x| if x.abs() > lambd { x } else { 0.0 }))
}
pub(crate) fn k_hardtanh(t: &Tensor, min: f64, max: f64) -> Result<Tensor, EvalError> {
    Ok(map_unary(t, |x| x.clamp(min, max)))
}
pub(crate) fn k_threshold(t: &Tensor, thr: f64, val: f64) -> Result<Tensor, EvalError> {
    Ok(map_unary(t, |x| if x > thr { x } else { val }))
}
pub(crate) fn k_logit(t: &Tensor, eps: Option<f64>) -> Result<Tensor, EvalError> {
    // torch.logit: clamp x into [eps, 1-eps] when eps given.
    let (lo, hi) = match eps {
        Some(e) => (e, 1.0 - e),
        None => (f64::MIN_POSITIVE, 1.0 - f64::EPSILON),
    };
    Ok(map_unary(t, |x| {
        let x = x.clamp(lo, hi);
        (x / (1.0 - x)).ln()
    }))
}
pub(crate) fn k_nan_to_num(
    t: &Tensor,
    nan: Option<f64>,
    posinf: Option<f64>,
    neginf: Option<f64>,
) -> Result<Tensor, EvalError> {
    // `None` bounds → f64 limits stand in for the dtype bounds torch
    // would use; explicit values replace them.
    let (npos, nneg) = (posinf.unwrap_or(f64::MAX), neginf.unwrap_or(f64::MIN));
    let nn = nan.unwrap_or(0.0);
    Ok(map_unary(t, |x| {
        if x.is_nan() {
            nn
        } else if x == f64::INFINITY {
            npos
        } else if x == f64::NEG_INFINITY {
            nneg
        } else {
            x
        }
    }))
}

/// Predicate kernels → 1.0/0.0.
macro_rules! pred_op {
    ($name:ident, $f:expr) => {
        pub(crate) fn $name(t: &Tensor) -> Result<Tensor, EvalError> {
            Ok(map_unary(t, |x| if $f(x) { 1.0 } else { 0.0 }))
        }
    };
}
pred_op!(k_isfinite, f64::is_finite);
pred_op!(k_isinf, f64::is_infinite);
pred_op!(k_isnan, f64::is_nan);
pred_op!(k_isposinf, |x| x == f64::INFINITY);
pred_op!(k_isneginf, |x| x == f64::NEG_INFINITY);
pred_op!(k_signbit, |x: f64| x.is_sign_negative());
pred_op!(k_logical_not, |x| x == 0.0);

// ---------------------------------------------------------------------------
// Binary kernels (Tensor rhs, numpy broadcasting)
// ---------------------------------------------------------------------------

macro_rules! bin_op {
    ($name:ident, $op:literal, $f:expr) => {
        pub(crate) fn $name(a: &Tensor, b: &Tensor) -> Result<Tensor, EvalError> {
            broadcast_elementwise(a, b, $op, $f)
        }
    };
}

bin_op!(k_atan2, "tensor.atan2", f64::atan2);
bin_op!(k_copysign, "tensor.copysign", f64::copysign);
bin_op!(k_hypot, "tensor.hypot", f64::hypot);
bin_op!(k_fmod, "tensor.fmod", |a, b| a % b);
bin_op!(k_floor_divide, "tensor.floor_divide", |a, b| (a / b).floor());
bin_op!(k_remainder_tt, "tensor.remainder", |a, b| {
    // Python-style remainder: sign follows the divisor.
    let r = a % b;
    if r != 0.0 && (r < 0.0) != (b < 0.0) {
        r + b
    } else {
        r
    }
});
bin_op!(k_logaddexp, "tensor.logaddexp", |a, b| {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
});
bin_op!(k_logaddexp2, "tensor.logaddexp2", |a, b| {
    // log2(2^a + 2^b) = m + log2(2^(a-m) + 2^(b-m)).
    let m = a.max(b);
    m + ((a - m).exp2() + (b - m).exp2()).log2()
});
bin_op!(k_xlogy_tt, "tensor.xlogy", |a, b| if a == 0.0 { 0.0 } else { a * b.ln() });
bin_op!(k_heaviside, "tensor.heaviside", |a, b| {
    if a > 0.0 {
        1.0
    } else if a < 0.0 {
        0.0
    } else {
        0.5 * b
    }
});
bin_op!(k_nextafter, "tensor.nextafter", |a, b| {
    if a == b || a.is_nan() || b.is_nan() {
        return b;
    }
    // One representable step toward b via the IEEE-754 bit pattern.
    let bits = a.to_bits();
    let nb = if a < b {
        if a >= 0.0 { bits + 1 } else { bits - 1 }
    } else if a >= 0.0 {
        bits - 1
    } else {
        bits + 1
    };
    f64::from_bits(nb)
});

// `maximum`/`minimum` — NaN propagates like torch.
bin_op!(k_maximum, "tensor.maximum", |a, b| {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
});
bin_op!(k_minimum, "tensor.minimum", |a, b| {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
});

// `logical_and`/`or`/`xor` — nonzero is true; result 1.0/0.0.
bin_op!(k_logical_and, "tensor.logical_and", |a, b| {
    ((a != 0.0) && (b != 0.0)) as u8 as f64
});
bin_op!(k_logical_or, "tensor.logical_or", |a, b| {
    ((a != 0.0) || (b != 0.0)) as u8 as f64
});
bin_op!(k_logical_xor, "tensor.logical_xor", |a, b| {
    ((a != 0.0) != (b != 0.0)) as u8 as f64
});

/// `pow` — exponent is `Tensor` or `Scalar` (Value arg 1).
pub(crate) fn k_pow(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let a = require_tensor(args, 0)?;
    match args.get(1) {
        Some(v) if v.is_number() => {
            let e = req_f64(v)?;
            Ok(enc(broadcast_elementwise(&a, &Tensor { shape: vec![], data: vec![e] }, "tensor.pow", |x, p| x.powf(p))?))
        }
        Some(v) => {
            let b = decode_tensor(v)?;
            Ok(enc(broadcast_elementwise(&a, &b, "tensor.pow", |x, p| x.powf(p))?))
        }
        None => Err(op_err("pow", "missing exponent")),
    }
}

/// Scalar-or-tensor rhs binary where the scalar broadcasts: used by
/// `fmod.Scalar`, `floor_divide.Scalar`, `copysign.Scalar`,
/// `remainder.Scalar_Tensor` variants — the generator emits per-op
/// wrappers that pick the right arm.
pub(crate) fn binary_scalar_rhs(
    a: &Tensor,
    rhs: &Value,
    op: &str,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Tensor, EvalError> {
    match rhs {
        v if v.is_number() => {
            let s = req_f64(v)?;
            broadcast_elementwise(a, &Tensor { shape: vec![], data: vec![s] }, op, f)
        }
        v => {
            let b = decode_tensor(v)?;
            broadcast_elementwise(a, &b, op, f)
        }
    }
}

/// Named scalar fns for `drv_bin_ss` (fn pointers can't be closures).
pub(crate) fn f_fmod(a: f64, b: f64) -> f64 {
    a % b
}
pub(crate) fn f_floor_divide(a: f64, b: f64) -> f64 {
    (a / b).floor()
}
pub(crate) fn f_copysign(a: f64, b: f64) -> f64 {
    a.copysign(b)
}
pub(crate) fn f_remainder(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r != 0.0 && (r < 0.0) != (b < 0.0) {
        r + b
    } else {
        r
    }
}

/// `(Tensor self, Scalar|Tensor other)` driver — covers the `.Scalar`
/// and base `Tensor,Tensor` overloads of `fmod`/`floor_divide`/
/// `copysign`.
pub(crate) fn drv_bin_ss(
    args: &[serde_json::Value],
    op: &str,
    f: fn(f64, f64) -> f64,
) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let a = require_tensor(args, 0)?;
    let rhs = arg_at(args, 1)?;
    Ok(enc(binary_scalar_rhs(&a, rhs, op, f)?))
}

/// `clamp(self, min?, max?)` — bounds are `Scalar?` or `Tensor?`.
pub(crate) fn k_clamp(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let a = require_tensor(args, 0)?;
    let lo = args.get(1).filter(|v| !v.is_null());
    let hi = args.get(2).filter(|v| !v.is_null());
    // Decode bounds as Option<Tensor-or-f64> (module-level `BoundVal`).
    let dec = |v: Option<&serde_json::Value>| -> Result<BoundVal, EvalError> {
        match v {
            None => Ok(BoundVal::None),
            Some(x) if x.is_null() => Ok(BoundVal::None),
            Some(x) if x.is_number() => Ok(BoundVal::Num(req_f64(x)?)),
            Some(x) => Ok(BoundVal::Ten(decode_tensor(x)?)),
        }
    };
    let (lo, hi) = (dec(lo)?, dec(hi)?);
    // Materialize: clamp = min(max(x, lo), hi) with broadcast bounds.
    let mut t = a.clone();
    if !matches!(lo, BoundVal::None) {
        t = clamp_side(&t, &lo, true, "clamp")?;
    }
    if !matches!(hi, BoundVal::None) {
        t = clamp_side(&t, &hi, false, "clamp")?;
    }
    Ok(enc(t))
}

fn clamp_side(t: &Tensor, b: &BoundVal, is_lo: bool, op: &str) -> Result<Tensor, EvalError> {
    let f = |x: f64, y: f64| if is_lo { x.max(y) } else { x.min(y) };
    match b {
        BoundVal::Num(s) => {
            broadcast_elementwise(t, &Tensor { shape: vec![], data: vec![*s] }, op, f)
        }
        BoundVal::Ten(tt) => broadcast_elementwise(t, tt, op, f),
        BoundVal::None => Ok(t.clone()),
    }
}

/// `clamp_min`/`clamp_max` — bound is `Scalar` or `Tensor`.
pub(crate) fn k_clamp_minmax(args: &[serde_json::Value], is_min: bool) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let a = require_tensor(args, 0)?;
    let rhs = args.get(1).ok_or_else(|| op_err("clamp", "missing bound"))?;
    let op = if is_min { "clamp_min" } else { "clamp_max" };
    let f = |x: f64, y: f64| if is_min { x.max(y) } else { x.min(y) };
    let t = binary_scalar_rhs(&a, rhs, op, f)?;
    Ok(enc(t))
}

/// `where(condition, self, other)` — condition nonzero selects `self`.
/// Operands may be `Scalar` in the `where.Scalar*` overloads.
pub(crate) fn k_where(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    if args.len() == 1 {
        // where(condition) → indices of nonzero elements, [nnz, rank].
        let c = require_tensor(args, 0)?;
        return Ok(enc(nonzero_impl(&c)?));
    }
    let cond = require_tensor(args, 0)?;
    let dec = |v: Option<&serde_json::Value>| -> Result<BoundVal, EvalError> {
        match v {
            Some(x) if x.is_number() => Ok(BoundVal::Num(req_f64(x)?)),
            Some(x) => Ok(BoundVal::Ten(decode_tensor(x)?)),
            None => Err(op_err("where", "missing operand")),
        }
    };
    let x = dec(args.get(1))?;
    let y = dec(args.get(2))?;
    // Broadcast cond with x and y; evaluate per element.
    let shape = broadcast_shape3(&cond, &x, &y)?;
    let cond_t = broadcast_to(&cond, &shape, "where")?;
    let xf = broadcast_to(&to_tensor_at(&x, &shape)?, &shape, "where")?;
    let yf = broadcast_to(&to_tensor_at(&y, &shape)?, &shape, "where")?;
    let data: Vec<f64> = cond_t
        .data
        .iter()
        .zip(&xf.data)
        .zip(&yf.data)
        .map(|((&c, &a), &b)| if c != 0.0 { a } else { b })
        .collect();
    Ok(enc(Tensor {
        shape: cond_t.shape,
        data,
    }))
}

enum BoundVal {
    None,
    Num(f64),
    Ten(Tensor),
}

fn to_tensor_at(b: &BoundVal, shape: &[usize]) -> Result<Tensor, EvalError> {
    match b {
        BoundVal::Num(s) => Ok(Tensor { shape: shape.to_vec(), data: vec![*s; shape.iter().product()] }),
        BoundVal::Ten(t) => Ok(t.clone()),
        BoundVal::None => Err(op_err("where", "missing operand")),
    }
}

fn bound_shape(b: &BoundVal) -> Vec<usize> {
    match b {
        BoundVal::Num(_) | BoundVal::None => vec![],
        BoundVal::Ten(t) => t.shape.clone(),
    }
}

fn broadcast_shape3(cond: &Tensor, x: &BoundVal, y: &BoundVal) -> Result<Vec<usize>, EvalError> {
    // Compute broadcast of all involved shapes.
    let s1 = broadcast_shape_pub(&cond.shape, &bound_shape(x), "where")?;
    broadcast_shape_pub(&s1, &bound_shape(y), "where")
}

pub(crate) fn broadcast_shape_pub(a: &[usize], b: &[usize], op: &str) -> Result<Vec<usize>, EvalError> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for (i, o) in out.iter_mut().enumerate() {
        let da = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let db = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        *o = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return Err(op_err(op, format!("shapes {a:?} and {b:?} not broadcast-compatible"))),
        };
    }
    Ok(out)
}

/// `eq`/`ne`/`lt`/`le`/`gt`/`ge` — rhs `Tensor` or `Scalar`.
pub(crate) fn k_compare(args: &[serde_json::Value], kind: CmpKind) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let a = require_tensor(args, 0)?;
    let rhs = args.get(1).ok_or_else(|| op_err("compare", "missing rhs"))?;
    let t = binary_scalar_rhs(&a, rhs, kind.name(), |x, y| {
        let r = match kind {
            CmpKind::Eq => x == y,
            CmpKind::Ne => x != y,
            CmpKind::Lt => x < y,
            CmpKind::Le => x <= y,
            CmpKind::Gt => x > y,
            CmpKind::Ge => x >= y,
        };
        if r { 1.0 } else { 0.0 }
    })?;
    Ok(enc(t))
}

#[derive(Clone, Copy)]
pub(crate) enum CmpKind {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}
impl CmpKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            CmpKind::Eq => "eq",
            CmpKind::Ne => "ne",
            CmpKind::Lt => "lt",
            CmpKind::Le => "le",
            CmpKind::Gt => "gt",
            CmpKind::Ge => "ge",
        }
    }
}

/// `xlogy` — either side may be `Scalar`.
pub(crate) fn k_xlogy(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    // Tensor,Tensor / Tensor,Scalar / Scalar,Tensor.
    let a_is_num = args.first().is_some_and(|v| v.is_number());
    let (a, b) = if a_is_num {
        let s = req_f64(&args[0])?;
        let bt = decode_tensor(&args[1])?;
        (Tensor { shape: vec![], data: vec![s] }, bt)
    } else {
        let at = require_tensor(args, 0)?;
        match args.get(1) {
            Some(v) if v.is_number() => {
                let s = req_f64(v)?;
                (at, Tensor { shape: vec![], data: vec![s] })
            }
            Some(v) => (at, decode_tensor(v)?),
            None => return Err(op_err("xlogy", "missing rhs")),
        }
    };
    Ok(enc(broadcast_elementwise(&a, &b, "tensor.xlogy", |x, y| {
        if x == 0.0 { 0.0 } else { x * y.ln() }
    })?))
}

/// `remainder` — `Tensor,Tensor` or `Scalar,Tensor`.
pub(crate) fn k_remainder(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let f = |a: f64, b: f64| {
        let r = a % b;
        if r != 0.0 && (r < 0.0) != (b < 0.0) { r + b } else { r }
    };
    if args.first().is_some_and(|v| v.is_number()) {
        let s = req_f64(&args[0])?;
        let b = require_tensor(args, 1)?;
        return Ok(enc(broadcast_elementwise(
            &Tensor { shape: vec![], data: vec![s] },
            &b,
            "tensor.remainder",
            f,
        )?));
    }
    let a = require_tensor(args, 0)?;
    let t = binary_scalar_rhs(&a, &args[1], "tensor.remainder", f)?;
    Ok(enc(t))
}

/// `lerp(self, end, weight)` — weight `Tensor` or `Scalar`.
pub(crate) fn k_lerp(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let a = require_tensor(args, 0)?;
    let end = require_tensor(args, 1)?;
    let w = args.get(2).ok_or_else(|| op_err("lerp", "missing weight"))?;
    // lerp = a + w*(end - a); broadcast all three.
    let diff = broadcast_elementwise(&end, &a, "tensor.lerp", |e, x| e - x)?;
    let scaled = match w {
        v if v.is_number() => {
            let s = req_f64(v)?;
            broadcast_elementwise(&diff, &Tensor { shape: vec![], data: vec![s] }, "tensor.lerp", |d, w| d * w)?
        }
        v => {
            let wt = decode_tensor(v)?;
            broadcast_elementwise(&diff, &wt, "tensor.lerp", |d, w| d * w)?
        }
    };
    // Broadcast `a` to the result shape and add.
    let shape = broadcast_shape_pub(&a.shape, &scaled.shape, "tensor.lerp")?;
    let af = broadcast_to(&a, &shape, "tensor.lerp")?;
    let sf = broadcast_to(&scaled, &shape, "tensor.lerp")?;
    Ok(enc(Tensor {
        shape,
        data: af.data.iter().zip(&sf.data).map(|(&x, &y)| x + y).collect(),
    }))
}

/// `prelu(self, weight)` — x>0 ? x : w*x, w broadcast.
pub(crate) fn k_prelu(a: &Tensor, w: &Tensor) -> Result<Tensor, EvalError> {
    let wx = broadcast_elementwise(w, a, "tensor.prelu", |wv, xv| wv * xv)?;
    let shape = broadcast_shape_pub(&a.shape, &wx.shape, "tensor.prelu")?;
    let af = broadcast_to(a, &shape, "tensor.prelu")?;
    let wf = broadcast_to(&wx, &shape, "tensor.prelu")?;
    Ok(Tensor {
        shape,
        data: af
            .data
            .iter()
            .zip(&wf.data)
            .map(|(&x, &y)| if x > 0.0 { x } else { y })
            .collect(),
    })
}

// ---------------------------------------------------------------------------
// Reductions with custom arg shapes
// ---------------------------------------------------------------------------

/// `max`/`min` — overloaded: `(self)` all-reduce, `(self, dim, keepdim)`
/// → `{values, indices}`, `(self, other)` elementwise.
pub(crate) fn k_max_min(args: &[serde_json::Value], want_max: bool) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let op = if want_max { "max" } else { "min" };
    let a = require_tensor(args, 0)?;
    match args.get(1) {
        // No second arg, or a tensor rhs → elementwise maximum/minimum.
        None => {
            // All-reduce → rank-0 tensor.
            let init = if want_max { f64::NEG_INFINITY } else { f64::INFINITY };
            let f = if want_max { f64::max } else { f64::min };
            let t = reduce_fold(&a, None, false, init, f, op)?;
            Ok(enc(t))
        }
        Some(v) if v.is_object() => {
            // (self, other) elementwise.
            let b = decode_tensor(v)?;
            let f = if want_max {
                |x: f64, y: f64| {
                    if x.is_nan() || y.is_nan() {
                        f64::NAN
                    } else {
                        x.max(y)
                    }
                }
            } else {
                |x: f64, y: f64| {
                    if x.is_nan() || y.is_nan() {
                        f64::NAN
                    } else {
                        x.min(y)
                    }
                }
            };
            Ok(enc(broadcast_elementwise(&a, &b, &format!("tensor.{op}"), f)?))
        }
        Some(v) => {
            // (self, dim, keepdim?) → {values, indices}. `dim` is an int.
            let dim = req_i64(v)?;
            let keepdim = opt_bool(args.get(2), false)?;
            let vals = reduce_arg(&a, Some(dim), keepdim, want_max, op)?;
            // values = amax/amin along dim.
            let values = if want_max {
                reduce_amax(&a, Some(vec![dim]), keepdim)?
            } else {
                reduce_amin(&a, Some(vec![dim]), keepdim)?
            };
            Ok(json!({
                "values": enc(values),
                "indices": enc(vals),
            }))
        }
    }
}

/// `all`/`any` — `(self)`, `(self, dim)`, `(self, dim, keepdim)` where
/// dim is int or int[].
pub(crate) fn k_all_any(args: &[serde_json::Value], want_all: bool) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let op = if want_all { "all" } else { "any" };
    let a = require_tensor(args, 0)?;
    let pred = |x: f64| x != 0.0;
    let combine = |acc: f64, x: f64| {
        let px = pred(x);
        let r = if want_all { (acc != 0.0) && px } else { (acc != 0.0) || px };
        if r { 1.0 } else { 0.0 }
    };
    // Determine dims arg.
    let dims: Option<Vec<i64>> = match args.get(1) {
        None => None,
        Some(v) if v.is_null() => None,
        Some(v) if v.is_i64() || v.is_u64() || (v.is_f64() && v.as_f64().is_some_and(|f| f.fract() == 0.0)) => {
            Some(vec![req_i64(v)?])
        }
        Some(v) => Some(req_i64s(v)?),
    };
    let keepdim = opt_bool(args.get(2), false)?;
    let t = reduce_fold(&a, dims.as_deref(), keepdim, if want_all { 1.0 } else { 0.0 }, combine, op)?;
    Ok(enc(t))
}

/// `var`/`std` — `(self)`, `(self, unbiased)`, `(self, dim…, unbiased?,
/// keepdim?)`. `correction` kwarg unsupported (documented seam).
pub(crate) fn k_var_std(args: &[serde_json::Value], want_std: bool) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let op = if want_std { "std" } else { "var" };
    let a = require_tensor(args, 0)?;
    // Arg1 disambiguates: bool → unbiased (all-reduce); int/list → dims.
    let (dims, unbiased, keepdim) = match args.get(1) {
        // `None`/null dim → all dims, but unbiased/keepdim still come
        // from their own positions (ATen `dim=None, unbiased, keepdim`).
        None => (
            None,
            opt_bool(args.get(2), true)?,
            opt_bool(args.get(3), false)?,
        ),
        Some(v) if v.is_null() => (
            None,
            opt_bool(args.get(2), true)?,
            opt_bool(args.get(3), false)?,
        ),
        Some(v) if v.is_boolean() => (None, req_bool(v)?, opt_bool(args.get(2), false)?),
        Some(v) => {
            let d = if v.is_array() { req_i64s(v)? } else { vec![req_i64(v)?] };
            (
                Some(d),
                opt_bool(args.get(2), true)?,
                opt_bool(args.get(3), false)?,
            )
        }
    };
    // Two-pass: mean then mean squared deviation.
    let mean = reduce_mean(&a, dims.clone(), true)?;
    let full = broadcast_to(&mean, &a.shape, op)?;
    let sq = Tensor {
        shape: a.shape.clone(),
        data: a.data.iter().zip(&full.data).map(|(&x, &m)| (x - m).powi(2)).collect(),
    };
    let var = reduce_mean(&sq, dims.clone(), keepdim)?;
    // Apply Bessel correction: n/(n-c) where c = 1 if unbiased.
    let Some(red) = norm_reduce_dims(&a, dims.as_deref(), op)? else {
        // Identity dims (empty list) — var of nothing is the tensor
        // itself per torch's empty-tuple rule; return scalar-of-same.
        let n = a.data.len() as f64;
        let c = if unbiased { 1.0 } else { 0.0 };
        let denom = (n - c).max(f64::MIN_POSITIVE);
        let v = var.data[0] * n / denom;
        let out = if want_std { v.sqrt() } else { v };
        return Ok(enc(Tensor { shape: vec![], data: vec![out] }));
    };
    let n: usize = red.iter().map(|&d| a.shape[d]).product();
    let c = if unbiased { 1.0 } else { 0.0 };
    let denom = (n as f64 - c).max(f64::MIN_POSITIVE);
    let data: Vec<f64> = var
        .data
        .iter()
        .map(|&v| {
            let v = v * n as f64 / denom;
            if want_std { v.sqrt() } else { v }
        })
        .collect();
    Ok(enc(Tensor { shape: var.shape, data }))
}

/// `squeeze` — `(self)`, `(self, int dim)`, `(self, int[] dim)`.
pub(crate) fn k_squeeze(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let a = require_tensor(args, 0)?;
    let rank = a.shape.len();
    let dims: Option<Vec<i64>> = match args.get(1) {
        None => None,
        Some(v) if v.is_null() => None,
        Some(v) if v.is_array() => Some(req_i64s(v)?),
        Some(v) => Some(vec![req_i64(v)?]),
    };
    let mut shape = a.shape.clone();
    match dims {
        None => {
            shape.retain(|&d| d != 1);
        }
        Some(ds) => {
            let mut drop = vec![false; rank];
            for &d in &ds {
                let w = wrap_dim(d, rank, "squeeze")?;
                if a.shape[w] != 1 {
                    return Err(op_err("squeeze", format!("dim {d} has size {} != 1", a.shape[w])));
                }
                drop[w] = true;
            }
            shape = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !drop[*i])
                .map(|(_, &s)| s)
                .collect();
        }
    }
    Ok(enc(Tensor { shape, data: a.data.clone() }))
}

/// `split` — `(self, int[] sizes, dim?)` or `(self, int size, dim?)`.
pub(crate) fn k_split(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let a = require_tensor(args, 0)?;
    let spec = args.get(1).ok_or_else(|| op_err("split", "missing split spec"))?;
    let dim = opt_i64(args.get(2), 0)?;
    let d = wrap_dim(dim, a.shape.len(), "split")?;
    let width = a.shape[d];
    let sizes: Vec<usize> = if spec.is_array() {
        req_i64s(spec)?.iter().map(|&n| usize::try_from(n).map_err(|_| op_err("split", "negative size"))).collect::<Result<_, _>>()?
    } else {
        let n = usize::try_from(req_i64(spec)?).map_err(|_| op_err("split", "negative size"))?;
        if n == 0 || width % n != 0 {
            return Err(op_err("split", format!("cannot split extent {width} into {n}")));
        }
        vec![width / n; n]
    };
    let total: usize = sizes.iter().sum();
    if total != width {
        return Err(op_err("split", format!("sizes sum {total} != extent {width}")));
    }
    // Slice along d.
    let inner: usize = a.shape[d + 1..].iter().product();
    let outer: usize = a.shape[..d].iter().product();
    let mut out = Vec::with_capacity(sizes.len());
    let mut start = 0usize;
    for &sz in &sizes {
        let mut shape = a.shape.clone();
        shape[d] = sz;
        let mut data = Vec::with_capacity(sz * inner * outer);
        for o in 0..outer {
            let src0 = (o * width + start) * inner;
            data.extend_from_slice(&a.data[src0..src0 + sz * inner]);
        }
        out.push(enc(Tensor { shape, data }));
        start += sz;
    }
    Ok(Value::Array(out))
}

/// `chunk` — `(self, int chunks, dim?)` → array of tensors.
pub(crate) fn k_chunk(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let a = require_tensor(args, 0)?;
    let chunks = usize::try_from(req_i64(&args[1])?).map_err(|_| op_err("chunk", "bad chunk count"))?;
    if chunks == 0 {
        return Err(op_err("chunk", "chunks must be positive"));
    }
    let dim = opt_i64(args.get(2), 0)?;
    // torch.chunk: sizes differ by at most 1, larger chunks first.
    let d = wrap_dim(dim, a.shape.len(), "chunk")?;
    let width = a.shape[d];
    let base = width / chunks;
    let rem = width % chunks;
    let sizes: Vec<usize> = (0..chunks).map(|i| base + usize::from(i < rem)).collect();
    // Reuse split machinery by fabricating the sizes arg.
    let sizes_json = json!(sizes.iter().map(|&s| s as i64).collect::<Vec<_>>());
    let fake = [args[0].clone(), sizes_json, args.get(2).cloned().unwrap_or(Value::Null)];
    k_split(&fake)
}

/// `stack` — `(Tensor[], dim?)` → tensor.
pub(crate) fn k_stack(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    let ts = req_tensors(&args[0])?;
    if ts.is_empty() {
        return Err(op_err("stack", "empty tensor list"));
    }
    let dim = opt_i64(args.get(1), 0)?;
    let rank = ts[0].shape.len();
    for t in &ts {
        if t.shape != ts[0].shape {
            return Err(op_err("stack", "all tensors must have the same shape"));
        }
    }
    let d = wrap_dim(dim, rank + 1, "stack")?;
    // Unsqueeze each at d then concat.
    let mut unsq = Vec::with_capacity(ts.len());
    for t in &ts {
        let mut shape = t.shape.clone();
        shape.insert(d, 1);
        unsq.push(Tensor { shape, data: t.data.clone() });
    }
    // concat along d.
    let out = concat_tensors(&unsq, d, "stack")?;
    Ok(enc(out))
}

fn concat_tensors(ts: &[Tensor], axis: usize, op: &str) -> Result<Tensor, EvalError> {
    let rank = ts[0].shape.len();
    if axis >= rank {
        return Err(op_err(op, format!("axis {axis} out of range for rank {rank}")));
    }
    for t in ts {
        if t.shape.len() != rank {
            return Err(op_err(op, "rank mismatch"));
        }
        for (d, (&a, &b)) in t.shape.iter().zip(ts[0].shape.iter()).enumerate() {
            if d != axis && a != b {
                return Err(op_err(op, format!("non-axis dim {d} mismatch")));
            }
        }
    }
    let outer: usize = ts[0].shape[..axis].iter().product();
    let inner: usize = ts[0].shape[axis + 1..].iter().product();
    let mut out_shape = ts[0].shape.clone();
    out_shape[axis] = ts.iter().map(|t| t.shape[axis]).sum();
    let mut data = Vec::with_capacity(out_shape.iter().product());
    for o in 0..outer {
        for t in ts {
            let slab = t.shape[axis] * inner;
            data.extend_from_slice(&t.data[o * slab..(o + 1) * slab]);
        }
    }
    Ok(Tensor { shape: out_shape, data })
}

/// `unbind` — `(self, dim?)` → array of rank-1-less tensors.
pub(crate) fn k_unbind(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let a = require_tensor(args, 0)?;
    let dim = opt_i64(args.get(1), 0)?;
    let d = wrap_dim(dim, a.shape.len(), "unbind")?;
    let width = a.shape[d];
    let inner: usize = a.shape[d + 1..].iter().product();
    let outer: usize = a.shape[..d].iter().product();
    let mut out_shape = a.shape.clone();
    out_shape.remove(d);
    let mut out = Vec::with_capacity(width);
    for i in 0..width {
        let mut data = Vec::with_capacity(a.data.len() / width.max(1));
        for o in 0..outer {
            let src0 = (o * width + i) * inner;
            data.extend_from_slice(&a.data[src0..src0 + inner]);
        }
        out.push(enc(Tensor { shape: out_shape.clone(), data }));
    }
    Ok(Value::Array(out))
}

/// `sort` — `(self, dim?, descending?)` → `{values, indices}`.
pub(crate) fn k_sort(args: &[serde_json::Value], argsort_only: bool) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let op = if argsort_only { "argsort" } else { "sort" };
    let a = require_tensor(args, 0)?;
    let dim = opt_i64(args.get(1), -1)?;
    let desc = opt_bool(args.get(2), false)?;
    let (values, indices) = sort_dim(&a, dim, desc, op)?;
    if argsort_only {
        Ok(enc(indices))
    } else {
        Ok(json!({ "values": enc(values), "indices": enc(indices) }))
    }
}

fn sort_dim(t: &Tensor, dim: i64, desc: bool, op: &str) -> Result<(Tensor, Tensor), EvalError> {
    let d = wrap_dim(dim, t.shape.len(), op)?;
    let rank = t.shape.len();
    let width = t.shape[d];
    let tail: usize = t.shape[d + 1..].iter().product();
    let outer: usize = t.shape[..d].iter().product();
    let mut values = vec![0.0; t.data.len()];
    let mut indices = vec![0.0; t.data.len()];
    for o in 0..outer {
        for j in 0..tail {
            let base = (o * width) * tail + j;
            let mut order: Vec<usize> = (0..width).collect();
            order.sort_by(|&x, &y| {
                let (a, b) = (t.data[base + x * tail], t.data[base + y * tail]);
                let ord = a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal);
                if desc { ord.reverse() } else { ord }
            });
            for (new_i, &old_i) in order.iter().enumerate() {
                values[base + new_i * tail] = t.data[base + old_i * tail];
                indices[base + new_i * tail] = old_i as f64;
            }
        }
    }
    let _ = rank;
    Ok((
        Tensor { shape: t.shape.clone(), data: values },
        Tensor { shape: t.shape.clone(), data: indices },
    ))
}

/// `topk` — `(self, k, dim?, largest?, sorted?)` → `{values, indices}`.
pub(crate) fn k_topk(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let a = require_tensor(args, 0)?;
    let k = usize::try_from(req_i64(&args[1])?).map_err(|_| op_err("topk", "bad k"))?;
    let dim = opt_i64(args.get(2), -1)?;
    let largest = opt_bool(args.get(3), true)?;
    // `sorted` only affects order guarantee; our sort is stable anyway.
    let _sorted = opt_bool(args.get(4), true)?;
    let (mut values, mut indices) = sort_dim(&a, dim, largest, "topk")?;
    // Truncate each slice along dim to k.
    let d = wrap_dim(dim, a.shape.len(), "topk")?;
    let width = a.shape[d];
    if k > width {
        return Err(op_err("topk", format!("k={k} exceeds dim size {width}")));
    }
    let tail: usize = a.shape[d + 1..].iter().product();
    let outer: usize = a.shape[..d].iter().product();
    let mut v2 = vec![0.0; values.data.len() / width * k];
    let mut i2 = vec![0.0; indices.data.len() / width * k];
    for o in 0..outer {
        for j in 0..tail {
            let src = (o * width) * tail + j;
            let dst = (o * k) * tail + j;
            for i in 0..k {
                v2[dst + i * tail] = values.data[src + i * tail];
                i2[dst + i * tail] = indices.data[src + i * tail];
            }
        }
    }
    let mut shape = a.shape.clone();
    shape[d] = k;
    values = Tensor { shape: shape.clone(), data: v2 };
    indices = Tensor { shape, data: i2 };
    Ok(json!({ "values": enc(values), "indices": enc(indices) }))
}

/// `nonzero`/`argwhere`/`where(cond)` — indices of nonzero elements,
/// shape `[nnz, rank]` (row-major coords, f64).
pub(crate) fn nonzero_impl(t: &Tensor) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (flat, &x) in t.data.iter().enumerate() {
        if x != 0.0 {
            // Decompose flat → coords (row-major).
            let mut rem = flat;
            let mut coords = vec![0f64; rank];
            for d in (0..rank).rev() {
                coords[d] = (rem % t.shape[d]) as f64;
                rem /= t.shape[d];
            }
            rows.push(coords);
        }
    }
    let nnz = rows.len();
    let mut data = Vec::with_capacity(nnz * rank);
    for r in rows {
        data.extend(r);
    }
    Ok(Tensor {
        shape: vec![nnz, rank],
        data,
    })
}

/// `index_select` — gather slices along `dim` at `index` positions.
pub(crate) fn k_index_select(t: &Tensor, dim: i64, idx: &Tensor) -> Result<Tensor, EvalError> {
    let d = wrap_dim(dim, t.shape.len(), "index_select")?;
    if idx.shape.len() != 1 {
        return Err(op_err("index_select", "index must be rank-1"));
    }
    let n = idx.data.len();
    let width = t.shape[d];
    for &i in &idx.data {
        let ii = i as i64;
        if ii < 0 || ii as usize >= width {
            return Err(op_err("index_select", format!("index {ii} out of range [0,{width})")));
        }
    }
    let strides = row_major_strides(&t.shape);
    let inner: usize = t.shape[d + 1..].iter().product();
    let outer: usize = t.shape[..d].iter().product();
    let mut out_shape = t.shape.clone();
    out_shape[d] = n;
    let mut data = Vec::with_capacity(out_shape.iter().product());
    for o in 0..outer {
        for &old_f in idx.data.iter() {
            let old_i = old_f as usize;
            let src0 = (o * width + old_i) * inner;
            data.extend_from_slice(&t.data[src0..src0 + inner]);
        }
    }
    let _ = strides;
    Ok(Tensor { shape: out_shape, data })
}

/// `gather` — `out[.., i, ..] = self[.., index[.., i, ..], ..]` along dim.
pub(crate) fn k_gather(t: &Tensor, dim: i64, idx: &Tensor) -> Result<Tensor, EvalError> {
    let d = wrap_dim(dim, t.shape.len(), "gather")?;
    if idx.shape != t.shape {
        return Err(op_err("gather", format!("index shape {:?} != self {:?}", idx.shape, t.shape)));
    }
    let rank = t.shape.len();
    let width = t.shape[d];
    let strides = row_major_strides(&t.shape);
    let istrides = row_major_strides(&idx.shape);
    let count = t.data.len();
    let mut data = vec![0.0; count];
    let mut cidx = vec![0usize; rank];
    for _ in 0..count {
        let gi = idx.data[cidx.iter().zip(&istrides).map(|(&i, &s)| i * s).sum::<usize>()];
        let g = gi as usize;
        if g >= width {
            return Err(op_err("gather", format!("index {g} out of range [0,{width})")));
        }
        let mut src = vec![0usize; rank];
        src.copy_from_slice(&cidx);
        src[d] = g;
        data[cidx.iter().zip(&strides).map(|(&i, &s)| i * s).sum::<usize>()] =
            t.data[src.iter().zip(&strides).map(|(&i, &s)| i * s).sum::<usize>()];
        for dd in (0..rank).rev() {
            cidx[dd] += 1;
            if cidx[dd] < t.shape[dd] {
                break;
            }
            cidx[dd] = 0;
        }
    }
    Ok(Tensor { shape: t.shape.clone(), data })
}

/// `index_add` — `out.index_add_(dim, index, source)` out-of-place,
/// `alpha` scaling (kwarg, default 1).
pub(crate) fn k_index_add(t: &Tensor, dim: i64, idx: &Tensor, src: &Tensor, alpha: f64) -> Result<Tensor, EvalError> {
    let d = wrap_dim(dim, t.shape.len(), "index_add")?;
    let rank = t.shape.len();
    if idx.shape.len() != 1 || src.shape.len() != rank {
        return Err(op_err("index_add", "index must be rank-1, source same rank as self"));
    }
    if src.shape[d] != idx.data.len() {
        return Err(op_err("index_add", "source extent along dim must equal index length"));
    }
    for (k, (&s, &o)) in src.shape.iter().zip(&t.shape).enumerate() {
        if k != d && s != o {
            return Err(op_err("index_add", format!("source shape {src:?} incompatible with {t:?}")));
        }
    }
    let width = t.shape[d];
    for &i in &idx.data {
        let ii = i as i64;
        if ii < 0 || ii as usize >= width {
            return Err(op_err("index_add", format!("index {ii} out of range")));
        }
    }
    let inner: usize = t.shape[d + 1..].iter().product();
    let outer: usize = t.shape[..d].iter().product();
    let mut data = t.data.clone();
    for o in 0..outer {
        for (new_i, &old_f) in idx.data.iter().enumerate() {
            let old_i = old_f as usize;
            let dst0 = (o * width + old_i) * inner;
            let src0 = (o * idx.data.len() + new_i) * inner;
            for j in 0..inner {
                data[dst0 + j] += alpha * src.data[src0 + j];
            }
        }
    }
    Ok(Tensor { shape: t.shape.clone(), data })
}

/// `take` — flatten `self`, gather at flat `index`.
pub(crate) fn k_take(t: &Tensor, idx: &Tensor) -> Result<Tensor, EvalError> {
    let n = t.data.len();
    let mut data = Vec::with_capacity(idx.data.len());
    for &i in &idx.data {
        let ii = i as i64;
        let j = if ii < 0 { ii + n as i64 } else { ii };
        if j < 0 || j as usize >= n {
            return Err(op_err("take", format!("index {ii} out of range")));
        }
        data.push(t.data[j as usize]);
    }
    Ok(Tensor { shape: idx.shape.clone(), data })
}

/// `index_fill` — `index_fill(dim, index, value)` out-of-place.
pub(crate) fn k_index_fill(t: &Tensor, dim: i64, idx: &Tensor, value: f64) -> Result<Tensor, EvalError> {
    let d = wrap_dim(dim, t.shape.len(), "index_fill")?;
    let mut out = t.clone();
    let inner: usize = t.shape[d + 1..].iter().product();
    let outer: usize = t.shape[..d].iter().product();
    let width = t.shape[d];
    for &i in &idx.data {
        let ii = i as i64;
        if ii < 0 || ii as usize >= width {
            return Err(op_err("index_fill", format!("index {ii} out of range")));
        }
        for o in 0..outer {
            let base = (o * width + ii as usize) * inner;
            for j in 0..inner {
                out.data[base + j] = value;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Shape ops
// ---------------------------------------------------------------------------

pub(crate) fn k_reshape(t: &Tensor, shape: &[i64]) -> Result<Tensor, EvalError> {
    let shape: Vec<usize> = shape
        .iter()
        .map(|&n| usize::try_from(n).map_err(|_| op_err("reshape", "negative extent")))
        .collect::<Result<_, _>>()?;
    reshape_like(t, &shape, "reshape")
}

pub(crate) fn k_flatten(t: &Tensor, start: i64, end: i64) -> Result<Tensor, EvalError> {
    let rank = t.shape.len() as i64;
    let s = start.rem_euclid(rank);
    let e = end.rem_euclid(rank);
    if s > e {
        return Err(op_err("flatten", format!("start_dim {start} > end_dim {end}")));
    }
    let mut shape = t.shape.clone();
    let merged: usize = shape[s as usize..=e as usize].iter().product();
    shape.splice(s as usize..=e as usize, [merged]);
    Ok(Tensor { shape, data: t.data.clone() })
}

pub(crate) fn k_permute(t: &Tensor, dims: &[i64]) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    if dims.len() != rank {
        return Err(op_err("permute", format!("expected {rank} dims, got {}", dims.len())));
    }
    let perm: Vec<usize> = dims
        .iter()
        .map(|&d| wrap_dim(d, rank, "permute"))
        .collect::<Result<_, _>>()?;
    permute_dims(t, &perm, "permute")
}

pub(crate) fn k_expand(t: &Tensor, size: &[i64]) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    if size.len() < rank {
        return Err(op_err("expand", "size has fewer dims than tensor"));
    }
    let off = size.len() - rank;
    // New dims: -1 keeps the input extent (torch expand), 1 stretches,
    // equal ok; anything else is an error.
    let mut out_shape = vec![0usize; size.len()];
    for (i, &n) in size.iter().enumerate() {
        let td = if i < off { 1 } else { t.shape[i - off] };
        out_shape[i] = if n == -1 {
            td
        } else {
            let n = usize::try_from(n).map_err(|_| op_err("expand", "negative extent"))?;
            match (td, n) {
                (1, n) => n,
                (d, n) if d == n => n,
                (d, n) => {
                    return Err(op_err("expand", format!("dim {i}: cannot expand {d} to {n}")))
                }
            }
        };
    }
    // A kept (-1) dim that was 1 stays 1 — torch then requires it to
    // match; nothing to do since td==out already.
    // Materialize via broadcast strides.
    let mut strides = vec![0usize; size.len()];
    let inner = row_major_strides(&t.shape);
    for (i, &s) in inner.iter().enumerate() {
        let d = off + i;
        strides[d] = if t.shape[i] == 1 { 0 } else { s };
    }
    let count: usize = out_shape.iter().product();
    let mut data = Vec::with_capacity(count);
    let mut idx = vec![0usize; size.len()];
    for _ in 0..count {
        let off2: usize = idx.iter().zip(&strides).map(|(&i, &s)| i * s).sum();
        data.push(t.data[off2]);
        for d in (0..size.len()).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(Tensor { shape: out_shape, data })
}

pub(crate) fn k_expand_as(t: &Tensor, other: &Tensor) -> Result<Tensor, EvalError> {
    k_expand(t, &other.shape.iter().map(|&d| d as i64).collect::<Vec<_>>())
}

pub(crate) fn k_narrow(t: &Tensor, dim: i64, start: i64, len: i64) -> Result<Tensor, EvalError> {
    let d = wrap_dim(dim, t.shape.len(), "narrow")?;
    let width = t.shape[d] as i64;
    let s = if start < 0 { start + width } else { start };
    if s < 0 || s + len > width || len < 0 {
        return Err(op_err("narrow", format!("start {start} len {len} out of bounds for dim size {width}")));
    }
    let rank = t.shape.len();
    let inner: usize = t.shape[d + 1..].iter().product();
    let outer: usize = t.shape[..d].iter().product();
    let mut out_shape = t.shape.clone();
    out_shape[d] = len as usize;
    let mut data = Vec::with_capacity(out_shape.iter().product());
    for o in 0..outer {
        let src0 = (o * width as usize + s as usize) * inner;
        data.extend_from_slice(&t.data[src0..src0 + len as usize * inner]);
    }
    let _ = rank;
    Ok(Tensor { shape: out_shape, data })
}

/// `slice(self, dim=0, start=None, end=None, step=1)` — `None` bounds
/// arrive as `None` from the generated wrapper.
pub(crate) fn k_slice_opt(
    t: &Tensor,
    dim: i64,
    start: Option<i64>,
    end: Option<i64>,
    step: i64,
) -> Result<Tensor, EvalError> {
    if step == 0 {
        return Err(op_err("slice", "step must be nonzero"));
    }
    let d = wrap_dim(dim, t.shape.len(), "slice")?;
    let width = t.shape[d] as i64;
    let (s, e) = if step > 0 {
        (
            start.unwrap_or(0).rem_euclid(width),
            end.unwrap_or(width).min(width),
        )
    } else {
        (
            start.unwrap_or(width - 1),
            end.unwrap_or(-width - 1).max(-width - 1),
        )
    };
    // Collect indices along d.
    let mut idxs = Vec::new();
    if step > 0 {
        let mut i = s;
        while i < e {
            idxs.push(i);
            i += step;
        }
    } else {
        let mut i = s;
        while i > e {
            idxs.push(i.rem_euclid(width));
            i += step;
        }
    }
    let rank = t.shape.len();
    let inner: usize = t.shape[d + 1..].iter().product();
    let outer: usize = t.shape[..d].iter().product();
    let mut out_shape = t.shape.clone();
    out_shape[d] = idxs.len();
    let mut data = Vec::with_capacity(out_shape.iter().product());
    for o in 0..outer {
        for &i in &idxs {
            let src0 = (o * width as usize + i.rem_euclid(width) as usize) * inner;
            data.extend_from_slice(&t.data[src0..src0 + inner]);
        }
    }
    let _ = rank;
    Ok(Tensor { shape: out_shape, data })
}

pub(crate) fn k_repeat(t: &Tensor, repeats: &[i64]) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    if repeats.len() != rank {
        return Err(op_err("repeat", format!("expected {rank} repeats, got {}", repeats.len())));
    }
    // Tile: out[i0*r0+j0, ...] = t[j0, ...]. For each output coord the
    // input coord is coord % t.shape[d].
    let mut out_shape = vec![0usize; rank];
    for d in 0..rank {
        out_shape[d] = t.shape[d]
            * usize::try_from(repeats[d]).map_err(|_| op_err("repeat", "negative repeat"))?;
    }
    let in_strides = row_major_strides(&t.shape);
    let count: usize = out_shape.iter().product();
    let mut data = Vec::with_capacity(count);
    let mut idx = vec![0usize; rank];
    for _ in 0..count {
        let mut off = 0usize;
        for d in 0..rank {
            off += (idx[d] % t.shape[d]) * in_strides[d];
        }
        data.push(t.data[off]);
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(Tensor { shape: out_shape, data })
}

/// `flip` — reverse along each dim in `dims`.
pub(crate) fn k_flip(t: &Tensor, dims: &[i64]) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    let mut flip = vec![false; rank];
    for &d in dims {
        flip[wrap_dim(d, rank, "flip")?] = true;
    }
    let in_strides = row_major_strides(&t.shape);
    let count = t.data.len();
    let mut data = Vec::with_capacity(count);
    let mut idx = vec![0usize; rank];
    for _ in 0..count {
        let off: usize = idx
            .iter()
            .enumerate()
            .map(|(d, &i)| {
                let i = if flip[d] { t.shape[d] - 1 - i } else { i };
                i * in_strides[d]
            })
            .sum();
        data.push(t.data[off]);
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < t.shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(Tensor { shape: t.shape.clone(), data })
}

/// `triu`/`tril` — zero above/below `diagonal` offset.
pub(crate) fn k_triu_tril(t: &Tensor, diagonal: i64, upper: bool) -> Result<Tensor, EvalError> {
    if t.shape.len() != 2 {
        return Err(op_err("triu", "expected rank-2 tensor"));
    }
    let (m, n) = (t.shape[0] as i64, t.shape[1] as i64);
    let mut data = t.data.clone();
    for i in 0..m {
        for j in 0..n {
            let keep = if upper { j - i >= diagonal } else { j - i <= diagonal };
            if !keep {
                data[(i * n + j) as usize] = 0.0;
            }
        }
    }
    Ok(Tensor { shape: t.shape.clone(), data })
}

/// `diag` — extract (`diagonal=0`, 2-D) or embed (1-D → 2-D) diagonal.
pub(crate) fn k_diag(t: &Tensor, diagonal: i64) -> Result<Tensor, EvalError> {
    if t.shape.len() == 1 {
        // Embed: n → n×n with `diagonal` offset.
        let n = t.shape[0] as i64;
        let size = n + diagonal.abs();
        let mut out = Tensor { shape: vec![size as usize, size as usize], data: vec![0.0; (size * size) as usize] };
        for (i, &v) in t.data.iter().enumerate() {
            let r = i as i64 + diagonal.max(0);
            let c = i as i64 + (-diagonal).max(0);
            if r >= 0 && c >= 0 && r < size && c < size {
                out.data[(r * size + c) as usize] = v;
            }
        }
        Ok(out)
    } else if t.shape.len() == 2 {
        // Extract the `diagonal`-th diagonal.
        let (m, n) = (t.shape[0] as i64, t.shape[1] as i64);
        let start_r = diagonal.max(0);
        let start_c = (-diagonal).max(0);
        let len = (m - start_r).min(n - start_c);
        if len <= 0 {
            return Ok(Tensor { shape: vec![0], data: vec![] });
        }
        let mut data = Vec::with_capacity(len as usize);
        for i in 0..len {
            data.push(t.data[((start_r + i) * n + start_c + i) as usize]);
        }
        Ok(Tensor { shape: vec![len as usize], data })
    } else {
        Err(op_err("diag", "expected rank-1 or rank-2 tensor"))
    }
}

/// `diagonal` — `(self, offset?, dim1?, dim2?)` → moved-to-end diagonal.
pub(crate) fn k_diagonal(t: &Tensor, offset: i64, dim1: i64, dim2: i64) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    if rank < 2 {
        return Err(op_err("diagonal", "expected rank >= 2"));
    }
    let d1 = wrap_dim(dim1, rank, "diagonal")?;
    let d2 = wrap_dim(dim2, rank, "diagonal")?;
    if d1 == d2 {
        return Err(op_err("diagonal", "dim1 and dim2 must differ"));
    }
    // Move d1,d2 to last two axes, extract, keep diag at end.
    let others: Vec<usize> = (0..rank).filter(|&d| d != d1 && d != d2).collect();
    let mut perm = others.clone();
    perm.push(d1);
    perm.push(d2);
    let tp = permute_dims(t, &perm, "diagonal")?;
    let inner_rank = rank - 2;
    let (m, n) = (tp.shape[inner_rank] as i64, tp.shape[inner_rank + 1] as i64);
    let outer: usize = tp.shape[..inner_rank].iter().product();
    let sr = offset.max(0);
    let sc = (-offset).max(0);
    let len = (m - sr).min(n - sc).max(0) as usize;
    let strides = row_major_strides(&tp.shape);
    let mut out_shape = tp.shape[..inner_rank].to_vec();
    out_shape.push(len);
    let mut data = Vec::with_capacity(out_shape.iter().product());
    for o in 0..outer {
        // coords of o in the leading inner_rank axes
        let mut rem = o;
        let mut coords = vec![0usize; inner_rank];
        for d in (0..inner_rank).rev() {
            coords[d] = rem % tp.shape[d];
            rem /= tp.shape[d];
        }
        let mut base = 0usize;
        for d in 0..inner_rank {
            base += coords[d] * strides[d];
        }
        let row_stride = strides[inner_rank];
        let col_stride = strides[inner_rank + 1];
        for i in 0..len {
            let off = base
                + ((sr + i as i64) as usize) * row_stride
                + ((sc + i as i64) as usize) * col_stride;
            data.push(tp.data[off]);
        }
    }
    Ok(Tensor { shape: out_shape, data })
}

/// `trace` — sum of the main diagonal (rank-2 only in torch).
pub(crate) fn k_trace(t: &Tensor) -> Result<Tensor, EvalError> {
    if t.shape.len() != 2 {
        return Err(op_err("trace", "expected rank-2 tensor"));
    }
    let (m, n) = (t.shape[0], t.shape[1]);
    let len = m.min(n);
    let mut s = 0.0;
    for i in 0..len {
        s += t.data[i * n + i];
    }
    Ok(Tensor { shape: vec![], data: vec![s] })
}

/// `t` — transpose last two dims (rank >= 2).
pub(crate) fn k_t(t: &Tensor) -> Result<Tensor, EvalError> {
    if t.shape.len() < 2 {
        return Err(op_err("t", "expected rank >= 2"));
    }
    let r = t.shape.len();
    transpose_dims(t, r - 2, r - 1)
}

/// `mT` — same as `t` for dense tensors.
#[allow(non_snake_case)] // ATen name is `mT`
pub(crate) fn k_mT(t: &Tensor) -> Result<Tensor, EvalError> {
    k_t(t)
}

/// `swapaxes` — alias of `transpose`.
pub(crate) fn k_swapaxes(t: &Tensor, d0: i64, d1: i64) -> Result<Tensor, EvalError> {
    let rank = t.shape.len() as i64;
    let a = d0.rem_euclid(rank);
    let b = d1.rem_euclid(rank);
    transpose_dims(t, a as usize, b as usize)
}

/// `moveaxis` — `(self, source, destination)` int or int[]; the
/// generated wrapper normalizes ints to single-element lists.
pub(crate) fn k_moveaxis(t: &Tensor, src: &[i64], dst: &[i64]) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    if src.len() != dst.len() {
        return Err(op_err("moveaxis", "source/destination length mismatch"));
    }
    // Remove the source axes (in order), then insert each at its
    // destination position — torch normalizes destination against the
    // rank-minus-len axis space.
    let mut src_n: Vec<usize> = src
        .iter()
        .map(|&d| wrap_dim(d, rank, "moveaxis"))
        .collect::<Result<_, _>>()?;
    let mut dst_n: Vec<usize> = dst
        .iter()
        .map(|&d| {
            // Destination range is -(rank-len)..(rank-len-1)+len... torch
            // allows -rank..rank-1 normalized on the reduced space.
            let space = rank - src.len();
            let x = if d < 0 { d + space as i64 } else { d };
            if x < 0 || x >= space as i64 {
                Err(op_err("moveaxis", format!("destination {d} out of range")))
            } else {
                Ok(x as usize)
            }
        })
        .collect::<Result<_, _>>()?;
    src_n.sort_unstable();
    dst_n.sort_unstable();
    // Pair sorted src with sorted dst (torch pairs them positionally
    // after independent normalization — documented approximation).
    let mut order: Vec<usize> = (0..rank).collect();
    let mut moved = Vec::with_capacity(src_n.len());
    for &d in src_n.iter().rev() {
        moved.push(order.remove(d));
    }
    moved.reverse();
    for (m, &pos) in moved.iter().zip(dst_n.iter()) {
        order.insert(pos.min(order.len()), *m);
    }
    permute_dims(t, &order, "moveaxis")
}

/// `rot90` — `(self, k=1, dims=[0,1])`.
pub(crate) fn k_rot90(t: &Tensor, k: i64, dims: &[i64]) -> Result<Tensor, EvalError> {
    if dims.len() != 2 {
        return Err(op_err("rot90", "dims must have exactly 2 elements"));
    }
    let rank = t.shape.len();
    let d0 = wrap_dim(dims[0], rank, "rot90")?;
    let d1 = wrap_dim(dims[1], rank, "rot90")?;
    if d0 == d1 {
        return Err(op_err("rot90", "dims must differ"));
    }
    let kk = k.rem_euclid(4);
    let mut out = t.clone();
    for _ in 0..kk {
        // rot90 once: flip d1, then swap d0/d1.
        out = k_flip(&out, &[dims[1]])?;
        out = transpose_dims(&out, d0, d1)?;
    }
    Ok(out)
}

/// `contiguous`/`clone`/`detach` — identity on our always-contiguous
/// dense representation.
pub(crate) fn k_identity(t: &Tensor) -> Result<Tensor, EvalError> {
    Ok(t.clone())
}

/// `pixel_shuffle` — `(N, C·r², H, W)` → `(N, C, H·r, W·r)`
/// (space-to-depth: each r×r input patch becomes r² output channels).
pub(crate) fn k_pixel_shuffle(t: &Tensor, r: i64) -> Result<Tensor, EvalError> {
    pixel_space_depth(t, r, true, "pixel_shuffle")
}

/// `pixel_unshuffle` — `(N, C, H, W)` → `(N, C·r², H/r, W/r)`
/// (depth-to-space, the inverse permutation).
pub(crate) fn k_pixel_unshuffle(t: &Tensor, r: i64) -> Result<Tensor, EvalError> {
    pixel_space_depth(t, r, false, "pixel_unshuffle")
}

fn pixel_space_depth(t: &Tensor, r: i64, shuffle: bool, op: &str) -> Result<Tensor, EvalError> {
    if t.shape.len() != 4 {
        return Err(op_err(op, "expected rank-4 NCHW"));
    }
    let r = usize::try_from(r).map_err(|_| op_err(op, "bad factor"))?;
    if r == 0 {
        return Err(op_err(op, "factor must be positive"));
    }
    let (n, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    let (out_c, out_h, out_w) = if shuffle {
        if c % (r * r) != 0 {
            return Err(op_err(op, format!("C={c} not divisible by r²={}", r * r)));
        }
        (c / (r * r), h * r, w * r)
    } else {
        if h % r != 0 || w % r != 0 {
            return Err(op_err(op, "H/W must be divisible by factor"));
        }
        (c * r * r, h / r, w / r)
    };
    let mut data = vec![0.0; n * out_c * out_h * out_w];
    // Both directions are the same bijection: for each (n, co, oh, ow)
    // the source channel packs the r×r sub-position, and the source
    // spatial coord is (oh/r·… ) — shuffle: out[n,co,oh,ow] reads
    // in[n, co·r² + (oh%r)·r + (ow%r), oh/r, ow/r]; unshuffle reads
    // in[n, co/(r²)… ] — implemented via explicit loops below.
    if shuffle {
        for ni in 0..n {
            for co in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let i = oh % r;
                        let j = ow % r;
                        let ic = co * r * r + i * r + j;
                        let ih = oh / r;
                        let iw = ow / r;
                        data[((ni * out_c + co) * out_h + oh) * out_w + ow] =
                            t.data[((ni * c + ic) * h + ih) * w + iw];
                    }
                }
            }
        }
    } else {
        // in: (N, C, H, W) → out: (N, C·r², H/r, W/r).
        // out[n, ci·r²+i·r+j, oh, ow] = in[n, ci, oh·r+i, ow·r+j].
        for ni in 0..n {
            for ci in 0..out_c / (r * r) {
                for i in 0..r {
                    for j in 0..r {
                        let oc = ci * r * r + i * r + j;
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let ih = oh * r + i;
                                let iw = ow * r + j;
                                data[((ni * out_c + oc) * out_h + oh) * out_w + ow] =
                                    t.data[((ni * c + ci) * h + ih) * w + iw];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Tensor { shape: vec![n, out_c, out_h, out_w], data })
}

/// `glu` — split last/dim into two halves: a * sigmoid(b).
pub(crate) fn k_glu(t: &Tensor, dim: i64) -> Result<Tensor, EvalError> {
    let d = wrap_dim(dim, t.shape.len(), "glu")?;
    let width = t.shape[d];
    if width % 2 != 0 {
        return Err(op_err("glu", format!("dim {d} extent {width} must be even")));
    }
    let half = width / 2;
    // a = narrow(0,half), b = narrow(half,half)
    let a = k_narrow(t, d as i64, 0, half as i64)?;
    let b = k_narrow(t, d as i64, half as i64, half as i64)?;
    let sig = map_unary(&b, |x| 1.0 / (1.0 + (-x).exp()));
    broadcast_elementwise(&a, &sig, "tensor.glu", |x, s| x * s)
}

/// `rms_norm` — `x * rsqrt(mean(x² over normalized_shape) + eps) * w`.
pub(crate) fn k_rms_norm(t: &Tensor, shape: &[i64], weight: Option<Tensor>, eps: f64) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    let ns: Vec<usize> = shape
        .iter()
        .map(|&n| usize::try_from(n).map_err(|_| op_err("rms_norm", "bad normalized_shape")))
        .collect::<Result<_, _>>()?;
    if ns.len() > rank || ns.len() == 0 {
        return Err(op_err("rms_norm", "normalized_shape must be a non-empty suffix"));
    }
    let off = rank - ns.len();
    for (i, &n) in ns.iter().enumerate() {
        if t.shape[off + i] != n {
            return Err(op_err("rms_norm", format!("normalized_shape {shape:?} does not match trailing dims {:?}", &t.shape[off..])));
        }
    }
    let inner: usize = ns.iter().product();
    let outer: usize = t.shape[..off].iter().product();
    // Per outer-slice: mean of squares over the trailing `inner` block
    // (the normalized dims are contiguous trailing — true only if they
    // are the LAST dims, which normalized_shape guarantees as suffix).
    let mut data = t.data.clone();
    for o in 0..outer {
        let base = o * inner;
        let ms = t.data[base..base + inner].iter().map(|x| x * x).sum::<f64>() / inner as f64;
        let rstd = 1.0 / (ms + eps).sqrt();
        for v in &mut data[base..base + inner] {
            *v *= rstd;
        }
    }
    let mut out = Tensor { shape: t.shape.clone(), data };
    if let Some(w) = weight {
        out = broadcast_elementwise(&out, &w, "tensor.rms_norm", |x, ww| x * ww)?;
    }
    Ok(out)
}

/// `layer_norm` — `(x - mean)/sqrt(var + eps) * w + b` over the
/// trailing `normalized_shape` dims (biased variance, like torch).
pub(crate) fn k_layer_norm(
    t: &Tensor,
    shape: &[i64],
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f64,
) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    let ns: Vec<usize> = shape
        .iter()
        .map(|&n| usize::try_from(n).map_err(|_| op_err("layer_norm", "bad normalized_shape")))
        .collect::<Result<_, _>>()?;
    if ns.is_empty() || ns.len() > rank {
        return Err(op_err("layer_norm", "normalized_shape must be a non-empty suffix"));
    }
    let off = rank - ns.len();
    for (i, &n) in ns.iter().enumerate() {
        if t.shape[off + i] != n {
            return Err(op_err("layer_norm", format!("normalized_shape {shape:?} does not match trailing dims {:?}", &t.shape[off..])));
        }
    }
    let inner: usize = ns.iter().product();
    let outer: usize = t.shape[..off].iter().product();
    let mut data = t.data.clone();
    for o in 0..outer {
        let base = o * inner;
        let slice = &t.data[base..base + inner];
        let mean = slice.iter().sum::<f64>() / inner as f64;
        let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / inner as f64;
        let rstd = 1.0 / (var + eps).sqrt();
        for v in data[base..base + inner].iter_mut() {
            *v = (*v - mean) * rstd;
        }
    }
    let mut out = Tensor { shape: t.shape.clone(), data };
    if let Some(w) = weight {
        out = broadcast_elementwise(&out, &w, "tensor.layer_norm", |x, ww| x * ww)?;
    }
    if let Some(b) = bias {
        out = broadcast_elementwise(&out, &b, "tensor.layer_norm", |x, bb| x + bb)?;
    }
    Ok(out)
}

/// `constant_pad_nd` — `pad` is the flat `[d0_l, d0_r, d1_l, d1_r, …]`
/// list ordered from the LAST dim backward (torch convention).
pub(crate) fn k_constant_pad_nd(t: &Tensor, pad: &[i64], value: f64) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    if pad.len() % 2 != 0 || pad.len() / 2 > rank {
        return Err(op_err("constant_pad_nd", format!("pad {pad:?} invalid for rank {rank}")));
    }
    let npad = pad.len() / 2;
    let mut out_shape = vec![0usize; rank];
    let mut pads = vec![(0i64, 0i64); rank];
    for i in 0..npad {
        // pad[2i], pad[2i+1] apply to dim rank-1-i.
        let d = rank - 1 - i;
        pads[d] = (pad[2 * i], pad[2 * i + 1]);
        out_shape[d] = (t.shape[d] as i64 + pads[d].0 + pads[d].1) as usize;
    }
    for d in npad..rank {
        out_shape[d] = t.shape[d];
    }
    let strides = row_major_strides(&t.shape);
    let ostrides = row_major_strides(&out_shape);
    let count: usize = out_shape.iter().product();
    let mut data = vec![value; count];
    let mut idx = vec![0usize; rank];
    for _ in 0..count {
        // Check whether idx is inside the padded region.
        let mut inside = true;
        let mut src_off = 0usize;
        for d in 0..rank {
            let (l, _) = pads[d];
            let i = idx[d] as i64 - l;
            if i < 0 || i >= t.shape[d] as i64 {
                inside = false;
                break;
            }
            src_off += i as usize * strides[d];
        }
        if inside {
            let dst_off: usize = idx.iter().zip(&ostrides).map(|(&i, &s)| i * s).sum();
            data[dst_off] = t.data[src_off];
        }
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(Tensor { shape: out_shape, data })
}

/// `avg_pool2d` — NCHW, `count_include_pad`/`ceil_mode`/`divisor_override`.
pub(crate) fn k_avg_pool2d(
    t: &Tensor,
    kernel: &[i64],
    stride: Option<Vec<i64>>,
    padding: Option<Vec<i64>>,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Option<i64>,
) -> Result<Tensor, EvalError> {
    if t.shape.len() != 4 {
        return Err(op_err("avg_pool2d", "expected rank-4 NCHW"));
    }
    if kernel.len() != 2 {
        return Err(op_err("avg_pool2d", "kernel_size must have 2 elements"));
    }
    let (kh, kw) = (kernel[0] as usize, kernel[1] as usize);
    if kh == 0 || kw == 0 {
        return Err(op_err("avg_pool2d", "kernel must be positive"));
    }
    let st = stride.unwrap_or_else(|| kernel.to_vec());
    let pad = padding.unwrap_or_else(|| vec![0, 0]);
    let (sh, sw) = (st[0] as usize, st[1] as usize);
    let (ph, pw) = (pad[0] as usize, pad[1] as usize);
    let (n, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    let oh = if ceil_mode {
        (h + 2 * ph).saturating_sub(kh).div_ceil(sh) + 1
    } else {
        (h + 2 * ph).saturating_sub(kh) / sh + 1
    };
    let ow = if ceil_mode {
        (w + 2 * pw).saturating_sub(kw).div_ceil(sw) + 1
    } else {
        (w + 2 * pw).saturating_sub(kw) / sw + 1
    };
    let mut data = vec![0.0; n * c * oh * ow];
    for ni in 0..n {
        for ci in 0..c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let y0 = oy * sh;
                    let x0 = ox * sw;
                    let mut sum = 0.0;
                    let mut cnt = 0usize;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = y0 + ky;
                            let ix = x0 + kx;
                            // Padded coords: iy- ph, ix - pw.
                            let py = iy as i64 - ph as i64;
                            let px = ix as i64 - pw as i64;
                            if py >= 0 && px >= 0 && (py as usize) < h && (px as usize) < w {
                                sum += t.data[((ni * c + ci) * h + py as usize) * w + px as usize];
                                cnt += 1;
                            } else if count_include_pad {
                                cnt += 1;
                            }
                        }
                    }
                    let div = divisor_override.unwrap_or(cnt as i64).max(1) as f64;
                    data[((ni * c + ci) * oh + oy) * ow + ox] = sum / div;
                }
            }
        }
    }
    Ok(Tensor { shape: vec![n, c, oh, ow], data })
}

// ---------------------------------------------------------------------------
// Linalg
// ---------------------------------------------------------------------------

/// Strict rank-2 matmul core (mm/bmm share the naive triple loop).
fn mm2(a: &Tensor, b: &Tensor, op: &str) -> Result<Tensor, EvalError> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(op_err(op, "expected rank-2 operands"));
    }
    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    if a.shape[1] != b.shape[0] {
        return Err(op_err(op, format!("inner dim mismatch {k} vs {}", b.shape[0])));
    }
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for l in 0..k {
            let av = a.data[i * k + l];
            if av == 0.0 {
                continue;
            }
            for j in 0..n {
                data[i * n + j] += av * b.data[l * n + j];
            }
        }
    }
    Ok(Tensor { shape: vec![m, n], data })
}

/// `mm` — rank-2 product.
pub(crate) fn k_mm(a: &Tensor, b: &Tensor) -> Result<Tensor, EvalError> {
    mm2(a, b, "mm")
}

/// `bmm` — strict `[B,m,k]×[B,k,n]`.
pub(crate) fn k_bmm(a: &Tensor, b: &Tensor) -> Result<Tensor, EvalError> {
    if a.shape.len() != 3 || b.shape.len() != 3 {
        return Err(op_err("bmm", "expected rank-3 operands"));
    }
    if a.shape[0] != b.shape[0] || a.shape[2] != b.shape[1] {
        return Err(op_err("bmm", format!("shape mismatch {:?} x {:?}", a.shape, b.shape)));
    }
    let (batch, m, k, n) = (a.shape[0], a.shape[1], a.shape[2], b.shape[2]);
    let mut data = vec![0.0; batch * m * n];
    for bi in 0..batch {
        for i in 0..m {
            for l in 0..k {
                let av = a.data[(bi * m + i) * k + l];
                if av == 0.0 {
                    continue;
                }
                for j in 0..n {
                    data[(bi * m + i) * n + j] += av * b.data[(bi * k + l) * n + j];
                }
            }
        }
    }
    Ok(Tensor { shape: vec![batch, m, n], data })
}

/// `addmm`/`addbmm`/`baddbmm` — `beta*self + alpha*(mm/bmm)`.
pub(crate) fn k_addmm(args: &[serde_json::Value], batched: bool, self_first: bool) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    // addmm(self, mat1, mat2, *, beta=1, alpha=1); addbmm/baddbmm same
    // with batch1/batch2 names — positions identical.
    let _ = self_first;
    let s = require_tensor(args, 0)?;
    let m1 = require_tensor(args, 1)?;
    let m2 = require_tensor(args, 2)?;
    let beta = opt_f64(args.get(3), 1.0)?;
    let alpha = opt_f64(args.get(4), 1.0)?;
    let prod = if batched { k_bmm(&m1, &m2)? } else { mm2(&m1, &m2, "addmm")? };
    // broadcast_elementwise handles beta/alpha via scalar tensors:
    // out = s*beta + prod*alpha — two broadcasts then add.
    let sb = broadcast_elementwise(&s, &Tensor { shape: vec![], data: vec![beta] }, "addmm", |x, b| x * b)?;
    let pa = broadcast_elementwise(&prod, &Tensor { shape: vec![], data: vec![alpha] }, "addmm", |x, a| x * a)?;
    let shape = broadcast_shape_pub(&sb.shape, &pa.shape, "addmm")?;
    let sf = broadcast_to(&sb, &shape, "addmm")?;
    let pf = broadcast_to(&pa, &shape, "addmm")?;
    Ok(enc(Tensor {
        shape,
        data: sf.data.iter().zip(&pf.data).map(|(&x, &y)| x + y).collect(),
    }))
}

/// `mv` — matrix @ vector → rank-1.
pub(crate) fn k_mv(a: &Tensor, v: &Tensor) -> Result<Tensor, EvalError> {
    if a.shape.len() != 2 || v.shape.len() != 1 {
        return Err(op_err("mv", "expected matrix (m,k) and vector (k,)"));
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    if v.shape[0] != k {
        return Err(op_err("mv", format!("vec length {} != k {k}", v.shape[0])));
    }
    let mut data = vec![0.0; m];
    for i in 0..m {
        for l in 0..k {
            data[i] += a.data[i * k + l] * v.data[l];
        }
    }
    Ok(Tensor { shape: vec![m], data })
}

/// `dot`/`vdot`/`vecdot` — flattened dot product → rank-0 tensor.
pub(crate) fn k_dot(a: &Tensor, b: &Tensor) -> Result<Tensor, EvalError> {
    if a.data.len() != b.data.len() {
        return Err(op_err("dot", format!("length mismatch {} vs {}", a.data.len(), b.data.len())));
    }
    let s: f64 = a.data.iter().zip(&b.data).map(|(&x, &y)| x * y).sum();
    Ok(Tensor { shape: vec![], data: vec![s] })
}

/// `ger`/`outer` — outer product `self[i]*vec2[j]` → (m,n).
pub(crate) fn k_ger(a: &Tensor, b: &Tensor) -> Result<Tensor, EvalError> {
    if a.shape.len() != 1 || b.shape.len() != 1 {
        return Err(op_err("ger", "expected rank-1 vectors"));
    }
    let (m, n) = (a.shape[0], b.shape[0]);
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            data[i * n + j] = a.data[i] * b.data[j];
        }
    }
    Ok(Tensor { shape: vec![m, n], data })
}

/// `addr` — `beta*self + alpha*(vec1 ⊗ vec2)`.
pub(crate) fn k_addr(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let s = require_tensor(args, 0)?;
    let v1 = require_tensor(args, 1)?;
    let v2 = require_tensor(args, 2)?;
    let beta = opt_f64(args.get(3), 1.0)?;
    let alpha = opt_f64(args.get(4), 1.0)?;
    let outer = k_ger(&v1, &v2)?;
    let sb = broadcast_elementwise(&s, &Tensor { shape: vec![], data: vec![beta] }, "addr", |x, b| x * b)?;
    let oa = broadcast_elementwise(&outer, &Tensor { shape: vec![], data: vec![alpha] }, "addr", |x, a| x * a)?;
    let shape = broadcast_shape_pub(&sb.shape, &oa.shape, "addr")?;
    let sf = broadcast_to(&sb, &shape, "addr")?;
    let of = broadcast_to(&oa, &shape, "addr")?;
    Ok(enc(Tensor {
        shape,
        data: sf.data.iter().zip(&of.data).map(|(&x, &y)| x + y).collect(),
    }))
}

/// `addmv` — `beta*self + alpha*(mat @ vec)`.
pub(crate) fn k_addmv(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let s = require_tensor(args, 0)?;
    let mat = require_tensor(args, 1)?;
    let vec = require_tensor(args, 2)?;
    let beta = opt_f64(args.get(3), 1.0)?;
    let alpha = opt_f64(args.get(4), 1.0)?;
    let mv = k_mv(&mat, &vec)?;
    let sb = broadcast_elementwise(&s, &Tensor { shape: vec![], data: vec![beta] }, "addmv", |x, b| x * b)?;
    let ma = broadcast_elementwise(&mv, &Tensor { shape: vec![], data: vec![alpha] }, "addmv", |x, a| x * a)?;
    let shape = broadcast_shape_pub(&sb.shape, &ma.shape, "addmv")?;
    let sf = broadcast_to(&sb, &shape, "addmv")?;
    let mf = broadcast_to(&ma, &shape, "addmv")?;
    Ok(enc(Tensor {
        shape,
        data: sf.data.iter().zip(&mf.data).map(|(&x, &y)| x + y).collect(),
    }))
}

// ---------------------------------------------------------------------------
// Constructors + metadata
// ---------------------------------------------------------------------------

/// `full`/`full_like`/`zeros_like`/`ones_like` — fill value or 0/1.
pub(crate) fn k_full(shape: &[i64], fill: f64) -> Result<Tensor, EvalError> {
    let shape: Vec<usize> = shape
        .iter()
        .map(|&n| usize::try_from(n).map_err(|_| op_err("full", "negative extent")))
        .collect::<Result<_, _>>()?;
    let n: usize = shape.iter().product();
    Ok(Tensor { shape, data: vec![fill; n] })
}

pub(crate) fn k_full_like(t: &Tensor, fill: f64) -> Result<Tensor, EvalError> {
    Ok(Tensor { shape: t.shape.clone(), data: vec![fill; t.data.len()] })
}

pub(crate) fn k_zeros_like(t: &Tensor) -> Result<Tensor, EvalError> {
    k_full_like(t, 0.0)
}

pub(crate) fn k_ones_like(t: &Tensor) -> Result<Tensor, EvalError> {
    k_full_like(t, 1.0)
}

/// `arange` — 1 arg (end), 2 args (start,end), or 3 (start,end,step).
pub(crate) fn k_arange(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    let (start, end, step) = match args.len() {
        1 => (0.0, req_f64(&args[0])?, 1.0),
        2 => (req_f64(&args[0])?, req_f64(&args[1])?, 1.0),
        _ => (
            req_f64(&args[0])?,
            req_f64(&args[1])?,
            req_f64(&args[2])?,
        ),
    };
    if step == 0.0 {
        return Err(op_err("arange", "step must be nonzero"));
    }
    let mut data = Vec::new();
    let mut x = start;
    if step > 0.0 {
        while x < end {
            data.push(x);
            x += step;
        }
    } else {
        while x > end {
            data.push(x);
            x += step;
        }
    }
    Ok(enc(Tensor { shape: vec![data.len()], data }))
}

/// `linspace`/`logspace` — `(start, end, steps)` (+base for logspace).
pub(crate) fn k_linspace(args: &[serde_json::Value], log: bool) -> Result<serde_json::Value, EvalError> {
    let start = req_f64(&args[0])?;
    let end = req_f64(&args[1])?;
    let steps = usize::try_from(req_i64(&args[2])?).map_err(|_| op_err("linspace", "bad steps"))?;
    if steps == 0 {
        return Err(op_err("linspace", "steps must be positive"));
    }
    let base = if log {
        opt_f64(args.get(3), 10.0)?
    } else {
        0.0
    };
    let mut data = Vec::with_capacity(steps);
    if steps == 1 {
        data.push(if log { base.powf(start) } else { start });
    } else {
        for i in 0..steps {
            let f = i as f64 / (steps - 1) as f64;
            let v = start + f * (end - start);
            data.push(if log { base.powf(v) } else { v });
        }
    }
    Ok(enc(Tensor { shape: vec![steps], data }))
}

/// `eye` — 1 arg (n → n×n) or 2 (n,m).
pub(crate) fn k_eye(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    let n = usize::try_from(req_i64(&args[0])?).map_err(|_| op_err("eye", "bad n"))?;
    let m = if args.len() >= 2 {
        usize::try_from(req_i64(&args[1])?).map_err(|_| op_err("eye", "bad m"))?
    } else {
        n
    };
    let mut data = vec![0.0; n * m];
    for i in 0..n.min(m) {
        data[i * m + i] = 1.0;
    }
    Ok(enc(Tensor { shape: vec![n, m], data }))
}

/// `size` — 0 args → shape list; 1 int arg → that dim's extent.
pub(crate) fn k_size(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let t = require_tensor(args, 0)?;
    match args.get(1) {
        None => Ok(json!(t.shape)),
        Some(v) => {
            let d = wrap_dim(req_i64(v)?, t.shape.len(), "size")?;
            Ok(json!(t.shape[d] as i64))
        }
    }
}

/// `stride` — row-major strides (the only layout we have).
pub(crate) fn k_stride(t: &Tensor) -> Result<serde_json::Value, EvalError> {
    Ok(json!(row_major_strides(&t.shape)))
}

/// `numel`/`nelement`/`dim`/`item`/`is_contiguous` — metadata.
pub(crate) fn k_numel(t: &Tensor) -> serde_json::Value {
    json!(t.data.len() as i64)
}
pub(crate) fn k_dim(t: &Tensor) -> serde_json::Value {
    json!(t.shape.len() as i64)
}
pub(crate) fn k_item(t: &Tensor) -> Result<serde_json::Value, EvalError> {
    if t.data.len() != 1 {
        return Err(op_err("item", "tensor must have exactly one element"));
    }
    Ok(json!(t.data[0]))
}
pub(crate) fn k_is_contiguous(_t: &Tensor) -> serde_json::Value {
    // The wire tensor is dense row-major — always contiguous.
    json!(true)
}

/// `unsqueeze` — insert a size-1 axis. `dim` follows torch's
/// `[-(rank+1), rank]` range.
pub(crate) fn k_unsqueeze(t: &Tensor, dim: i64) -> Result<Tensor, EvalError> {
    let rank = t.shape.len() as i64;
    if dim < -(rank + 1) || dim > rank {
        return Err(op_err("unsqueeze", "dim out of range"));
    }
    let d = dim.rem_euclid(rank + 1) as usize;
    let mut shape = t.shape.clone();
    shape.insert(d, 1);
    Ok(Tensor { shape, data: t.data.clone() })
}

// ---------------------------------------------------------------------------
// Custom multi-arg drivers (called with raw `args` by generated code)
// ---------------------------------------------------------------------------

/// `layer_norm` positional decode: `(input, normalized_shape,
/// weight?, bias?, eps?, cudnn_enable?)` — cudnn_enable ignored.
pub(crate) fn drv_layer_norm(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let t = require_tensor(args, 0)?;
    let shape = req_i64s(arg_at(args, 1)?)?;
    let weight = opt_tensor_pos(args, 2)?;
    let bias = opt_tensor_pos(args, 3)?;
    let eps = opt_f64(args.get(4), 1e-5)?;
    // args[5] cudnn_enable ignored.
    Ok(enc(k_layer_norm(&t, &shape, weight, bias, eps)?))
}

/// `rms_norm` — `(input, normalized_shape, weight?, eps?)` (no bias in ATen).
pub(crate) fn drv_rms_norm(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let t = require_tensor(args, 0)?;
    let shape = req_i64s(arg_at(args, 1)?)?;
    let weight = opt_tensor_pos(args, 2)?;
    // ATen `eps=None` → 1e-6 (matches torch.nn.RMSNorm's documented
    // default; layer_norm uses 1e-5).
    let eps = opt_f64n(args.get(3))?.unwrap_or(1e-6);
    Ok(enc(k_rms_norm(&t, &shape, weight, eps)?))
}

fn opt_tensor_pos(args: &[serde_json::Value], pos: usize) -> Result<Option<Tensor>, EvalError> {
    match args.get(pos) {
        Some(v) if !v.is_null() => Ok(Some(decode_tensor(v)?)),
        _ => Ok(None),
    }
}

/// `gelu` raw driver — `(self, approximate?)` where `approximate` is
/// the ATen `str` kwarg `"none"` (default, exact) or `"tanh"`.
pub(crate) fn drv_gelu(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let t = require_tensor(args, 0)?;
    let tanh = match args.get(1) {
        None | Some(Value::Null) => false,
        Some(Value::String(s)) if s == "none" => false,
        Some(Value::String(s)) if s == "tanh" => true,
        Some(v) => {
            return Err(op_err(
                "gelu",
                format!("approximate must be \"none\" or \"tanh\", got {v}"),
            ))
        }
    };
    let out = if tanh { k_gelu_tanh(&t)? } else { k_gelu(&t)? };
    Ok(enc(out))
}

/// Collect an int list from either a JSON array at `pos` or consecutive
/// integer args starting at `pos` (torch's `*sizes` varargs surface).
fn int_args(args: &[serde_json::Value], pos: usize, op: &str) -> Result<Vec<i64>, EvalError> {
    match args.get(pos) {
        Some(v) if v.is_array() => req_i64s(v),
        Some(v) if v.is_i64() || v.is_u64() => {
            let mut out = vec![req_i64(v)?];
            let mut i = pos + 1;
            while let Some(v) = args.get(i) {
                if v.is_i64() || v.is_u64() {
                    out.push(req_i64(v)?);
                    i += 1;
                } else {
                    break;
                }
            }
            Ok(out)
        }
        _ => Err(op_err(op, "expected int list or varargs ints")),
    }
}

/// `repeat` — `(self, *repeats)` list-or-varargs.
pub(crate) fn drv_repeat(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let t = require_tensor(args, 0)?;
    let reps = int_args(args, 1, "repeat")?;
    Ok(enc(k_repeat(&t, &reps)?))
}

/// `view` — `(self, *size)` list-or-varargs; alias of reshape.
pub(crate) fn drv_view(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let t = require_tensor(args, 0)?;
    let size = int_args(args, 1, "view")?;
    Ok(enc(k_reshape(&t, &size)?))
}

/// `expand` raw driver — `(self, size-or-*sizes, implicit=False)`.
/// `implicit` expand (arbitrary growth without the broadcast-1 rule)
/// is a seam.
pub(crate) fn drv_expand(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let t = require_tensor(args, 0)?;
    // Determine where the size args end: array form → pos2 may be
    // implicit; varargs form → first non-int arg may be implicit.
    let (size, implicit_pos) = if args.get(1).is_some_and(|v| v.is_array()) {
        (req_i64s(arg_at(args, 1)?)?, 2)
    } else {
        (int_args(args, 1, "expand")?, {
            // count leading ints from pos 1
            let mut i = 1;
            while args.get(i).is_some_and(|v| v.is_i64() || v.is_u64()) {
                i += 1;
            }
            i
        })
    };
    let implicit = opt_bool(args.get(implicit_pos), false)?;
    if implicit {
        return Err(op_err(
            "expand",
            "implicit=True expand is a manual seam (arbitrary-growth semantics)",
        ));
    }
    Ok(enc(k_expand(&t, &size)?))
}

/// ATen accepts `int` or `(int, int)` for 2-D params — normalize a
/// single-element list to a pair.
fn pair2(v: Vec<i64>, op: &str, name: &str) -> Result<Vec<i64>, EvalError> {
    match v.len() {
        1 => Ok(vec![v[0], v[0]]),
        2 => Ok(v),
        _ => Err(op_err(op, &format!("{name} must have 1 or 2 elements"))),
    }
}

/// `avg_pool2d` positional decode — kernel/stride/padding take int or
/// pair, matching ATen's `int[2]`/`SymInt[2]`-or-scalar surface.
pub(crate) fn drv_avg_pool2d(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let t = require_tensor(args, 0)?;
    let kernel = pair2(req_dimlist(arg_at(args, 1)?)?, "avg_pool2d", "kernel_size")?;
    let stride = opt_dimlist(args.get(2), None)?
        .map(|v| pair2(v, "avg_pool2d", "stride"))
        .transpose()?;
    let padding = opt_dimlist(args.get(3), Some(&[0, 0][..]))?
        .map(|v| pair2(v, "avg_pool2d", "padding"))
        .transpose()?;
    let ceil_mode = opt_bool(args.get(4), false)?;
    let count_include_pad = opt_bool(args.get(5), true)?;
    let divisor_override = opt_i64n(args.get(6))?;
    Ok(enc(k_avg_pool2d(
        &t,
        &kernel,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )?))
}

/// `var`/`std` raw driver is `k_var_std` (already raw).

// ---------------------------------------------------------------------------
// Seam closures — handwritten kernels for formerly-declared `# seam` ops
// and a few trivially-supported promotions. Same conventions as above:
// decoded args in, `Result<Tensor>`/`Result<Value>` out; multi-return
// ATen tuples encode as `{"values","indices"}` records, tensor lists as
// JSON arrays of wire tensors.
// ---------------------------------------------------------------------------

/// `ord`-norm of `t` over `dims` (`None` → all axes; `Some([])` →
/// identity). `keepdim` retains size-1 reduced axes. `ord == 0` counts
/// nonzeros; `±inf` → max/min `|x|`; finite `p` → `(Σ|x|^p)^(1/p)`.
fn p_norm(t: &Tensor, dims: Option<&[i64]>, ord: f64, keepdim: bool) -> Result<Tensor, EvalError> {
    let dv = dims.map(|d| d.to_vec());
    if ord == f64::INFINITY {
        let a = map_unary(t, f64::abs);
        return reduce_amax(&a, dv, keepdim);
    }
    if ord == f64::NEG_INFINITY {
        let a = map_unary(t, f64::abs);
        return reduce_amin(&a, dv, keepdim);
    }
    if ord == 0.0 {
        let mask = map_unary(t, |x| if x != 0.0 { 1.0 } else { 0.0 });
        return reduce_sum(&mask, dv, keepdim);
    }
    let a = map_unary(t, |x| x.abs().powf(ord));
    let s = reduce_sum(&a, dv, keepdim)?;
    Ok(map_unary(&s, |x| x.powf(1.0 / ord)))
}

/// `norm` — `aten::norm.Scalar(self, p=2)`: p-norm over ALL elements →
/// rank-0 tensor.
pub(crate) fn k_norm(t: &Tensor, p: f64) -> Result<Tensor, EvalError> {
    p_norm(t, None, p, false)
}

/// `renorm` — `aten::renorm(self, p, dim, maxnorm)`: scale each slice
/// along `dim` by `min(1, maxnorm/‖·‖_p)`. A zero norm leaves the slice
/// unchanged (the `maxnorm/0 → inf/NaN → min → 1` path handles it).
pub(crate) fn k_renorm(t: &Tensor, p: f64, dim: i64, maxnorm: f64) -> Result<Tensor, EvalError> {
    let n = p_norm(t, Some(&[dim]), p, true)?;
    let full = broadcast_to(&n, &t.shape, "renorm")?;
    Ok(Tensor {
        shape: t.shape.clone(),
        data: t
            .data
            .iter()
            .zip(&full.data)
            .map(|(&x, &nv)| x * (maxnorm / nv).min(1.0))
            .collect(),
    })
}

/// `normalize` — the `F.normalize` composite over the pinned
/// `aten::linalg_vector_norm` schema: `x / max(‖x‖_ord(dim), eps)` with
/// `eps = 1e-12` (F.normalize's default). The output shape is always
/// `t.shape`; `keepdim` only shaped the internal norm and is accepted
/// for schema compatibility. `dim=None` norms over ALL axes.
pub(crate) fn k_normalize(
    t: &Tensor,
    ord: f64,
    dims: Option<&[i64]>,
    _keepdim: bool,
) -> Result<Tensor, EvalError> {
    const EPS: f64 = 1e-12;
    let n = p_norm(t, dims, ord, true)?;
    let full = broadcast_to(&n, &t.shape, "normalize")?;
    Ok(Tensor {
        shape: t.shape.clone(),
        data: t
            .data
            .iter()
            .zip(&full.data)
            .map(|(&x, &nv)| {
                let d = if nv.is_nan() { f64::NAN } else { nv.max(EPS) };
                x / d
            })
            .collect(),
    })
}

/// `cross` — `aten::cross(self, other, dim=None)`. `dim=None` uses dim 0
/// for rank-1 operands and errors for higher rank (torch infers "first
/// dim of size 3"; we don't guess). Operands must have equal shapes.
pub(crate) fn k_cross(a: &Tensor, b: &Tensor, dim: Option<i64>) -> Result<Tensor, EvalError> {
    if a.shape != b.shape {
        return Err(op_err(
            "cross",
            format!("shape mismatch {:?} vs {:?}", a.shape, b.shape),
        ));
    }
    let d = match dim {
        Some(d) => wrap_dim(d, a.shape.len(), "cross")?,
        None => {
            if a.shape.len() == 1 {
                0
            } else {
                return Err(op_err("cross", "dim required for rank>1 operands"));
            }
        }
    };
    if a.shape[d] != 3 {
        return Err(op_err(
            "cross",
            format!("cross dim extent {} != 3", a.shape[d]),
        ));
    }
    let inner: usize = a.shape[d + 1..].iter().product();
    let outer: usize = a.shape[..d].iter().product();
    let mut data = vec![0.0; a.data.len()];
    for o in 0..outer {
        for j in 0..inner {
            let at = |_t: &Tensor, i: usize| (o * 3 + i) * inner + j;
            let (a0, a1, a2) = (a.data[at(a, 0)], a.data[at(a, 1)], a.data[at(a, 2)]);
            let (b0, b1, b2) = (b.data[at(b, 0)], b.data[at(b, 1)], b.data[at(b, 2)]);
            data[at(a, 0)] = a1 * b2 - a2 * b1;
            data[at(a, 1)] = a2 * b0 - a0 * b2;
            data[at(a, 2)] = a0 * b1 - a1 * b0;
        }
    }
    Ok(Tensor {
        shape: a.shape.clone(),
        data,
    })
}

/// `tensordot` — `aten::tensordot(self, other, dims_self, dims_other)`:
/// contract the listed axes pairwise; result dims are a's free dims (in
/// order) followed by b's free dims. Empty contract lists give the outer
/// product (handled uniformly — the contract extent is then 1).
pub(crate) fn k_tensordot(
    a: &Tensor,
    b: &Tensor,
    dims_self: &[i64],
    dims_other: &[i64],
) -> Result<Tensor, EvalError> {
    if dims_self.len() != dims_other.len() {
        return Err(op_err(
            "tensordot",
            format!(
                "contract count mismatch {} vs {}",
                dims_self.len(),
                dims_other.len()
            ),
        ));
    }
    let (ra, rb) = (a.shape.len(), b.shape.len());
    let mut ca = Vec::with_capacity(dims_self.len());
    let mut cb = Vec::with_capacity(dims_other.len());
    for (&x, &y) in dims_self.iter().zip(dims_other) {
        let da = wrap_dim(x, ra, "tensordot")?;
        let db = wrap_dim(y, rb, "tensordot")?;
        if ca.contains(&da) || cb.contains(&db) {
            return Err(op_err("tensordot", "duplicate contract dim"));
        }
        if a.shape[da] != b.shape[db] {
            return Err(op_err(
                "tensordot",
                format!("contract size mismatch {} vs {}", a.shape[da], b.shape[db]),
            ));
        }
        ca.push(da);
        cb.push(db);
    }
    let free_a: Vec<usize> = (0..ra).filter(|d| !ca.contains(d)).collect();
    let free_b: Vec<usize> = (0..rb).filter(|d| !cb.contains(d)).collect();
    let mut out_shape: Vec<usize> = free_a.iter().map(|&d| a.shape[d]).collect();
    out_shape.extend(free_b.iter().map(|&d| b.shape[d]));
    let na: usize = free_a.iter().map(|&d| a.shape[d]).product();
    let nb: usize = free_b.iter().map(|&d| b.shape[d]).product();
    let k: usize = ca.iter().map(|&d| a.shape[d]).product();
    // permute → [free…, contract…] / [contract…, free…], reshape, mm.
    let perm_a: Vec<usize> = free_a.iter().chain(&ca).copied().collect();
    let perm_b: Vec<usize> = cb.iter().chain(&free_b).copied().collect();
    let a2 = reshape_like(&permute_dims(a, &perm_a, "tensordot")?, &[na, k], "tensordot")?;
    let b2 = reshape_like(&permute_dims(b, &perm_b, "tensordot")?, &[k, nb], "tensordot")?;
    let c = mm2(&a2, &b2, "tensordot")?;
    reshape_like(&c, &out_shape, "tensordot")
}

/// `kron` — `aten::kron(self, other)`: same-rank operands; per axis the
/// output extent is `a[d]*b[d]` and `out[…, o_d, …] =
/// a[…, o_d/b_d, …] * b[…, o_d%b_d, …]`.
pub(crate) fn k_kron(a: &Tensor, b: &Tensor) -> Result<Tensor, EvalError> {
    if a.shape.len() != b.shape.len() {
        return Err(op_err(
            "kron",
            format!("rank mismatch {:?} vs {:?}", a.shape, b.shape),
        ));
    }
    let rank = a.shape.len();
    let out_shape: Vec<usize> = a.shape.iter().zip(&b.shape).map(|(&x, &y)| x * y).collect();
    let sa = row_major_strides(&a.shape);
    let sb = row_major_strides(&b.shape);
    let count: usize = out_shape.iter().product();
    let mut data = Vec::with_capacity(count);
    let mut idx = vec![0usize; rank];
    for _ in 0..count {
        let (mut ia, mut ib) = (0usize, 0usize);
        for d in 0..rank {
            ia += (idx[d] / b.shape[d]) * sa[d];
            ib += (idx[d] % b.shape[d]) * sb[d];
        }
        data.push(a.data[ia] * b.data[ib]);
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

/// `bincount` — `aten::bincount(self, weights?, minlength=0)`: histogram
/// of non-negative integer-valued entries; `weights` (same numel)
/// multiplies each count.
pub(crate) fn k_bincount(
    t: &Tensor,
    weights: Option<&Tensor>,
    minlength: i64,
) -> Result<Tensor, EvalError> {
    if let Some(w) = weights {
        if w.data.len() != t.data.len() {
            return Err(op_err("bincount", "weights numel mismatch"));
        }
    }
    let mut max_i = -1i64;
    for &x in &t.data {
        if !x.is_finite() || x < 0.0 || x.fract() != 0.0 {
            return Err(op_err(
                "bincount",
                format!("input must be non-negative integers, got {x}"),
            ));
        }
        let i = x as i64;
        if i > 100_000_000 {
            return Err(op_err("bincount", format!("index {i} too large")));
        }
        max_i = max_i.max(i);
    }
    let min = usize::try_from(minlength).map_err(|_| op_err("bincount", "negative minlength"))?;
    let n = (max_i.max(0) as usize + 1).max(min);
    let mut data = vec![0.0; n];
    for (k, &x) in t.data.iter().enumerate() {
        data[x as usize] += weights.map_or(1.0, |w| w.data[k]);
    }
    Ok(Tensor {
        shape: vec![n],
        data,
    })
}

/// `corrcoef` — `aten::corrcoef(self)`: rowvar semantics — each ROW is a
/// variable, columns are observations. Rank-1 input is a single variable
/// → rank-0 self-correlation (1.0, or NaN for a constant/empty input).
pub(crate) fn k_corrcoef(t: &Tensor) -> Result<Tensor, EvalError> {
    match t.shape.len() {
        1 => {
            let n = t.shape[0] as f64;
            if n < 2.0 {
                return Ok(Tensor {
                    shape: vec![],
                    data: vec![f64::NAN],
                });
            }
            let mean = t.data.iter().sum::<f64>() / n;
            let var = t
                .data
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / (n - 1.0);
            Ok(Tensor {
                shape: vec![],
                data: vec![if var > 0.0 { 1.0 } else { f64::NAN }],
            })
        }
        2 => {
            let (m, n) = (t.shape[0], t.shape[1]);
            // Center each row, then cov = Xc·Xcᵀ/(n−1), corr = cov/√(diag⊗diag).
            let mut xc = t.data.clone();
            for i in 0..m {
                let mean = xc[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64;
                for v in &mut xc[i * n..(i + 1) * n] {
                    *v -= mean;
                }
            }
            let denom = (n as f64 - 1.0).max(f64::MIN_POSITIVE);
            let mut cov = vec![0.0; m * m];
            for i in 0..m {
                for j in 0..=i {
                    let s: f64 = (0..n).map(|k| xc[i * n + k] * xc[j * n + k]).sum();
                    cov[i * m + j] = s / denom;
                    cov[j * m + i] = s / denom;
                }
            }
            for i in 0..m {
                for j in 0..m {
                    let d = (cov[i * m + i] * cov[j * m + j]).sqrt();
                    cov[i * m + j] = if d == 0.0 { f64::NAN } else { cov[i * m + j] / d };
                }
            }
            Ok(Tensor {
                shape: vec![m, m],
                data: cov,
            })
        }
        r => Err(op_err("corrcoef", format!("expected rank 1 or 2, got {r}"))),
    }
}

/// `to.dtype` — dtype cast. The wire is f64-dense, so float64-family
/// requests (`"float64"`/`"double"`/`"f64"`) are the identity; every
/// other dtype name errors loudly rather than silently degrading
/// precision. A tensor rhs (`aten::to.other`-style) is a same-dtype
/// copy → clone. `non_blocking`/`copy`/`memory_format` are accepted and
/// ignored (immutable wire — a fresh encoded tensor is always returned).
pub(crate) fn k_to(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let t = require_tensor(args, 0)?;
    let spec = args
        .get(1)
        .ok_or_else(|| op_err("to", "missing dtype argument"))?;
    match spec {
        Value::String(s) => match s.to_ascii_lowercase().as_str() {
            "float64" | "double" | "f64" => Ok(enc(t)),
            other => Err(op_err(
                "to",
                format!("unsupported dtype {other:?} — the surface is f64 dense"),
            )),
        },
        v if v.get("__tensor__").is_some() => Ok(enc(t)),
        _ => Err(op_err(
            "to",
            "dtype must be a string (e.g. \"float64\") or a tensor",
        )),
    }
}

/// Take position `pos` along axis `d`, dropping the axis; `keepdim`
/// reinserts it as size-1 (identical flat layout).
fn select_dim_index(
    t: &Tensor,
    d: usize,
    pos: usize,
    keepdim: bool,
    op: &str,
) -> Result<Tensor, EvalError> {
    let w = t.shape[d];
    if pos >= w {
        return Err(op_err(op, format!("index {pos} out of range for extent {w}")));
    }
    let tail: usize = t.shape[d + 1..].iter().product();
    let outer: usize = t.shape[..d].iter().product();
    let mut data = Vec::with_capacity(t.data.len() / w.max(1));
    for o in 0..outer {
        for j in 0..tail {
            data.push(t.data[(o * w + pos) * tail + j]);
        }
    }
    let mut shape: Vec<usize> = t
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != d)
        .map(|(_, &s)| s)
        .collect();
    if keepdim {
        shape.insert(d, 1);
    }
    Ok(Tensor { shape, data })
}

/// `median` — `aten::median(self)` scalar form: the LOWER median of the
/// flattened tensor (torch returns an element, never the mean of the two
/// middle values) → rank-0. The `median.dim` sibling `(self, dim,
/// keepdim)` → `{values, indices}` is covered by the same raw driver
/// (k_max_min-style merged overload).
pub(crate) fn k_median(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let a = require_tensor(args, 0)?;
    match args.get(1) {
        None => {
            if a.data.is_empty() {
                return Err(op_err("median", "empty tensor"));
            }
            let mut v = a.data.clone();
            v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            Ok(enc(Tensor {
                shape: vec![],
                data: vec![v[(v.len() - 1) / 2]],
            }))
        }
        Some(v) => {
            let dim = req_i64(v)?;
            let keepdim = opt_bool(args.get(2), false)?;
            let d = wrap_dim(dim, a.shape.len(), "median")?;
            let w = a.shape[d];
            if w == 0 {
                return Err(op_err("median", "empty dim"));
            }
            let pos = (w - 1) / 2; // lower median position in sorted order
            let (vals, idxs) = sort_dim(&a, dim, false, "median")?;
            let mv = select_dim_index(&vals, d, pos, keepdim, "median")?;
            let mi = select_dim_index(&idxs, d, pos, keepdim, "median")?;
            Ok(json!({ "values": enc(mv), "indices": enc(mi) }))
        }
    }
}

/// `mode` — `aten::mode(self, dim=-1, keepdim=False)` →
/// `{values, indices}`: the most frequent value along `dim` and its
/// first-occurrence index. Ties break to the smallest first-occurrence
/// position (documented choice — torch's tie rule is version-dependent).
pub(crate) fn k_mode(t: &Tensor, dim: i64, keepdim: bool) -> Result<serde_json::Value, EvalError> {
    let d = wrap_dim(dim, t.shape.len(), "mode")?;
    let w = t.shape[d];
    if w == 0 {
        return Err(op_err("mode", "empty dim"));
    }
    let tail: usize = t.shape[d + 1..].iter().product();
    let outer: usize = t.shape[..d].iter().product();
    let kept: Vec<usize> = t
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != d)
        .map(|(_, &s)| s)
        .collect();
    let mut vshape = kept.clone();
    if keepdim {
        vshape.insert(d, 1);
    }
    let n_out = t.data.len() / w;
    let mut vdata = vec![0.0; n_out];
    let mut idata = vec![0.0; n_out];
    for o in 0..outer {
        for j in 0..tail {
            let mut items: Vec<(f64, usize)> = (0..w)
                .map(|i| (t.data[(o * w + i) * tail + j], i))
                .collect();
            items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let (mut bv, mut bi, mut bc) = (0.0f64, usize::MAX, 0usize);
            let mut p = 0usize;
            while p < items.len() {
                let mut q = p + 1;
                while q < items.len() && items[q].0 == items[p].0 {
                    q += 1;
                }
                let count = q - p;
                let first = items[p..q].iter().map(|&(_, i)| i).min().unwrap();
                if count > bc || (count == bc && first < bi) {
                    bc = count;
                    bv = items[p].0;
                    bi = first;
                }
                p = q;
            }
            let at = o * tail + j;
            vdata[at] = bv;
            idata[at] = bi as f64;
        }
    }
    Ok(json!({
        "values": enc(Tensor { shape: vshape.clone(), data: vdata }),
        "indices": enc(Tensor { shape: vshape, data: idata }),
    }))
}

/// `kthvalue` — `aten::kthvalue(self, k, dim=-1, keepdim=False)` →
/// `{values, indices}`: the k-th smallest (1-based) along `dim`, via the
/// same stable sort `sort`/`topk` use.
pub(crate) fn k_kthvalue(
    t: &Tensor,
    k: i64,
    dim: i64,
    keepdim: bool,
) -> Result<serde_json::Value, EvalError> {
    let d = wrap_dim(dim, t.shape.len(), "kthvalue")?;
    let w = t.shape[d];
    if k < 1 || k as usize > w {
        return Err(op_err("kthvalue", format!("k={k} out of range [1, {w}]")));
    }
    let pos = (k - 1) as usize;
    let (vals, idxs) = sort_dim(t, dim, false, "kthvalue")?;
    let v = select_dim_index(&vals, d, pos, keepdim, "kthvalue")?;
    let i = select_dim_index(&idxs, d, pos, keepdim, "kthvalue")?;
    Ok(json!({ "values": enc(v), "indices": enc(i) }))
}

/// `meshgrid` — `aten::meshgrid(Tensor[])` with ij indexing: rank-1
/// inputs of lengths `L_i` produce `n` outputs of shape `[L_1..L_n]`
/// where output `i` varies along axis `i`. Rank>1 inputs error loudly
/// (the N-D broadcasting rules are a documented subset boundary).
pub(crate) fn k_meshgrid(ts: &[Tensor]) -> Result<serde_json::Value, EvalError> {
    if ts.is_empty() {
        return Err(op_err("meshgrid", "empty tensor list"));
    }
    let n = ts.len();
    for t in ts {
        if t.shape.len() != 1 {
            return Err(op_err(
                "meshgrid",
                format!("only rank-1 inputs supported, got shape {:?}", t.shape),
            ));
        }
    }
    let out_shape: Vec<usize> = ts.iter().map(|t| t.shape[0]).collect();
    let count: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut data = Vec::with_capacity(count);
        let mut idx = vec![0usize; n];
        for _ in 0..count {
            data.push(ts[i].data[idx[i]]);
            for d in (0..n).rev() {
                idx[d] += 1;
                if idx[d] < out_shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out.push(enc(Tensor {
            shape: out_shape.clone(),
            data,
        }));
    }
    Ok(Value::Array(out))
}

// ---------------------------------------------------------------------------
// einsum — the equation parser + general contraction engine
// ---------------------------------------------------------------------------

/// One parsed operand/output subscript. Without `...`, `labels` binds
/// dims in order; with `...`, `left` labels bind the leading dims,
/// `right` labels the trailing dims, and the middle `rank-l-r` dims are
/// the broadcast (ellipsis) dims.
struct EinsumSub {
    left: Vec<char>,
    right: Vec<char>,
    ell: bool,
}

fn parse_einsum_sub(s: &str) -> Result<EinsumSub, EvalError> {
    let bad = |c: char| !(c.is_ascii_alphanumeric() || c == '_');
    if let Some((pre, post)) = s.split_once("...") {
        let left: Vec<char> = pre.chars().filter(|c| !c.is_whitespace()).collect();
        let right: Vec<char> = post.chars().filter(|c| !c.is_whitespace()).collect();
        for &c in left.iter().chain(&right) {
            if bad(c) {
                return Err(op_err("einsum", format!("bad label {c:?} in {s:?}")));
            }
        }
        Ok(EinsumSub { left, right, ell: true })
    } else {
        let labs: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
        for &c in &labs {
            if bad(c) {
                return Err(op_err("einsum", format!("bad label {c:?} in {s:?}")));
            }
        }
        Ok(EinsumSub {
            left: labs,
            right: Vec::new(),
            ell: false,
        })
    }
}

/// Per-operand axis classification: a labeled dim or an ellipsis (batch)
/// dim — `Bat(j)` is the j-th of THIS operand's ellipsis dims, mapped to
/// a global batch axis once `batch_shape` is known.
enum EAxis {
    Lab(usize),
    Bat(usize),
}

struct EOperand {
    t: Tensor,
    axes: Vec<EAxis>,
    /// This operand's ellipsis dim count (for the local→global map).
    e: usize,
}

/// `einsum` — `aten::einsum(str equation, Tensor[] tensors, *, path?)`.
///
/// Grammar: `lhs->rhs` comma-separated operand subscripts; `->` may be
/// omitted (numpy convention: broadcast ellipsis dims first, then the
/// labels appearing exactly once in alphabetical order). `...` broadcasts
/// leading/middle/trailing unlabeled dims;
/// ellipsis dims are kept iff `...` appears in the output subscript,
/// otherwise summed with the contracted labels. A label repeated inside
/// one operand is rejected (no diagonal extraction). `path` is an
/// opt_einsum contraction-order hint — ignored, semantics unchanged.
/// Execution is the direct sum-of-products engine: for each output
/// index, sum over every contracted-index combination of the operand
/// product — O(out·contract·Σrank), the same naive-f64 spirit as the
/// rest of this file.
pub(crate) fn k_einsum(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use std::collections::HashMap;
    let eq = args
        .get(0)
        .and_then(|v| v.as_str())
        .ok_or_else(|| op_err("einsum", "equation must be a string"))?;
    let ts = req_tensors(arg_at(args, 1)?)?;
    let (lhs, rhs, has_arrow) = match eq.split_once("->") {
        Some((a, b)) => (a, b, true),
        None => (eq, "", false),
    };
    let subs: Vec<&str> = lhs.split(',').map(str::trim).collect();
    if subs.len() != ts.len() {
        return Err(op_err(
            "einsum",
            format!("equation has {} operands but {} tensors", subs.len(), ts.len()),
        ));
    }

    // Label table: char → id, with sizes recorded from operand dims.
    let mut label_id: HashMap<char, usize> = HashMap::new();
    let mut label_size: Vec<Option<usize>> = Vec::new();
    let mut label_chars: Vec<char> = Vec::new();
    let mut operands: Vec<EOperand> = Vec::with_capacity(ts.len());
    let mut batch_dims: Vec<Vec<usize>> = Vec::with_capacity(ts.len());
    let mut first_seen: Vec<usize> = Vec::new(); // label ids in first-appearance order

    for (sub_s, t) in subs.iter().zip(&ts) {
        let sub = parse_einsum_sub(sub_s)?;
        let r = t.shape.len();
        let mut axes = Vec::with_capacity(r);
        let mut local_labels: Vec<char> = Vec::new();
        if sub.ell {
            let lb = sub.left.len();
            let la = sub.right.len();
            if lb + la > r {
                return Err(op_err(
                    "einsum",
                    format!("subscript {sub_s:?} has {} labels for rank-{r} tensor", lb + la),
                ));
            }
            let e = r - lb - la;
            // dims [0,lb) → left labels; [r-la,r) → right labels; middle → batch.
            for d in 0..r {
                let c = if d < lb {
                    Some(sub.left[d])
                } else if d >= r - la {
                    Some(sub.right[d - (r - la)])
                } else {
                    None
                };
                match c {
                    Some(ch) => {
                        let id = *label_id.entry(ch).or_insert_with(|| {
                            label_size.push(None);
                            label_chars.push(ch);
                            first_seen.push(label_size.len() - 1);
                            label_size.len() - 1
                        });
                        let sz = t.shape[d];
                        match label_size[id] {
                            None => label_size[id] = Some(sz),
                            Some(s) if s != sz => {
                                return Err(op_err(
                                    "einsum",
                                    format!("label {ch:?} size mismatch {s} vs {sz}"),
                                ));
                            }
                            _ => {}
                        }
                        axes.push(EAxis::Lab(id));
                        local_labels.push(ch);
                    }
                    None => axes.push(EAxis::Bat(d - lb)),
                }
            }
            batch_dims.push(t.shape[lb..r - la].to_vec());
            operands.push(EOperand { t: t.clone(), axes, e });
        } else {
            let labs = &sub.left; // no ellipsis → all labels in `left`
            if labs.len() != r {
                return Err(op_err(
                    "einsum",
                    format!("subscript {sub_s:?} has {} labels for rank-{r} tensor", labs.len()),
                ));
            }
            for (d, &ch) in labs.iter().enumerate() {
                if local_labels.contains(&ch) {
                    return Err(op_err("einsum", format!("repeated label {ch:?} in {sub_s:?}")));
                }
                let id = *label_id.entry(ch).or_insert_with(|| {
                    label_size.push(None);
                    label_chars.push(ch);
                    first_seen.push(label_size.len() - 1);
                    label_size.len() - 1
                });
                let sz = t.shape[d];
                match label_size[id] {
                    None => label_size[id] = Some(sz),
                    Some(s) if s != sz => {
                        return Err(op_err(
                            "einsum",
                            format!("label {ch:?} size mismatch {s} vs {sz}"),
                        ));
                    }
                    _ => {}
                }
                axes.push(EAxis::Lab(id));
                local_labels.push(ch);
            }
            batch_dims.push(Vec::new());
            operands.push(EOperand { t: t.clone(), axes, e: 0 });
        }
    }

    // A label repeated inside ONE operand is diagonal extraction —
    // outside this engine's grammar; reject it for both branches.
    for op in &operands {
        let mut seen = std::collections::HashSet::new();
        for ax in &op.axes {
            if let EAxis::Lab(id) = ax {
                if !seen.insert(*id) {
                    return Err(op_err("einsum", "repeated label within one operand"));
                }
            }
        }
    }

    // Broadcast batch shape across operands (right-aligned).
    let mut batch_shape: Vec<usize> = Vec::new();
    for b in &batch_dims {
        batch_shape = broadcast_shape_pub(&batch_shape, b, "einsum")?;
    }
    let b_ndim = batch_shape.len();

    // Output axis spec in output order: kept labels and (when `...` is
    // written) the broadcast batch axes at the ellipsis position.
    enum OutAxis {
        Lab(usize),
        Bat(usize),
    }
    let mut out_axes: Vec<OutAxis> = Vec::new();
    let mut out_has_ell = false;
    if has_arrow {
        let sub = parse_einsum_sub(rhs)?;
        out_has_ell = sub.ell;
        let push_lab = |ch: char, out_axes: &mut Vec<OutAxis>| -> Result<(), EvalError> {
            match label_id.get(&ch) {
                Some(&id) => {
                    out_axes.push(OutAxis::Lab(id));
                    Ok(())
                }
                None => Err(op_err(
                    "einsum",
                    format!("output label {ch:?} does not appear in any operand"),
                )),
            }
        };
        for &ch in &sub.left {
            push_lab(ch, &mut out_axes)?;
        }
        if sub.ell {
            for b in 0..b_ndim {
                out_axes.push(OutAxis::Bat(b));
            }
            for &ch in &sub.right {
                push_lab(ch, &mut out_axes)?;
            }
        }
    } else {
        // Implicit output (numpy/torch convention): broadcast ellipsis
        // dims FIRST, then the labels appearing in exactly ONE operand
        // in alphabetical order.
        for b in 0..b_ndim {
            out_axes.push(OutAxis::Bat(b));
        }
        out_has_ell = true; // batch dims are kept, not summed
        let mut count = vec![0usize; label_size.len()];
        for op in &operands {
            let mut seen_here = std::collections::HashSet::new();
            for ax in &op.axes {
                if let EAxis::Lab(id) = ax {
                    if seen_here.insert(*id) {
                        count[*id] += 1;
                    }
                }
            }
        }
        let mut once: Vec<usize> = first_seen
            .iter()
            .copied()
            .filter(|&id| count[id] == 1)
            .collect();
        once.sort_by_key(|&id| label_chars[id]);
        for id in once {
            out_axes.push(OutAxis::Lab(id));
        }
    }

    // Positions: out_pos[id] = axis in the output index vector;
    // red_pos[id] = axis in the contraction index vector.
    let mut out_pos = vec![usize::MAX; label_size.len()];
    let mut batch_out_pos0 = usize::MAX;
    for (i, ax) in out_axes.iter().enumerate() {
        match ax {
            OutAxis::Lab(id) => out_pos[*id] = i,
            OutAxis::Bat(_) => {
                if batch_out_pos0 == usize::MAX {
                    batch_out_pos0 = i;
                }
            }
        }
    }
    // Contracted labels = operand labels absent from the output,
    // first-appearance order; then the batch axes when the output has
    // no `...` (they are summed like any non-output dim).
    let contract: Vec<usize> = first_seen
        .iter()
        .copied()
        .filter(|&id| out_pos[id] == usize::MAX)
        .collect();
    let mut red_pos = vec![usize::MAX; label_size.len()];
    for (i, &id) in contract.iter().enumerate() {
        red_pos[id] = i;
    }
    let n_contract = contract.len();
    let mut red_shape: Vec<usize> = contract.iter().map(|&id| label_size[id].unwrap()).collect();
    if !out_has_ell {
        red_shape.extend(batch_shape.iter().copied());
    }
    let mut out_shape: Vec<usize> = Vec::with_capacity(out_axes.len());
    for ax in &out_axes {
        match ax {
            OutAxis::Lab(id) => out_shape.push(label_size[*id].unwrap()),
            OutAxis::Bat(b) => out_shape.push(batch_shape[*b]),
        }
    }

    // Per-operand axis → index source + row-major stride.
    enum Src {
        O(usize),
        R(usize),
    }
    struct ExecOp<'a> {
        data: &'a [f64],
        srcs: Vec<Src>,
        strides: Vec<usize>,
    }
    let mut execs: Vec<ExecOp> = Vec::with_capacity(operands.len());
    for op in &operands {
        let mut strides = row_major_strides(&op.t.shape);
        let mut srcs = Vec::with_capacity(op.axes.len());
        for (d, ax) in op.axes.iter().enumerate() {
            srcs.push(match ax {
                EAxis::Lab(id) => {
                    if out_pos[*id] != usize::MAX {
                        Src::O(out_pos[*id])
                    } else {
                        Src::R(red_pos[*id])
                    }
                }
                EAxis::Bat(j) => {
                    // Local ellipsis dim → right-aligned global batch axis.
                    let b = b_ndim - op.e + *j;
                    if out_has_ell {
                        Src::O(batch_out_pos0 + b)
                    } else {
                        Src::R(n_contract + b)
                    }
                }
            });
            // Ellipsis dims broadcast: an extent-1 operand dim reads
            // index 0 regardless of the global batch coordinate.
            if matches!(ax, EAxis::Bat(_)) && op.t.shape[d] == 1 {
                strides[d] = 0;
            }
        }
        execs.push(ExecOp {
            data: &op.t.data,
            srcs,
            strides,
        });
    }

    // Sum-of-products engine: out[O] = Σ_R Π_i operand_i[offset_i(O,R)].
    let out_numel: usize = out_shape.iter().product();
    let red_numel: usize = red_shape.iter().product();
    let mut data = Vec::with_capacity(out_numel);
    let mut oi = vec![0usize; out_shape.len()];
    let mut ri = vec![0usize; red_shape.len()];
    for _ in 0..out_numel {
        let mut acc = 0.0;
        for _ in 0..red_numel {
            let mut prod = 1.0;
            for e in &execs {
                let mut off = 0usize;
                for (d, s) in e.srcs.iter().enumerate() {
                    let c = match s {
                        Src::O(p) => oi[*p],
                        Src::R(p) => ri[*p],
                    };
                    off += c * e.strides[d];
                }
                prod *= e.data[off];
            }
            acc += prod;
            for d in (0..red_shape.len()).rev() {
                ri[d] += 1;
                if ri[d] < red_shape[d] {
                    break;
                }
                ri[d] = 0;
            }
        }
        data.push(acc);
        for d in (0..out_shape.len()).rev() {
            oi[d] += 1;
            if oi[d] < out_shape[d] {
                break;
            }
            oi[d] = 0;
        }
    }
    Ok(enc(Tensor {
        shape: out_shape,
        data,
    }))
}

/// `searchsorted` — `aten::searchsorted.Tensor(sorted_sequence, self,
/// right?, side?)` (the `.Scalar` sibling is covered too — a numeric
/// `self` is treated as a single query). Searches the LAST dim of
/// `sorted_sequence`; `self` broadcasts against `sorted.shape[..-1]`
/// (extent-1 dims stretch, others must match exactly). `right` selects
/// upper_bound vs lower_bound; `side` ("left"/"right") overrides it.
/// `out_int32` is accepted and ignored (indices are always f64 on this
/// wire); a `sorter` argument errors — we don't honor precomputed order.
/// NaN compares greater than every number (sorted-last convention).
pub(crate) fn k_searchsorted(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let sorted = require_tensor(args, 0)?;
    let rank = sorted.shape.len();
    if rank == 0 {
        return Err(op_err("searchsorted", "sorted_sequence must have rank >= 1"));
    }
    let n = sorted.shape[rank - 1];
    let mut right = opt_bool(args.get(3), false)?;
    if let Some(v) = args.get(4) {
        if !v.is_null() {
            match v.as_str() {
                Some("right") => right = true,
                Some("left") => right = false,
                _ => {
                    return Err(op_err(
                        "searchsorted",
                        "side must be \"left\" or \"right\"",
                    ))
                }
            }
        }
    }
    if args.get(5).is_some_and(|v| !v.is_null()) {
        return Err(op_err("searchsorted", "sorter argument unsupported"));
    }

    // self: tensor (any broadcastable shape) or scalar.
    let self_t = match args.get(1) {
        Some(v) if v.is_number() => None,
        Some(v) => Some(decode_tensor(v)?),
        None => return Err(op_err("searchsorted", "missing self")),
    };
    let scalar_q = args.get(1).and_then(|v| v.as_f64());

    // Outer (all but last) dims of sorted; rows are contiguous.
    let outer_shape = &sorted.shape[..rank - 1];
    let outer_numel: usize = outer_shape.iter().product();

    // Binary search one row.
    let search = |row: &[f64], v: f64| -> f64 {
        let (mut lo, mut hi) = (0usize, row.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            let adv = if right {
                // upper_bound: first index with row[i] > v
                match row[mid].partial_cmp(&v) {
                    Some(std::cmp::Ordering::Greater) => false,
                    Some(_) => true,
                    None => v.is_nan(), // row NaN vs v: NaN is largest → row > v unless v NaN
                }
            } else {
                // lower_bound: first index with row[i] >= v
                match row[mid].partial_cmp(&v) {
                    Some(std::cmp::Ordering::Less) => true,
                    Some(_) => false,
                    None => !v.is_nan(), // row NaN: row >= v unless v is NaN too
                }
            };
            if adv {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo as f64
    };

    match self_t {
        None => {
            // Scalar query: one index per outer position.
            let v = scalar_q.unwrap_or(f64::NAN);
            let mut data = Vec::with_capacity(outer_numel);
            for o in 0..outer_numel {
                // Row-major: outer index o selects the contiguous row.
                data.push(search(&sorted.data[o * n..(o + 1) * n], v));
            }
            Ok(enc(Tensor {
                shape: outer_shape.to_vec(),
                data,
            }))
        }
        Some(st) => {
            let rs = st.shape.len();
            if rs == 0 || rs > rank {
                return Err(op_err(
                    "searchsorted",
                    format!("self rank {rs} incompatible with sorted rank {rank}"),
                ));
            }
            let m = st.shape[rs - 1];
            // self outer dims must broadcast to sorted outer dims
            // (right-aligned; extent-1 stretches).
            let s_outer = &st.shape[..rs - 1];
            let s_outer_rank = s_outer.len();
            let out_rank = rank - 1;
            if s_outer_rank > out_rank {
                return Err(op_err("searchsorted", "self has too many outer dims"));
            }
            for i in 0..out_rank {
                let sd = if i < out_rank - s_outer_rank {
                    1
                } else {
                    s_outer[i - (out_rank - s_outer_rank)]
                };
                let od = outer_shape[i];
                if sd != 1 && sd != od {
                    return Err(op_err(
                        "searchsorted",
                        format!("self outer dim {sd} not broadcastable to {od}"),
                    ));
                }
            }
            let mut out_shape = outer_shape.to_vec();
            out_shape.push(m);
            let out_numel: usize = out_shape.iter().product();
            // self outer strides (row-major over its own outer dims).
            let s_outer_strides = row_major_strides(s_outer);
            let mut data = vec![0.0; out_numel];
            for o in 0..outer_numel {
                // self outer offset for this sorted-outer position:
                // decompose o into outer coords, map through broadcast.
                let row = &sorted.data[o * n..(o + 1) * n];
                // coords of o
                let mut coords = vec![0usize; out_rank];
                let mut rem = o;
                for d in (0..out_rank).rev() {
                    coords[d] = rem % outer_shape[d];
                    rem /= outer_shape[d];
                }
                // self outer offset
                let mut soff = 0usize;
                for i in 0..out_rank {
                    let sd = if i < out_rank - s_outer_rank {
                        1
                    } else {
                        s_outer[i - (out_rank - s_outer_rank)]
                    };
                    let c = if sd == 1 { 0 } else { coords[i] };
                    if i >= out_rank - s_outer_rank {
                        soff += c * s_outer_strides[i - (out_rank - s_outer_rank)];
                    }
                }
                for q in 0..m {
                    let v = st.data[soff * m + q];
                    data[o * m + q] = search(row, v);
                }
            }
            Ok(enc(Tensor {
                shape: out_shape,
                data,
            }))
        }
    }
}

/// `quantile` — `aten::quantile(self, q, dim?, keepdim?, interpolation)`.
/// Only `"linear"` interpolation is supported (others error loudly);
/// `q` may be a tensor (any shape — its shape leads the output) or a
/// scalar. `dim=None` flattens all axes → output shape = q.shape;
/// `dim=d` sorts along d → output = q.shape ++ kept dims (keepdim
/// retains the axis as size-1). q values must lie in [0, 1].
pub(crate) fn k_quantile(args: &[serde_json::Value]) -> Result<serde_json::Value, EvalError> {
    use super::codec::require_tensor;
    let t = require_tensor(args, 0)?;
    // q: tensor or scalar → (q_shape, q_data).
    let (q_shape, q_data): (Vec<usize>, Vec<f64>) = match args.get(1) {
        Some(v) if v.is_number() => (vec![], vec![req_f64(v)?]),
        Some(v) => {
            let q = decode_tensor(v)?;
            (q.shape.clone(), q.data.clone())
        }
        None => return Err(op_err("quantile", "missing q")),
    };
    for &q in &q_data {
        if !(0.0..=1.0).contains(&q) {
            return Err(op_err("quantile", format!("q={q} outside [0, 1]")));
        }
    }
    // interpolation: string, default "linear" — only linear implemented.
    if let Some(v) = args.get(4) {
        if !v.is_null() {
            match v.as_str() {
                Some("linear") | None => {}
                Some(other) => {
                    return Err(op_err(
                        "quantile",
                        format!("interpolation {other:?} unsupported (only \"linear\")"),
                    ))
                }
            }
        }
    }
    // dim: absent/null → flatten; int (or 1-elem list) → single axis.
    let dim: Option<i64> = match args.get(2) {
        None => None,
        Some(v) if v.is_null() => None,
        Some(v) if v.is_array() => {
            let ds = req_i64s(v)?;
            if ds.len() != 1 {
                return Err(op_err("quantile", "dim must be a single int"));
            }
            Some(ds[0])
        }
        Some(v) => Some(req_i64(v)?),
    };
    let keepdim = opt_bool(args.get(3), false)?;

    // Per-q interpolation knots: h = q*(n-1) → (i0, i1, frac).
    let knots = |n: usize| -> Vec<(usize, usize, f64)> {
        q_data
            .iter()
            .map(|&q| {
                let h = q * (n.saturating_sub(1)) as f64;
                let i0 = (h.floor() as usize).min(n.saturating_sub(1));
                let i1 = (h.ceil() as usize).min(n.saturating_sub(1));
                (i0, i1, h - h.floor())
            })
            .collect()
    };

    match dim {
        None => {
            if t.data.is_empty() {
                return Err(op_err("quantile", "empty tensor"));
            }
            let mut s = t.data.clone();
            s.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            let ks = knots(s.len());
            let data: Vec<f64> = ks
                .iter()
                .map(|&(i0, i1, f)| s[i0] + f * (s[i1] - s[i0]))
                .collect();
            Ok(enc(Tensor {
                shape: q_shape,
                data,
            }))
        }
        Some(d0) => {
            let d = wrap_dim(d0, t.shape.len(), "quantile")?;
            let w = t.shape[d];
            if w == 0 {
                return Err(op_err("quantile", "empty dim"));
            }
            let (vals, _) = sort_dim(&t, d0, false, "quantile")?;
            let tail: usize = t.shape[d + 1..].iter().product();
            let outer: usize = t.shape[..d].iter().product();
            let kept_numel = t.data.len() / w;
            let ks = knots(w);
            // kept shape (q.shape leads).
            let mut kept_shape: Vec<usize> = t
                .shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != d)
                .map(|(_, &s)| s)
                .collect();
            if keepdim {
                kept_shape.insert(d, 1);
            }
            let mut out_shape = q_shape.clone();
            out_shape.extend(kept_shape);
            let q_numel: usize = q_shape.iter().product();
            let mut data = vec![0.0; q_numel * kept_numel];
            for (qi, &(i0, i1, f)) in ks.iter().enumerate() {
                for o in 0..outer {
                    for j in 0..tail {
                        let at = |i: usize| (o * w + i) * tail + j;
                        let v = vals.data[at(i0)] + f * (vals.data[at(i1)] - vals.data[at(i0)]);
                        data[qi * kept_numel + o * tail + j] = v;
                    }
                }
            }
            Ok(enc(Tensor {
                shape: out_shape,
                data,
            }))
        }
    }
}

/// `count_nonzero` — `aten::count_nonzero(self, dim?)`: count of `x != 0`.
/// `dim=None` → rank-0 total; a given dim keeps size-1 (torch's
/// count_nonzero is keepdim-when-dim-given).
pub(crate) fn k_count_nonzero(t: &Tensor, dim: Option<i64>) -> Result<Tensor, EvalError> {
    let mask = map_unary(t, |x| if x != 0.0 { 1.0 } else { 0.0 });
    match dim {
        None => reduce_sum(&mask, None, false),
        Some(d) => reduce_sum(&mask, Some(vec![d]), true),
    }
}

/// `dist` — `aten::dist(self, other, p=2)`: `‖a−b‖_p` over all elements
/// (broadcast operands) → rank-0.
pub(crate) fn k_dist(a: &Tensor, b: &Tensor, p: f64) -> Result<Tensor, EvalError> {
    let d = broadcast_elementwise(a, b, "tensor.dist", |x, y| x - y)?;
    p_norm(&d, None, p, false)
}

/// `allclose` — `aten::allclose(self, other, rtol, atol, equal_nan)` →
/// bool: every broadcast pair satisfies `|a−b| <= atol + rtol·|b|`
/// (or both NaN when `equal_nan`).
pub(crate) fn k_allclose(
    a: &Tensor,
    b: &Tensor,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> Result<serde_json::Value, EvalError> {
    let d = broadcast_elementwise(a, b, "tensor.allclose", |x, y| {
        let ok = (x - y).abs() <= atol + rtol * y.abs()
            || (equal_nan && x.is_nan() && y.is_nan());
        if ok { 1.0 } else { 0.0 }
    })?;
    Ok(json!(d.data.iter().all(|&v| v != 0.0)))
}

/// `is_nonzero` — `aten::is_nonzero(self)` → any element `!= 0`.
pub(crate) fn k_is_nonzero(t: &Tensor) -> Result<serde_json::Value, EvalError> {
    Ok(json!(t.data.iter().any(|&x| x != 0.0)))
}

/// `msort` — `aten::msort(self)`: sort the last dim ascending, values
/// only (the values half of `sort`).
pub(crate) fn k_msort(t: &Tensor) -> Result<Tensor, EvalError> {
    Ok(sort_dim(t, -1, false, "msort")?.0)
}

fn nan_zero(t: &Tensor) -> Tensor {
    map_unary(t, |x| if x.is_nan() { 0.0 } else { x })
}

fn nan_mask(t: &Tensor) -> Tensor {
    map_unary(t, |x| if x.is_nan() { 0.0 } else { 1.0 })
}

/// `nansum` — `aten::nansum(self, dim?, keepdim?)`: NaN treated as 0.
pub(crate) fn k_nansum(
    t: &Tensor,
    dims: Option<&[i64]>,
    keepdim: bool,
) -> Result<Tensor, EvalError> {
    reduce_sum(&nan_zero(t), dims.map(|d| d.to_vec()), keepdim)
}

/// `nanmean` — `aten::nanmean(self, dim?, keepdim?)`: mean over
/// non-NaN elements; an all-NaN slice yields NaN.
pub(crate) fn k_nanmean(
    t: &Tensor,
    dims: Option<&[i64]>,
    keepdim: bool,
) -> Result<Tensor, EvalError> {
    let s = reduce_sum(&nan_zero(t), dims.map(|d| d.to_vec()), keepdim)?;
    let c = reduce_sum(&nan_mask(t), dims.map(|d| d.to_vec()), keepdim)?;
    Ok(Tensor {
        shape: s.shape.clone(),
        data: s
            .data
            .iter()
            .zip(&c.data)
            .map(|(&a, &b)| if b == 0.0 { f64::NAN } else { a / b })
            .collect(),
    })
}

/// p-distance between two equal-length rows: `(Σ|a−b|^p)^(1/p)`;
/// `p = ∞` → max|a−b|.
fn p_dist_rows(a: &[f64], b: &[f64], p: f64) -> f64 {
    if p == f64::INFINITY {
        return a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0, f64::max);
    }
    let s: f64 = a.iter().zip(b).map(|(&x, &y)| (x - y).abs().powf(p)).sum();
    s.powf(1.0 / p)
}

/// `cdist` — `aten::cdist(x1, x2, p=2, compute_mode?)`: pairwise
/// p-distances. Rank-2 `(B,P)×(R,P) → (B,R)`; rank-3 `(N,B,P)×(N,R,P)
/// → (N,B,R)` with strictly-equal batch dims (torch broadcasts batch
/// dims — a documented subset boundary). `compute_mode` is a perf hint,
/// accepted and ignored.
pub(crate) fn k_cdist(
    x1: &Tensor,
    x2: &Tensor,
    p: f64,
    _mode: Option<i64>,
) -> Result<Tensor, EvalError> {
    match (x1.shape.len(), x2.shape.len()) {
        (2, 2) => {
            let (b1, p1, r2, p2) = (x1.shape[0], x1.shape[1], x2.shape[0], x2.shape[1]);
            if p1 != p2 {
                return Err(op_err("cdist", format!("feature dim mismatch {p1} vs {p2}")));
            }
            let mut data = vec![0.0; b1 * r2];
            for i in 0..b1 {
                for j in 0..r2 {
                    data[i * r2 + j] = p_dist_rows(
                        &x1.data[i * p1..(i + 1) * p1],
                        &x2.data[j * p2..(j + 1) * p2],
                        p,
                    );
                }
            }
            Ok(Tensor { shape: vec![b1, r2], data })
        }
        (3, 3) => {
            let (nb, b1, f1, rb, f2) = (x1.shape[0], x1.shape[1], x1.shape[2], x2.shape[1], x2.shape[2]);
            if x1.shape[0] != x2.shape[0] || f1 != f2 {
                return Err(op_err("cdist", "batched cdist requires equal batch and feature dims"));
            }
            let _ = rb;
            let mut data = vec![0.0; nb * b1 * x2.shape[1]];
            for n in 0..nb {
                for i in 0..b1 {
                    for j in 0..x2.shape[1] {
                        data[(n * b1 + i) * x2.shape[1] + j] = p_dist_rows(
                            &x1.data[(n * b1 + i) * f1..(n * b1 + i + 1) * f1],
                            &x2.data[(n * x2.shape[1] + j) * f2..(n * x2.shape[1] + j + 1) * f2],
                            p,
                        );
                    }
                }
            }
            Ok(Tensor {
                shape: vec![nb, b1, x2.shape[1]],
                data,
            })
        }
        _ => Err(op_err(
            "cdist",
            format!("expected rank-2 or rank-3 operands, got {:?} x {:?}", x1.shape, x2.shape),
        )),
    }
}

/// `pdist` — `aten::pdist(self, p=2)`: condensed pairwise distances —
/// rank-2 `(B,P) → (B(B−1)/2,)`, rank-3 `(N,B,P) → (N, B(B−1)/2)`.
pub(crate) fn k_pdist(t: &Tensor, p: f64) -> Result<Tensor, EvalError> {
    let pairs = |b: usize| b * b.saturating_sub(1) / 2;
    // Emit condensed distances for one (…, B, P) slab; `base` is the
    // flat offset of the slab in `t.data`.
    let slab = |t: &Tensor, base: usize, b: usize, f: usize, out: &mut Vec<f64>| {
        for i in 0..b {
            for j in (i + 1)..b {
                out.push(p_dist_rows(
                    &t.data[base + i * f..base + (i + 1) * f],
                    &t.data[base + j * f..base + (j + 1) * f],
                    p,
                ));
            }
        }
    };
    match t.shape.len() {
        2 => {
            let (b, f) = (t.shape[0], t.shape[1]);
            let mut data = Vec::with_capacity(pairs(b));
            slab(t, 0, b, f, &mut data);
            Ok(Tensor { shape: vec![data.len()], data })
        }
        3 => {
            let (nb, b, f) = (t.shape[0], t.shape[1], t.shape[2]);
            let per = pairs(b);
            let mut data = Vec::with_capacity(nb * per);
            for n in 0..nb {
                slab(t, n * b * f, b, f, &mut data);
            }
            Ok(Tensor {
                shape: vec![nb, per],
                data,
            })
        }
        r => Err(op_err("pdist", format!("expected rank 2 or 3, got {r}"))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::testutil::tensor;
    use serde_json::json;

    fn t(shape: &[usize], data: &[f64]) -> Tensor {
        decode_tensor(&tensor(shape, data)).unwrap()
    }

    #[test]
    fn unary_and_binary_smoke() {
        let a = t(&[2], &[1.0, 4.0]);
        assert_eq!(k_abs(&a).unwrap().data, vec![1.0, 4.0]);
        assert_eq!(k_sqrt(&a).unwrap().data, vec![1.0, 2.0]);
        let b = t(&[2], &[2.0, 2.0]);
        assert_eq!(k_atan2(&a, &b).unwrap().data[0], 1.0f64.atan2(2.0));
    }

    #[test]
    fn reduce_sum_mean_dims() {
        let a = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = reduce_sum(&a, Some(vec![1]), false).unwrap();
        assert_eq!(s.shape, vec![2]);
        assert_eq!(s.data, vec![6.0, 15.0]);
        let m = reduce_mean(&a, None, false).unwrap();
        assert_eq!(m.shape, Vec::<usize>::new());
        assert_eq!(m.data, vec![3.5]);
        let sk = reduce_sum(&a, Some(vec![1]), true).unwrap();
        assert_eq!(sk.shape, vec![2, 1]);
        // empty dim list → identity
        let e = reduce_sum(&a, Some(vec![]), false).unwrap();
        assert_eq!(e.shape, a.shape);
    }

    #[test]
    fn softmax_sums_to_one() {
        let a = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let s = softmax_dim(&a, 1, false, "softmax").unwrap();
        let row0: f64 = s.data[0..2].iter().sum();
        assert!((row0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn argmax_f64_indices() {
        let a = t(&[2, 3], &[1.0, 5.0, 2.0, 0.0, 3.0, 1.0]);
        let idx = reduce_arg(&a, Some(1), false, true, "argmax").unwrap();
        assert_eq!(idx.data, vec![1.0, 1.0]);
        let flat = reduce_arg(&a, None, false, true, "argmax").unwrap();
        assert_eq!(flat.data, vec![1.0]);
    }

    #[test]
    fn shape_kernels() {
        let a = t(&[2, 3, 4], &(0..24).map(|x| x as f64).collect::<Vec<_>>());
        assert_eq!(k_flatten(&a, 1, 2).unwrap().shape, vec![2, 12]);
        assert_eq!(k_permute(&a, &[2, 0, 1]).unwrap().shape, vec![4, 2, 3]);
        assert_eq!(k_expand(&t(&[3], &[1.0, 2.0, 3.0]), &[2, 3]).unwrap().shape, vec![2, 3]);
        let un = k_unbind(&[encode_tensor(t(&[2, 2], &[1.0, 2.0, 3.0, 4.0]))]).unwrap();
        assert_eq!(un.as_array().unwrap().len(), 2);
    }

    #[test]
    fn linalg_smoke() {
        let a = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = t(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        assert_eq!(k_mm(&a, &b).unwrap().data, vec![58.0, 64.0, 139.0, 154.0]);
        let v = t(&[3], &[1.0, 1.0, 1.0]);
        assert_eq!(k_mv(&a, &v).unwrap().data, vec![6.0, 15.0]);
        assert_eq!(k_dot(&t(&[2], &[1.0, 2.0]), &t(&[2], &[3.0, 4.0])).unwrap().data, vec![11.0]);
    }

    #[test]
    fn constructors_and_metadata() {
        assert_eq!(k_full(&[2, 2], 3.0).unwrap().data, vec![3.0; 4]);
        let a = t(&[2, 3], &[1.0; 6]);
        assert_eq!(k_size(&[encode_tensor(a.clone())]).unwrap(), json!([2, 3]));
        assert_eq!(k_dim(&a), json!(2));
        assert_eq!(k_numel(&a), json!(6));
    }

    // --- seam closures -------------------------------------------------

    fn tv(shape: &[usize], data: &[f64]) -> Value {
        tensor(shape, data)
    }

    #[test]
    fn einsum_matmul_and_implicit() {
        // "ij,jk->ik" == mm.
        let a = tv(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tv(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let out = k_einsum(&[json!("ij,jk->ik"), json!([a.clone(), b.clone()])]).unwrap();
        assert_eq!(decode_tensor(&out).unwrap().data, vec![58.0, 64.0, 139.0, 154.0]);
        // Implicit output (numpy rule): once-labels alphabetical →
        // 'hi', i.e. (A@B)ᵀ layout — the known np.einsum gotcha.
        let out = k_einsum(&[json!("ij,jh"), json!([a.clone(), b.clone()])]).unwrap();
        let r = decode_tensor(&out).unwrap();
        assert_eq!(r.shape, vec![2, 2]);
        assert_eq!(r.data, vec![58.0, 139.0, 64.0, 154.0]);
        // Transpose via explicit output.
        let a = tv(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = k_einsum(&[json!("ij->ji"), json!([a])]).unwrap();
        let tt = decode_tensor(&out).unwrap();
        assert_eq!(tt.shape, vec![3, 2]);
        assert_eq!(tt.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // Three-operand chain: "ij,jk,kl->il".
        let c = tv(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let out = k_einsum(&[json!("ij,jk,kl->il"), json!([a, b, c])]).unwrap();
        let r = decode_tensor(&out).unwrap();
        assert_eq!(r.shape, vec![2, 2]);
        // a·b = [[58,64],[139,154]]; ·I = same.
        assert_eq!(r.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn einsum_ellipsis_batch() {
        // bmm via "...ij,jk->...ik": (2,2,3) x (3,2) → (2,2,2).
        let a_data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let a = tv(&[2, 2, 3], &a_data);
        let b = tv(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let out = k_einsum(&[json!("...ij,jk->...ik"), json!([a, b])]).unwrap();
        let r = decode_tensor(&out).unwrap();
        assert_eq!(r.shape, vec![2, 2, 2]);
        // Batch 0: rows [0,1,2]·B and [3,4,5]·B where B=[[1,0],[0,1],[1,1]]
        // out[0,0,:] = [0*1+1*0+2*1, 0*0+1*1+2*1] = [2,3]
        assert_eq!(r.data[0..2], vec![2.0, 3.0]);
        // Implicit ellipsis-sum: "...i,...i" → per-batch dot → shape [2].
        let x = tv(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = tv(&[2, 3], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let out = k_einsum(&[json!("...i,...i"), json!([x, y])]).unwrap();
        let r = decode_tensor(&out).unwrap();
        assert_eq!(r.shape, vec![2]);
        assert_eq!(r.data, vec![6.0, 15.0]);
    }

    #[test]
    fn einsum_errors() {
        // Repeated label in one operand → rejected (no diagonal).
        let m = tv(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert!(k_einsum(&[json!("ii->"), json!([m.clone()])]).is_err());
        // Output label not in inputs.
        assert!(k_einsum(&[json!("ij->ik"), json!([m])]).is_err());
        // Operand count mismatch.
        assert!(k_einsum(&[json!("ij,jk->ik"), json!([m])]).is_err());
        // Label size mismatch across operands.
        let a = tv(&[2, 3], &[0.0; 6]);
        let b = tv(&[4, 2], &[0.0; 8]);
        assert!(k_einsum(&[json!("ij,jk->ik"), json!([a, b])]).is_err());
    }

    #[test]
    fn tensordot_kron_cross() {
        let a = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = t(&[3, 4], &(0..12).map(|x| x as f64 + 1.0).collect::<Vec<_>>());
        // contract last of a with first of b → (2,4) == mm.
        let td = k_tensordot(&a, &b, &vec![1], &vec![0]).unwrap();
        assert_eq!(td.shape, vec![2, 4]);
        assert_eq!(td.data, k_mm(&a, &b).unwrap().data);
        // kron: (2,)⊗(3,) → (6,)
        let u = t(&[2], &[1.0, 2.0]);
        let v = t(&[3], &[3.0, 4.0, 5.0]);
        assert_eq!(k_kron(&u, &v).unwrap().data, vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
        // cross: [1,0,0]×[0,1,0] = [0,0,1]
        let e1 = t(&[3], &[1.0, 0.0, 0.0]);
        let e2 = t(&[3], &[0.0, 1.0, 0.0]);
        assert_eq!(k_cross(&e1, &e2, None).unwrap().data, vec![0.0, 0.0, 1.0]);
        // cross along dim 1 of (2,3)
        let m1 = t(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let m2 = t(&[2, 3], &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        let c = k_cross(&m1, &m2, Some(1)).unwrap();
        assert_eq!(c.shape, vec![2, 3]);
        assert_eq!(c.data, vec![0.0, 0.0, 1.0, 0.0, 0.0, -1.0]);
    }

    #[test]
    fn norms_renorm_normalize() {
        let a = t(&[3], &[3.0, 4.0, 0.0]);
        assert_eq!(k_norm(&a, 2.0).unwrap().data, vec![5.0]);
        assert_eq!(k_norm(&a, 1.0).unwrap().data, vec![7.0]);
        assert_eq!(k_norm(&a, f64::INFINITY).unwrap().data, vec![4.0]);
        // renorm: p=2, dim=0? use dim -1 on (2,2): row norms.
        let m = t(&[2, 2], &[3.0, 4.0, 1.0, 1.0]);
        let r = k_renorm(&m, 2.0, 1, 2.5).unwrap();
        // row0 norm 5 → scale 0.5 → [1.5,2]; row1 norm √2 → scale 1 (2.5/1.414>1)
        assert_eq!(r.data[0..2], vec![1.5, 2.0]);
        assert!((r.data[2] - 1.0).abs() < 1e-12);
        // normalize: x/max(||x||,eps), dim=1 keepdim internal.
        let n = k_normalize(&a, 2.0, Some(&[0]), false).unwrap();
        assert_eq!(n.shape, vec![3]);
        assert!((n.data[0] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn tuple_reductions() {
        let a = t(&[4], &[3.0, 1.0, 4.0, 1.0]);
        // median flat → lower median = 2nd smallest = 1.0
        let med = k_median(&[encode_tensor(a.clone())]).unwrap();
        assert_eq!(decode_tensor(&med).unwrap().data, vec![1.0]);
        // mode dim=-1 → most frequent 1.0 at index 1.
        let mo = k_mode(&a, -1, false).unwrap();
        assert_eq!(mo["values"], encode_tensor(t(&[], &[1.0])));
        assert_eq!(mo["indices"], encode_tensor(t(&[], &[1.0])));
        // kthvalue k=3 → 3.0 at sorted pos 2 → index 0.
        let kv = k_kthvalue(&a, 3, -1, false).unwrap();
        assert_eq!(kv["values"], encode_tensor(t(&[], &[3.0])));
        assert_eq!(kv["indices"], encode_tensor(t(&[], &[0.0])));
    }

    #[test]
    fn meshgrid_bincount_corrcoef() {
        let xs = tv(&[2], &[1.0, 2.0]);
        let ys = tv(&[3], &[10.0, 20.0, 30.0]);
        let mg = k_meshgrid(&[decode_tensor(&xs).unwrap(), decode_tensor(&ys).unwrap()]).unwrap();
        let arr = mg.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        let g0 = decode_tensor(&arr[0]).unwrap();
        let g1 = decode_tensor(&arr[1]).unwrap();
        assert_eq!(g0.shape, vec![2, 3]);
        assert_eq!(g0.data, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        assert_eq!(g1.data, vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
        // bincount
        let bc = k_bincount(&t(&[4], &[0.0, 1.0, 1.0, 3.0]), None, 4).unwrap();
        assert_eq!(bc.data, vec![1.0, 2.0, 0.0, 1.0]);
        // corrcoef of identical rows → 1.0
        let c = k_corrcoef(&t(&[2, 3], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0])).unwrap();
        assert!((c.data[0] - 1.0).abs() < 1e-12);
        assert!((c.data[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn to_dtype_cast() {
        let a = t(&[2], &[1.0, 2.0]);
        // f64-family dtype names → identity.
        let out = k_to(&[encode_tensor(a.clone()), json!("float64")]).unwrap();
        assert_eq!(decode_tensor(&out).unwrap().data, vec![1.0, 2.0]);
        let out = k_to(&[encode_tensor(a.clone()), json!("double")]).unwrap();
        assert_eq!(decode_tensor(&out).unwrap().data, vec![1.0, 2.0]);
        // Other dtypes error loudly.
        assert!(k_to(&[encode_tensor(a.clone()), json!("float32")]).is_err());
        assert!(k_to(&[encode_tensor(a.clone()), json!("int64")]).is_err());
        // Tensor rhs → same-dtype copy.
        let out = k_to(&[encode_tensor(a.clone()), encode_tensor(t(&[2], &[9.0, 9.0]))]).unwrap();
        assert_eq!(decode_tensor(&out).unwrap().data, vec![1.0, 2.0]);
    }

    #[test]
    fn searchsorted_quantile() {
        // sorted [1,3,5,7], queries [0,3,4,8] → [0,1,2,4]
        let s = t(&[4], &[1.0, 3.0, 5.0, 7.0]);
        let q = t(&[4], &[0.0, 3.0, 4.0, 8.0]);
        let out = k_searchsorted(&[encode_tensor(s.clone()), encode_tensor(q), json!(false), json!(false), Value::Null, Value::Null]).unwrap();
        assert_eq!(decode_tensor(&out).unwrap().data, vec![0.0, 1.0, 2.0, 4.0]);
        // right=true (upper_bound): [0,3,4,8] → [0,2,2,4]
        let out = k_searchsorted(&[encode_tensor(s.clone()), encode_tensor(t(&[4], &[0.0, 3.0, 4.0, 8.0])), json!(false), json!(true), Value::Null, Value::Null]).unwrap();
        assert_eq!(decode_tensor(&out).unwrap().data, vec![0.0, 2.0, 2.0, 4.0]);
        // quantile flat, q=0.5 of [0,10,20,30] → 15
        let x = t(&[4], &[0.0, 10.0, 20.0, 30.0]);
        let out = k_quantile(&[encode_tensor(x), json!(0.5), Value::Null, json!(false), json!("linear")]).unwrap();
        assert_eq!(decode_tensor(&out).unwrap().data, vec![15.0]);
        // non-linear interpolation errors.
        let x = t(&[4], &[0.0, 10.0, 20.0, 30.0]);
        assert!(k_quantile(&[encode_tensor(x), json!(0.5), Value::Null, json!(false), json!("nearest")]).is_err());
    }

    #[test]
    fn misc_promotions() {
        let a = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        // max.dim → {values,indices}
        let mx = k_max_min(&[encode_tensor(a.clone()), json!(1), json!(false)], true).unwrap();
        assert_eq!(mx["values"], encode_tensor(t(&[2], &[2.0, 4.0])));
        assert_eq!(mx["indices"], encode_tensor(t(&[2], &[1.0, 1.0])));
        // count_nonzero
        assert_eq!(k_count_nonzero(&t(&[3], &[0.0, 1.0, -2.0]), None).unwrap().data, vec![2.0]);
        // inner == dot
        assert_eq!(k_dot(&t(&[2], &[1.0, 2.0]), &t(&[2], &[3.0, 4.0])).unwrap().data, vec![11.0]);
        // dist p=2 of [0,0] vs [3,4] → 5
        assert_eq!(k_dist(&t(&[2], &[0.0, 0.0]), &t(&[2], &[3.0, 4.0]), 2.0).unwrap().data, vec![5.0]);
        // allclose
        assert_eq!(k_allclose(&t(&[2], &[1.0, 2.0]), &t(&[2], &[1.0, 2.0000001]), 1e-5, 1e-8, false).unwrap(), json!(true));
        // is_nonzero
        assert_eq!(k_is_nonzero(&t(&[2], &[0.0, 0.0])).unwrap(), json!(false));
        // msort == sort values
        let s = t(&[3], &[3.0, 1.0, 2.0]);
        assert_eq!(k_msort(&s).unwrap().data, vec![1.0, 2.0, 3.0]);
        // nansum / nanmean (NaN can't ride the JSON wire — build directly)
        let nan = Tensor { shape: vec![3], data: vec![1.0, f64::NAN, 3.0] };
        assert_eq!(k_nansum(&nan, None, false).unwrap().data, vec![4.0]);
        assert_eq!(k_nanmean(&nan, None, false).unwrap().data, vec![2.0]);
        // cdist / pdist
        let x1 = t(&[2, 2], &[0.0, 0.0, 3.0, 4.0]);
        let x2 = t(&[1, 2], &[0.0, 0.0]);
        let cd = k_cdist(&x1, &x2, 2.0, None).unwrap();
        assert_eq!(cd.shape, vec![2, 1]);
        assert!((cd.data[1] - 5.0).abs() < 1e-12);
        let pd = k_pdist(&x1, 2.0).unwrap();
        assert_eq!(pd.shape, vec![1]);
        assert!((pd.data[0] - 5.0).abs() < 1e-12);
    }
}
