//! `add` / `mul` / `sub` / `div` with numpy broadcasting — the impl
//! mirrored by lib-tensor's `elementwise.hnl`.

use handl_platform::{EvalContext, EvalError, Function};
use serde_json::{Value, json};

use super::codec::{Tensor, encode_tensor, require_tensor, row_major_strides};

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
    let inner = row_major_strides(shape);
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

// ---------------------------------------------------------------------------
// Backward (adjoints of the broadcast)
// ---------------------------------------------------------------------------

/// Expand `t` to `out_shape` by materializing the broadcast — each `1`
/// (or missing leading) dim's single element repeats across the stretch.
/// `out_shape` must be a broadcast of `t`'s shape.
pub(crate) fn broadcast_to(t: &Tensor, out_shape: &[usize]) -> Tensor {
    let s = broadcast_strides(&t.shape, out_shape);
    let rank = out_shape.len();
    let mut data = Vec::with_capacity(out_shape.iter().product());
    let mut idx = vec![0usize; rank];
    for _ in 0..out_shape.iter().product::<usize>() {
        let off: usize = idx.iter().zip(&s).map(|(&i, &s)| i * s).sum();
        data.push(t.data[off]);
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Tensor {
        shape: out_shape.to_vec(),
        data,
    }
}

/// Sum `g` back down to `target` — the adjoint of broadcasting: dims where
/// `target` is `1` but `g`'s is larger, and leading dims `target` lacks,
/// are summed out. `target` must be broadcast-compatible under `g`'s
/// shape (each aligned pair equal, or `target` `1`).
pub(crate) fn unbroadcast(g: &Tensor, target: &[usize]) -> Result<Tensor, EvalError> {
    if target.len() > g.shape.len() {
        return Err(EvalError::Other(format!(
            "unbroadcast: target {target:?} has higher rank than {:?}",
            g.shape
        )));
    }
    let off = g.shape.len() - target.len();
    for (i, &t) in target.iter().enumerate() {
        let d = g.shape[off + i];
        if t != d && t != 1 {
            return Err(EvalError::Other(format!(
                "unbroadcast: target {target:?} is not broadcast-compatible \
                 with {:?} (axis {i}: {t} vs {d})",
                g.shape
            )));
        }
    }
    let t_strides = row_major_strides(target);
    let mut data = vec![0.0; target.iter().product()];
    let rank = g.shape.len();
    let mut idx = vec![0usize; rank];
    for &x in &g.data {
        let mut tidx = 0usize;
        for (d, &i) in idx.iter().enumerate() {
            if d >= off && target[d - off] != 1 {
                tidx += i * t_strides[d - off];
            }
        }
        data[tidx] += x;
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < g.shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(Tensor {
        shape: target.to_vec(),
        data,
    })
}

/// The driver behind the elementwise adjoints: `g` must be the primal
/// output's shape (the broadcast of `a` and `b`); each input's gradient
/// is its per-element expression over the broadcast space — `fa`/`fb`
/// take `(a_elem, b_elem, g_elem)` — then [`unbroadcast`] back to the
/// input's shape. Returns `{"a": da, "b": db}` keyed by operand.
fn elementwise_backward(
    a: &Tensor,
    b: &Tensor,
    g: &Tensor,
    op: &str,
    fa: impl Fn(f64, f64, f64) -> f64,
    fb: impl Fn(f64, f64, f64) -> f64,
) -> Result<Value, EvalError> {
    let out_shape = broadcast_shape(&a.shape, &b.shape, op)?;
    if g.shape != out_shape {
        return Err(EvalError::Other(format!(
            "{op}: grad_out shape {:?} != broadcast output {out_shape:?} of {:?} and {:?}",
            g.shape, a.shape, b.shape
        )));
    }
    let a_bc = broadcast_to(a, &out_shape);
    let b_bc = broadcast_to(b, &out_shape);
    let ga = Tensor {
        shape: out_shape.clone(),
        data: (0..g.data.len())
            .map(|i| fa(a_bc.data[i], b_bc.data[i], g.data[i]))
            .collect(),
    };
    let gb = Tensor {
        shape: out_shape,
        data: (0..g.data.len())
            .map(|i| fb(a_bc.data[i], b_bc.data[i], g.data[i]))
            .collect(),
    };
    Ok(json!({
        "a": encode_tensor(unbroadcast(&ga, &a.shape)?),
        "b": encode_tensor(unbroadcast(&gb, &b.shape)?),
    }))
}

/// The Rust handl for `op "tensor.add_backward" : (Tensor * Tensor * Tensor) -> Value`.
///
/// The adjoint [`DERIV_PAIRS`](crate::ops::DERIV_PAIRS) registers for
/// `tensor.add` — `add_backward(a, b, grad_out)`: `da = g`, `db = g`,
/// each summed back over the dims `a`/`b` broadcast across.
#[derive(Debug, Clone, Copy)]
pub struct AddBackward;
impl Function for AddBackward {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 3)?;
        let a = require_tensor(args, 0)?;
        let b = require_tensor(args, 1)?;
        let g = require_tensor(args, 2)?;
        elementwise_backward(&a, &b, &g, "tensor.add_backward", |_, _, g| g, |_, _, g| g)
    }
}

/// The Rust handl for `op "tensor.mul_backward" : (Tensor * Tensor * Tensor) -> Value`.
///
/// The adjoint of `tensor.mul` — `mul_backward(a, b, grad_out)`:
/// `da = g·b`, `db = g·a`, each unbroadcast to its operand's shape.
#[derive(Debug, Clone, Copy)]
pub struct MulBackward;
impl Function for MulBackward {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 3)?;
        let a = require_tensor(args, 0)?;
        let b = require_tensor(args, 1)?;
        let g = require_tensor(args, 2)?;
        elementwise_backward(&a, &b, &g, "tensor.mul_backward", |_, y, g| g * y, |x, _, g| g * x)
    }
}

/// The Rust handl for `op "tensor.sub_backward" : (Tensor * Tensor * Tensor) -> Value`.
///
/// The adjoint of `tensor.sub` — `sub_backward(a, b, grad_out)`:
/// `da = g`, `db = −g`, unbroadcast like [`AddBackward`].
#[derive(Debug, Clone, Copy)]
pub struct SubBackward;
impl Function for SubBackward {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 3)?;
        let a = require_tensor(args, 0)?;
        let b = require_tensor(args, 1)?;
        let g = require_tensor(args, 2)?;
        elementwise_backward(&a, &b, &g, "tensor.sub_backward", |_, _, g| g, |_, _, g| -g)
    }
}

/// The Rust handl for `op "tensor.div_backward" : (Tensor * Tensor * Tensor) -> Value`.
///
/// The adjoint of `tensor.div` — `div_backward(a, b, grad_out)`:
/// `da = g/b`, `db = −g·a/b²`, unbroadcast like [`AddBackward`]. The
/// same `f64` caveat as `Div` applies: a `b` element of `0` produces
/// `inf`/`NaN` gradients, which the `serde_json` wire renders `null`.
#[derive(Debug, Clone, Copy)]
pub struct DivBackward;
impl Function for DivBackward {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 3)?;
        let a = require_tensor(args, 0)?;
        let b = require_tensor(args, 1)?;
        let g = require_tensor(args, 2)?;
        elementwise_backward(
            &a,
            &b,
            &g,
            "tensor.div_backward",
            |_, y, g| g / y,
            |x, y, g| -g * x / (y * y),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::testutil::{call_fn, tensor};

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

    // ── backward ──────────────────────────────────────────────────────

    #[test]
    fn add_backward_same_shape_passes_grad_through() {
        let v = call_fn(
            &AddBackward,
            &[
                tensor(&[2], &[1.0, 2.0]),
                tensor(&[2], &[10.0, 20.0]),
                tensor(&[2], &[0.5, 1.5]),
            ],
        )
        .unwrap();
        assert_eq!(v["a"], tensor(&[2], &[0.5, 1.5]));
        assert_eq!(v["b"], tensor(&[2], &[0.5, 1.5]));
    }

    #[test]
    fn add_backward_unbroadcasts_the_stretched_side() {
        // forward [2,3] + [3] → [2,3]; db sums the grad over axis 0.
        let v = call_fn(
            &AddBackward,
            &[
                tensor(&[2, 3], &[0.0; 6]),
                tensor(&[3], &[10.0, 20.0, 30.0]),
                tensor(&[2, 3], &[1.0; 6]),
            ],
        )
        .unwrap();
        assert_eq!(v["a"], tensor(&[2, 3], &[1.0; 6]));
        assert_eq!(v["b"], tensor(&[3], &[2.0, 2.0, 2.0]));
        // Outer case: [2,1] + [1,3] → [2,3]; da sums across columns.
        let v = call_fn(
            &AddBackward,
            &[
                tensor(&[2, 1], &[1.0, 10.0]),
                tensor(&[1, 3], &[100.0, 200.0, 300.0]),
                tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ],
        )
        .unwrap();
        assert_eq!(v["a"], tensor(&[2, 1], &[6.0, 15.0]));
        assert_eq!(v["b"], tensor(&[1, 3], &[5.0, 7.0, 9.0]));
    }

    #[test]
    fn mul_backward_scales_by_the_other_operand() {
        // a=[2,3] b=[4,5] g=[1,1]: da = g·b = [4,5]; db = g·a = [2,3].
        let v = call_fn(
            &MulBackward,
            &[
                tensor(&[2], &[2.0, 3.0]),
                tensor(&[2], &[4.0, 5.0]),
                tensor(&[2], &[1.0, 1.0]),
            ],
        )
        .unwrap();
        assert_eq!(v["a"], tensor(&[2], &[4.0, 5.0]));
        assert_eq!(v["b"], tensor(&[2], &[2.0, 3.0]));
        // Broadcast: a [2] × b [1] with g [1,1] → db = Σ g·a = 5.
        let v = call_fn(
            &MulBackward,
            &[
                tensor(&[2], &[2.0, 3.0]),
                tensor(&[1], &[10.0]),
                tensor(&[2], &[1.0, 1.0]),
            ],
        )
        .unwrap();
        assert_eq!(v["a"], tensor(&[2], &[10.0, 10.0]));
        assert_eq!(v["b"], tensor(&[1], &[5.0]));
    }

    #[test]
    fn sub_backward_negates_b() {
        let v = call_fn(
            &SubBackward,
            &[
                tensor(&[2], &[0.0, 0.0]),
                tensor(&[2], &[0.0, 0.0]),
                tensor(&[2], &[2.0, 4.0]),
            ],
        )
        .unwrap();
        assert_eq!(v["a"], tensor(&[2], &[2.0, 4.0]));
        assert_eq!(v["b"], tensor(&[2], &[-2.0, -4.0]));
    }

    #[test]
    fn div_backward_quotients() {
        // a=[8,9] b=[2,3] g=[1,1]: da = g/b = [1/2, 1/3];
        // db = −g·a/b² = [−8/4, −9/9] = [−2, −1].
        let v = call_fn(
            &DivBackward,
            &[
                tensor(&[2], &[8.0, 9.0]),
                tensor(&[2], &[2.0, 3.0]),
                tensor(&[2], &[1.0, 1.0]),
            ],
        )
        .unwrap();
        assert_eq!(v["a"], tensor(&[2], &[0.5, 1.0 / 3.0]));
        assert_eq!(v["b"], tensor(&[2], &[-2.0, -1.0]));
    }

    #[test]
    fn backward_rejects_bad_grad_shape_and_nonbroadcast_operands() {
        // grad_out must be the broadcast output shape, not an operand's.
        assert!(
            call_fn(
                &AddBackward,
                &[
                    tensor(&[2, 3], &[0.0; 6]),
                    tensor(&[3], &[0.0; 3]),
                    tensor(&[3], &[0.0; 3]),
                ]
            )
            .is_err()
        );
        // Non-broadcast operands fail at the same broadcast_shape check.
        assert!(
            call_fn(
                &SubBackward,
                &[
                    tensor(&[2], &[0.0; 2]),
                    tensor(&[3], &[0.0; 3]),
                    tensor(&[2], &[0.0; 2]),
                ]
            )
            .is_err()
        );
        assert!(call_fn(&MulBackward, &[tensor(&[2], &[0.0; 2])]).is_err());
    }
}
