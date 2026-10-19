//! `add` / `mul` / `sub` / `div` with numpy broadcasting — the impl
//! mirrored by lib-tensor's `elementwise.hnl`.

use handl_platform::{EvalContext, EvalError, Function};
use serde_json::Value;

use super::codec::{Tensor, encode_tensor, require_tensor};

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
}
