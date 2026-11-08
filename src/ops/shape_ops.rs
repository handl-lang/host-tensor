//! `reshape` / `transpose` / `concat`: the re-labeling ops — the impl
//! mirrored by lib-tensor's `shape_ops.hnl`.

use handl_platform::{EvalContext, EvalError, Function};
use serde_json::{Value, json};

use super::codec::{
    Tensor, decode_tensor, encode_tensor, require_extent, require_shape, require_tensor,
    row_major_strides, type_mismatch,
};

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

/// The Rust handl for `op "tensor.reshape" : (Tensor * (List Int)) -> Tensor`.
///
/// Re-labels the flat row-major payload under a new shape; the extents'
/// product must equal the element count (the data order never changes —
/// a reshape is a view, not a permutation). No `-1` wildcard: the full
/// shape is spelled out, matching `zeros`' shape convention.
#[derive(Debug, Clone, Copy)]
pub struct Reshape;
impl Function for Reshape {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 2)?;
        let t = require_tensor(args, 0)?;
        let shape = require_shape(args, 1)?;
        let expected = shape.iter().product::<usize>();
        if expected != t.data.len() {
            return Err(EvalError::Other(format!(
                "tensor.reshape: shape {shape:?} has {expected} elements, \
                 tensor {:?} has {}",
                t.shape,
                t.data.len()
            )));
        }
        Ok(encode_tensor(Tensor {
            shape,
            data: t.data,
        }))
    }
}

/// The Rust handl for `op "tensor.transpose" : (Tensor * Int * Int) -> Tensor`.
///
/// Swaps two axes: `transpose(t, d0, d1)` returns the tensor whose
/// `(…, i_d1, …, i_d0, …)` element is `t`'s `(…, i_d0, …, i_d1, …)`.
/// `d0 == d1` is the identity. A rank-2 `transpose(t, 0, 1)` is the
/// matrix transpose.
#[derive(Debug, Clone, Copy)]
pub struct Transpose;
impl Function for Transpose {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 3)?;
        let t = require_tensor(args, 0)?;
        let d0 = require_extent(&args[1])?;
        let d1 = require_extent(&args[2])?;
        Ok(encode_tensor(transpose_dims(&t, d0, d1)?))
    }
}

/// `t` with axes `d0`/`d1` swapped — the shared core of [`Transpose`] and
/// `matmul_backward`'s per-batch inner transpose.
pub(crate) fn transpose_dims(t: &Tensor, d0: usize, d1: usize) -> Result<Tensor, EvalError> {
    let rank = t.shape.len();
    if d0 >= rank || d1 >= rank {
        return Err(EvalError::Other(format!(
            "tensor.transpose: dims ({d0}, {d1}) out of range for rank {rank} ({:?})",
            t.shape
        )));
    }
    // `perm[d]` = the input axis output axis `d` reads from.
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(d0, d1);
    let out_shape: Vec<usize> = perm.iter().map(|&d| t.shape[d]).collect();
    let in_strides = row_major_strides(&t.shape);
    // Output odometer → input offset via the permuted strides.
    let mut data = Vec::with_capacity(t.data.len());
    let mut idx = vec![0usize; rank];
    for _ in 0..t.data.len() {
        let off: usize = idx
            .iter()
            .zip(&perm)
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

/// Decode a possibly-negative integer dim — `unsqueeze`'s `dim` attr
/// counts from the end when negative (torch semantics, matching
/// mlopt's `d % (rank + 1)` typing rule). Integral floats count, the
/// `require_extent` convention.
fn require_dim(arg: &Value) -> Result<i64, EvalError> {
    let n = arg
        .as_i64()
        .or_else(|| arg.as_u64().and_then(|n| i64::try_from(n).ok()))
        .or_else(|| {
            arg.as_f64()
                .filter(|n| n.fract() == 0.0 && n.is_finite())
                .map(|n| n as i64)
        });
    n.ok_or_else(|| type_mismatch("integer dim", "other"))
}

/// The Rust handl for `op "tensor.unsqueeze" : (Tensor * Int) -> Tensor`.
///
/// `unsqueeze(t, dim)` inserts a size-1 axis at `dim` — torch's
/// `unsqueeze`: the payload is unchanged (a view, like `reshape`),
/// `dim` may be negative (`-(rank+1) <= dim <= rank`, wrapped by
/// `rem_euclid(rank + 1)` exactly as mlopt's shape typing spells it).
/// No adjoint is registered: the backward is `squeeze`, which is not
/// on the surface — `tensor.derive("unsqueeze")` honestly returns
/// `null`.
#[derive(Debug, Clone, Copy)]
pub struct Unsqueeze;
impl Function for Unsqueeze {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 2)?;
        let t = require_tensor(args, 0)?;
        let dim = require_dim(&args[1])?;
        let rank = t.shape.len() as i64;
        let d = dim.rem_euclid(rank + 1);
        let mut shape = t.shape.clone();
        shape.insert(d as usize, 1);
        Ok(encode_tensor(Tensor {
            shape,
            data: t.data,
        }))
    }
}

/// The Rust handl for `op "tensor.select" : (Tensor * Int * Int) -> Tensor`.
///
/// `select(t, dim, index)` removes axis `dim` and keeps only the
/// `index`-th slice along it — torch's `select`: rank drops by one
/// (a rank-1 select yields a rank-0 scalar tensor), the payload is the
/// gathered slice in row-major order. `dim` wraps like `unsqueeze`'s
/// (`rem_euclid(rank)`); `index` follows torch — negative counts from
/// the end, and an index outside `[-size, size)` refuses rather than
/// silently wrapping. No adjoint is registered: the backward is
/// `select_scatter`/`index_add`-family scatter, undeclared —
/// `tensor.derive("select")` honestly returns `null`.
#[derive(Debug, Clone, Copy)]
pub struct Select;
impl Function for Select {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 3)?;
        let t = require_tensor(args, 0)?;
        let dim = require_dim(&args[1])?;
        let index = require_dim(&args[2])?;
        let rank = t.shape.len() as i64;
        if rank == 0 {
            return Err(EvalError::Other(
                "tensor.select: cannot select an axis of a rank-0 tensor".to_string(),
            ));
        }
        let d = dim.rem_euclid(rank) as usize;
        let size = t.shape[d] as i64;
        if !(-size..size).contains(&index) {
            return Err(EvalError::Other(format!(
                "tensor.select: index {index} out of range for axis {d} of size {size} ({:?})",
                t.shape
            )));
        }
        let i = index.rem_euclid(size) as usize;
        // Kept axes in input order; output axis j reads input axis kept[j].
        let kept: Vec<usize> = (0..t.shape.len()).filter(|&k| k != d).collect();
        let out_shape: Vec<usize> = kept.iter().map(|&k| t.shape[k]).collect();
        let in_strides = row_major_strides(&t.shape);
        let fixed = i * in_strides[d];
        let count: usize = out_shape.iter().product();
        let mut data = Vec::with_capacity(count);
        let mut idx = vec![0usize; kept.len()];
        for _ in 0..count {
            let off: usize = idx
                .iter()
                .zip(&kept)
                .map(|(&c, &k)| c * in_strides[k])
                .sum::<usize>()
                + fixed;
            data.push(t.data[off]);
            for j in (0..kept.len()).rev() {
                idx[j] += 1;
                if idx[j] < out_shape[j] {
                    break;
                }
                idx[j] = 0;
            }
        }
        Ok(encode_tensor(Tensor {
            shape: out_shape,
            data,
        }))
    }
}

/// The Rust handl for `op "tensor.concat" : (Array * Int) -> Tensor`.
///
/// `concat(xs, axis)` joins a non-empty list of tensors along `axis`.
/// All inputs must share a rank and agree on every axis but `axis`,
/// whose extents sum. `concat(xs, 0)` stacks, `concat(xs, rank−1)`
/// appends along the fastest-varying axis.
#[derive(Debug, Clone, Copy)]
pub struct Concat;
impl Function for Concat {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 2)?;
        let arr = args[0]
            .as_array()
            .ok_or_else(|| type_mismatch("tensor.concat: array of tensors", "non-array"))?;
        if arr.is_empty() {
            return Err(EvalError::Other(
                "tensor.concat: expected a non-empty list of tensors".to_string(),
            ));
        }
        let axis = require_extent(&args[1])?;
        let ts: Vec<Tensor> = arr.iter().map(decode_tensor).collect::<Result<_, _>>()?;
        let rank = ts[0].shape.len();
        if rank == 0 {
            return Err(EvalError::Other(
                "tensor.concat: cannot concat rank-0 tensors".to_string(),
            ));
        }
        if axis >= rank {
            return Err(EvalError::Other(format!(
                "tensor.concat: axis {axis} out of range for rank {rank} ({:?})",
                ts[0].shape
            )));
        }
        for (i, t) in ts.iter().enumerate().skip(1) {
            if t.shape.len() != rank {
                return Err(EvalError::Other(format!(
                    "tensor.concat: element {i} has rank {} != {rank} ({:?} vs {:?})",
                    t.shape.len(),
                    t.shape,
                    ts[0].shape
                )));
            }
            for (d, (&a, &b)) in t.shape.iter().zip(ts[0].shape.iter()).enumerate() {
                if d != axis && a != b {
                    return Err(EvalError::Other(format!(
                        "tensor.concat: element {i} differs on non-concat axis {d} \
                         ({:?} vs {:?})",
                        t.shape, ts[0].shape
                    )));
                }
            }
        }
        // Row-major concat along `axis`: dims before `axis` are equal
        // across inputs (the `outer` repeat count), dims after are equal
        // (the `inner` contiguous run per axis element). Each input's
        // per-outer block is contiguous — copy block-by-block.
        let outer: usize = ts[0].shape[..axis].iter().product();
        let inner: usize = ts[0].shape[axis + 1..].iter().product();
        let mut out_shape = ts[0].shape.clone();
        out_shape[axis] = ts.iter().map(|t| t.shape[axis]).sum();
        let mut data = Vec::with_capacity(out_shape.iter().product());
        for o in 0..outer {
            for t in &ts {
                let slab = t.shape[axis] * inner;
                data.extend_from_slice(&t.data[o * slab..(o + 1) * slab]);
            }
        }
        Ok(encode_tensor(Tensor {
            shape: out_shape,
            data,
        }))
    }
}
/// The Rust handl for `op "tensor.reshape_backward" : Value... -> Value`.
///
/// The adjoint [`DERIV_PAIRS`](crate::ops::DERIV_PAIRS) registers for
/// `tensor.reshape` — `reshape_backward(input, shape, grad_out)`: `g`
/// relabeled back to the input's shape (a reshape moves no data).
/// Returns `{"input": dx}`.
#[derive(Debug, Clone, Copy)]
pub struct ReshapeBackward;
impl Function for ReshapeBackward {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 3)?;
        let t = require_tensor(args, 0)?;
        let shape = require_shape(args, 1)?;
        let g = require_tensor(args, 2)?;
        let expected = shape.iter().product::<usize>();
        if expected != t.data.len() {
            return Err(EvalError::Other(format!(
                "tensor.reshape_backward: shape {shape:?} has {expected} elements, \
                 input {:?} has {}",
                t.shape,
                t.data.len()
            )));
        }
        if g.shape != shape {
            return Err(EvalError::Other(format!(
                "tensor.reshape_backward: grad_out shape {:?} != reshape output {shape:?}",
                g.shape
            )));
        }
        Ok(json!({
            "input": encode_tensor(Tensor { shape: t.shape, data: g.data }),
        }))
    }
}

/// The Rust handl for `op "tensor.transpose_backward" : (Tensor * Int * Int * Tensor) -> Value`.
///
/// The adjoint of `tensor.transpose` — `transpose_backward(input, d0,
/// d1, grad_out)`: `g` transposed along the same axes (transpose is
/// self-inverse). Returns `{"input": dx}`.
#[derive(Debug, Clone, Copy)]
pub struct TransposeBackward;
impl Function for TransposeBackward {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 4)?;
        let t = require_tensor(args, 0)?;
        let d0 = require_extent(&args[1])?;
        let d1 = require_extent(&args[2])?;
        let g = require_tensor(args, 3)?;
        let dx = transpose_dims(&g, d0, d1)?;
        if dx.shape != t.shape {
            return Err(EvalError::Other(format!(
                "tensor.transpose_backward: grad_out {:?} transposed to {:?} \
                 != input {:?}",
                g.shape, dx.shape, t.shape
            )));
        }
        Ok(json!({ "input": encode_tensor(dx) }))
    }
}

/// The Rust handl for `op "tensor.concat_backward" : (Array * Int * Tensor) -> Value`.
///
/// The adjoint of `tensor.concat` — `concat_backward(xs, axis,
/// grad_out)`: `g` split back along `axis` at each input's extent,
/// mirroring the primal's rank/axis-agreement checks. Returns
/// `{"xs": [dx_1, ..., dx_n]}`.
#[derive(Debug, Clone, Copy)]
pub struct ConcatBackward;
impl Function for ConcatBackward {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 3)?;
        let arr = args[0]
            .as_array()
            .ok_or_else(|| type_mismatch("tensor list", "non-array"))?;
        if arr.is_empty() {
            return Err(EvalError::Other(
                "tensor.concat_backward: empty tensor list".to_string(),
            ));
        }
        let ts: Vec<Tensor> = arr
            .iter()
            .enumerate()
            .map(|(_i, v)| decode_tensor(v))
            .collect::<Result<_, _>>()?;
        let axis = require_extent(&args[1])?;
        let g = require_tensor(args, 2)?;
        let rank = ts[0].shape.len();
        if axis >= rank {
            return Err(EvalError::Other(format!(
                "tensor.concat_backward: axis {axis} out of range for rank {rank}"
            )));
        }
        for (i, t) in ts.iter().enumerate().skip(1) {
            if t.shape.len() != rank {
                return Err(EvalError::Other(format!(
                    "tensor.concat_backward: rank mismatch: element 0 rank {rank}, \
                     element {i} rank {}",
                    t.shape.len()
                )));
            }
            for (d, (&e0, &ei)) in ts[0].shape.iter().zip(&t.shape).enumerate() {
                if d != axis && e0 != ei {
                    return Err(EvalError::Other(format!(
                        "tensor.concat_backward: non-axis dim {d} mismatch: \
                         element 0 extent {e0}, element {i} extent {ei}"
                    )));
                }
            }
        }
        let total: usize = ts.iter().map(|t| t.shape[axis]).sum();
        let mut out_shape = ts[0].shape.clone();
        out_shape[axis] = total;
        if g.shape != out_shape {
            return Err(EvalError::Other(format!(
                "tensor.concat_backward: grad_out shape {:?} != concat output {out_shape:?}",
                g.shape
            )));
        }
        let g_strides = row_major_strides(&g.shape);
        let mut dxs = Vec::with_capacity(ts.len());
        let mut off = 0usize;
        for t in &ts {
            let t_rank = t.shape.len();
            let mut data = Vec::with_capacity(t.data.len());
            let mut idx = vec![0usize; t_rank];
            for _ in 0..t.data.len() {
                let mut goff = off * g_strides[axis];
                for (d, &i) in idx.iter().enumerate() {
                    goff += i * g_strides[d];
                }
                data.push(g.data[goff]);
                for d in (0..t_rank).rev() {
                    idx[d] += 1;
                    if idx[d] < t.shape[d] {
                        break;
                    }
                    idx[d] = 0;
                }
            }
            off += t.shape[axis];
            dxs.push(encode_tensor(Tensor {
                shape: t.shape.clone(),
                data,
            }));
        }
        Ok(json!({ "xs": dxs }))
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
    fn reshape_relables_shape_keeps_data() {
        let v = call_fn(
            &Reshape,
            &[
                tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                json!([3, 2]),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        // To rank-1 and to scalar.
        assert_eq!(
            call_fn(&Reshape, &[tensor(&[2, 2], &[1.0; 4]), json!([4])]).unwrap(),
            tensor(&[4], &[1.0; 4])
        );
        assert_eq!(
            call_fn(&Reshape, &[tensor(&[1], &[7.0]), json!([])]).unwrap(),
            tensor(&[], &[7.0])
        );
    }

    #[test]
    fn reshape_rejects_wrong_element_count() {
        assert!(call_fn(&Reshape, &[tensor(&[2, 3], &[0.0; 6]), json!([2, 2])]).is_err());
        assert!(call_fn(&Reshape, &[tensor(&[2, 3], &[0.0; 6]), json!([7])]).is_err());
    }

    #[test]
    fn unsqueeze_inserts_axis_and_wraps_negative_dim() {
        let t = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // dim 0 → leading axis; dim 2 → trailing; dim -1 → same.
        assert_eq!(
            call_fn(&Unsqueeze, &[t.clone(), json!(0)]).unwrap(),
            tensor(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
        assert_eq!(
            call_fn(&Unsqueeze, &[t.clone(), json!(2)]).unwrap(),
            tensor(&[2, 3, 1], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
        assert_eq!(
            call_fn(&Unsqueeze, &[t.clone(), json!(-1)]).unwrap(),
            tensor(&[2, 3, 1], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
        // rank-0 → rank-1; out-of-range dims wrap like mlopt's typing.
        assert_eq!(
            call_fn(&Unsqueeze, &[tensor(&[], &[7.0]), json!(0)]).unwrap(),
            tensor(&[1], &[7.0])
        );
        assert_eq!(
            call_fn(&Unsqueeze, &[t.clone(), json!(9)]).unwrap(),
            call_fn(&Unsqueeze, &[t.clone(), json!(0)]).unwrap()
        );
        // non-integer dims refuse.
        assert!(call_fn(&Unsqueeze, &[t.clone(), json!(0.5)]).is_err());
        assert!(call_fn(&Unsqueeze, &[t.clone(), json!("a")]).is_err());
    }

    #[test]
    fn select_drops_axis_and_gathers_slice() {
        // [[1,2,3],[4,5,6]] select dim 0 index 1 → [4,5,6].
        let t = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            call_fn(&Select, &[t.clone(), json!(0), json!(1)]).unwrap(),
            tensor(&[3], &[4.0, 5.0, 6.0])
        );
        // dim 1 index 0 → column 0 as a rank-1 [2].
        assert_eq!(
            call_fn(&Select, &[t.clone(), json!(1), json!(0)]).unwrap(),
            tensor(&[2], &[1.0, 4.0])
        );
        // negative dim and index wrap like torch.
        assert_eq!(
            call_fn(&Select, &[t.clone(), json!(-1), json!(-1)]).unwrap(),
            tensor(&[2], &[3.0, 6.0])
        );
        // rank-1 → rank-0 scalar.
        assert_eq!(
            call_fn(
                &Select,
                &[tensor(&[3], &[7.0, 8.0, 9.0]), json!(0), json!(2)]
            )
            .unwrap(),
            tensor(&[], &[9.0])
        );
        // out-of-range index refuses; rank-0 has no axis to select.
        assert!(call_fn(&Select, &[t.clone(), json!(0), json!(2)]).is_err());
        assert!(call_fn(&Select, &[t.clone(), json!(0), json!(-3)]).is_err());
        assert!(call_fn(&Select, &[tensor(&[], &[7.0]), json!(0), json!(0)]).is_err());
        // non-integer args refuse.
        assert!(call_fn(&Select, &[t.clone(), json!(0.5), json!(0)]).is_err());
        assert!(call_fn(&Select, &[t.clone(), json!(0), json!("a")]).is_err());
    }

    #[test]
    fn transpose_matrix() {
        // [[1,2,3],[4,5,6]]ᵀ = [[1,4],[2,5],[3,6]].
        let v = call_fn(
            &Transpose,
            &[
                tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                json!(0),
                json!(1),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[3, 2], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
    }

    #[test]
    fn transpose_rank3_swaps_inner_axes() {
        // [1,2,3] = [[[1,2,3],[4,5,6]]] → transpose(1,2) = [1,3,2].
        let v = call_fn(
            &Transpose,
            &[
                tensor(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                json!(1),
                json!(2),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[1, 3, 2], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
        // transpose(0,2): [1,2,3] → [3,2,1]; out[i,j,0] = in[0,j,i], so
        // out = [[[1],[4]], [[2],[5]], [[3],[6]]] flattened.
        let v = call_fn(
            &Transpose,
            &[
                tensor(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                json!(0),
                json!(2),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[3, 2, 1], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
    }

    #[test]
    fn transpose_identity_and_rejects_bad_dims() {
        let t = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            call_fn(&Transpose, &[t.clone(), json!(1), json!(1)]).unwrap(),
            t
        );
        assert!(call_fn(&Transpose, &[t.clone(), json!(0), json!(2)]).is_err());
        assert!(call_fn(&Transpose, &[t.clone(), json!(5), json!(0)]).is_err());
        assert!(call_fn(&Transpose, &[tensor(&[], &[1.0]), json!(0), json!(1)]).is_err());
    }

    #[test]
    fn concat_axis0_stacks() {
        // [[1,2],[3,4]] ++ [[5,6]] along axis 0 → [[1,2],[3,4],[5,6]].
        let v = call_fn(
            &Concat,
            &[
                json!([
                    tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                    tensor(&[1, 2], &[5.0, 6.0])
                ]),
                json!(0),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    }

    #[test]
    fn concat_axis1_appends_columns() {
        // [[1,2],[3,4]] ++ [[5],[6]] along axis 1 → [[1,2,5],[3,4,6]].
        let v = call_fn(
            &Concat,
            &[
                json!([
                    tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                    tensor(&[2, 1], &[5.0, 6.0])
                ]),
                json!(1),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[2, 3], &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]));
    }

    #[test]
    fn concat_rejects_bad_inputs() {
        let t = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        // empty list / non-array / non-tensor element.
        assert!(call_fn(&Concat, &[json!([]), json!(0)]).is_err());
        assert!(call_fn(&Concat, &[json!(5), json!(0)]).is_err());
        assert!(call_fn(&Concat, &[json!([t.clone(), json!(7)]), json!(0)]).is_err());
        // axis out of range / mismatched non-concat axis / rank mismatch.
        assert!(call_fn(&Concat, &[json!([t.clone()]), json!(2)]).is_err());
        assert!(
            call_fn(
                &Concat,
                &[json!([t.clone(), tensor(&[2, 3], &[0.0; 6])]), json!(0)]
            )
            .is_err()
        );
        assert!(
            call_fn(
                &Concat,
                &[json!([t.clone(), tensor(&[4], &[0.0; 4])]), json!(0)]
            )
            .is_err()
        );
        // rank-0 tensors.
        assert!(
            call_fn(
                &Concat,
                &[json!([tensor(&[], &[1.0]), tensor(&[], &[2.0])]), json!(0)]
            )
            .is_err()
        );
    }

    // ── backward ──────────────────────────────────────────────────────

    #[test]
    fn reshape_backward_relabels_grad() {
        let v = call_fn(
            &ReshapeBackward,
            &[
                tensor(&[2, 3], &[0.0; 6]),
                json!([3, 2]),
                tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ],
        )
        .unwrap();
        assert_eq!(v["input"], tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        // grad must be the reshape output's shape.
        assert!(
            call_fn(
                &ReshapeBackward,
                &[
                    tensor(&[2, 3], &[0.0; 6]),
                    json!([3, 2]),
                    tensor(&[2, 3], &[0.0; 6]),
                ]
            )
            .is_err()
        );
    }

    #[test]
    fn transpose_backward_is_self_inverse() {
        let v = call_fn(
            &TransposeBackward,
            &[
                tensor(&[2, 3], &[0.0; 6]),
                json!(0),
                json!(1),
                tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ],
        )
        .unwrap();
        // g [3,2] transposed back → [[1,3,5],[2,4,6]] shape [2,3].
        assert_eq!(v["input"], tensor(&[2, 3], &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]));
        // grad that transposes to the wrong shape errors.
        assert!(
            call_fn(
                &TransposeBackward,
                &[
                    tensor(&[2, 3], &[0.0; 6]),
                    json!(0),
                    json!(1),
                    tensor(&[2, 3], &[0.0; 6]),
                ]
            )
            .is_err()
        );
    }

    #[test]
    fn concat_backward_splits_at_extents() {
        // concat([[2,2],[2,2]], axis 1) → [2,4]; g = 1..8 splits back.
        let v = call_fn(
            &ConcatBackward,
            &[
                json!([tensor(&[2, 2], &[0.0; 4]), tensor(&[2, 2], &[0.0; 4])]),
                json!(1),
                tensor(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            ],
        )
        .unwrap();
        let xs = v["xs"].as_array().unwrap();
        assert_eq!(xs[0], tensor(&[2, 2], &[1.0, 2.0, 5.0, 6.0]));
        assert_eq!(xs[1], tensor(&[2, 2], &[3.0, 4.0, 7.0, 8.0]));
        // axis-0 stack: g [4,1] splits to [2,1]+[2,1].
        let v = call_fn(
            &ConcatBackward,
            &[
                json!([tensor(&[2, 1], &[0.0; 2]), tensor(&[2, 1], &[0.0; 2])]),
                json!(0),
                tensor(&[4, 1], &[1.0, 2.0, 3.0, 4.0]),
            ],
        )
        .unwrap();
        let xs = v["xs"].as_array().unwrap();
        assert_eq!(xs[0], tensor(&[2, 1], &[1.0, 2.0]));
        assert_eq!(xs[1], tensor(&[2, 1], &[3.0, 4.0]));
        // mismatched grad shape / rank errors.
        assert!(
            call_fn(
                &ConcatBackward,
                &[
                    json!([tensor(&[2, 2], &[0.0; 4]), tensor(&[2, 2], &[0.0; 4])]),
                    json!(1),
                    tensor(&[2, 2], &[0.0; 4]),
                ]
            )
            .is_err()
        );
    }
}

