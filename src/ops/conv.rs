//! `conv2d` / `max_pool`: the NCHW spatial ops — the impl mirrored by
//! lib-tensor's `conv.hnl`.

use handl_platform::{EvalContext, EvalError, Function};
use serde_json::{Value, json};

use super::codec::{
    Tensor, encode_tensor, has_opt_arg, opt_extent_pair, opt_positive_int, opt_tensor,
    require_arg_range, require_extent, require_tensor, type_mismatch,
};

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
/// The Rust handl for `op "tensor.conv2d_backward" : Value... -> Value`.
///
/// The adjoint [`DERIV_PAIRS`](crate::ops::DERIV_PAIRS) registers for
/// `tensor.conv2d` — `conv2d_backward(input, weight, grad_out, bias?,
/// stride?, padding?, groups?)`: the required primal args, then the
/// upstream gradient (the primal's `[N, C_out, H_out, W_out]` output
/// shape — checked), then the primal's positional optionals in the same
/// order. Returns `{"input": dx, "weight": dw}` plus `"bias": db` (the
/// per-channel sum of `grad_out`) when a `bias` argument is present.
/// `dx` is the transposed convolution and `dw` the input–gradient
/// cross-correlation — naive `f64` loops like `Conv2d`.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dBackward;
impl Function for Conv2dBackward {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        require_arg_range(args, 3, 7, "tensor.conv2d_backward")?;
        let input = require_tensor(args, 0)?;
        let weight = require_tensor(args, 1)?;
        let grad = require_tensor(args, 2)?;
        let bias = opt_tensor(args, 3)?;
        let [s_h, s_w] = opt_extent_pair(args, 4, [1, 1], "tensor.conv2d_backward: stride")?;
        let [p_h, p_w] = opt_extent_pair(args, 5, [0, 0], "tensor.conv2d_backward: padding")?;
        let groups = opt_positive_int(args, 6, 1, "tensor.conv2d_backward: groups")?;
        if s_h == 0 || s_w == 0 {
            return Err(EvalError::Other(format!(
                "tensor.conv2d_backward: stride must be positive, got [{s_h}, {s_w}]"
            )));
        }
        if input.shape.len() != 4 || weight.shape.len() != 4 || grad.shape.len() != 4 {
            return Err(EvalError::Other(format!(
                "tensor.conv2d_backward: expected rank-4 input, weight, and grad_out, \
                 got ranks {}, {}, and {}",
                input.shape.len(),
                weight.shape.len(),
                grad.shape.len()
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
                "tensor.conv2d_backward: groups {groups} must divide both C_in {c_in} and C_out {c_out}"
            )));
        }
        let c_in_g = c_in / groups;
        let c_out_g = c_out / groups;
        if c_in_w != c_in_g {
            return Err(EvalError::Other(format!(
                "tensor.conv2d_backward: weight channels {c_in_w} != C_in/groups {c_in_g} \
                 (input [N, {c_in}, {h}, {w}], weight [{c_out}, {c_in_w}, {k_h}, {k_w}], groups {groups})"
            )));
        }
        if let Some(b) = &bias {
            if b.shape != [c_out] {
                return Err(EvalError::Other(format!(
                    "tensor.conv2d_backward: bias shape {:?} != [{c_out}]",
                    b.shape
                )));
            }
        }
        let Some(h_out_num) = (h + 2 * p_h).checked_sub(k_h) else {
            return Err(EvalError::Other(format!(
                "tensor.conv2d_backward: kernel height {k_h} exceeds padded input height {}",
                h + 2 * p_h
            )));
        };
        let Some(w_out_num) = (w + 2 * p_w).checked_sub(k_w) else {
            return Err(EvalError::Other(format!(
                "tensor.conv2d_backward: kernel width {k_w} exceeds padded input width {}",
                w + 2 * p_w
            )));
        };
        let h_out = h_out_num / s_h + 1;
        let w_out = w_out_num / s_w + 1;
        if grad.shape != [n, c_out, h_out, w_out] {
            return Err(EvalError::Other(format!(
                "tensor.conv2d_backward: grad_out shape {:?} != conv2d output {:?}",
                grad.shape,
                [n, c_out, h_out, w_out]
            )));
        }
        let mut dx = vec![0.0; n * c_in * h * w];
        let mut dw = vec![0.0; c_out * c_in_g * k_h * k_w];
        let mut db = vec![0.0; c_out];
        for bn in 0..n {
            for g in 0..groups {
                for oc in 0..c_out_g {
                    let co = g * c_out_g + oc;
                    for oy in 0..h_out {
                        for ox in 0..w_out {
                            let gv = grad.data[((bn * c_out + co) * h_out + oy) * w_out + ox];
                            db[co] += gv;
                            for ic in 0..c_in_g {
                                let ci = g * c_in_g + ic;
                                for ky in 0..k_h {
                                    // Same window arithmetic as the
                                    // primal; out-of-range is the
                                    // zero padding and contributes
                                    // nothing to either gradient.
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
                                        let xv =
                                            input.data[((bn * c_in + ci) * h + iy) * w + ix];
                                        let wv = weight.data
                                            [((co * c_in_g + ic) * k_h + ky) * k_w + kx];
                                        dx[((bn * c_in + ci) * h + iy) * w + ix] += gv * wv;
                                        dw[((co * c_in_g + ic) * k_h + ky) * k_w + kx] += gv * xv;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut out = json!({
            "input": encode_tensor(Tensor { shape: input.shape, data: dx }),
            "weight": encode_tensor(Tensor { shape: weight.shape, data: dw }),
        });
        if bias.is_some() {
            out["bias"] = encode_tensor(Tensor {
                shape: vec![c_out],
                data: db,
            });
        }
        Ok(out)
    }
}

/// The required `[h, w]` (or scalar → `[k, k]`) spatial pair — `kernel`
/// on `max_pool_backward`, which has no default. Mirrors `pair_or` on
/// the `.hnl` side / `opt_extent_pair` without the `null`-is-default arm.
fn require_extent_pair(v: &Value, what: &str) -> Result<[usize; 2], EvalError> {
    match v {
        Value::Number(_) => {
            let k = require_extent(v)?;
            Ok([k, k])
        }
        Value::Array(a) if a.len() == 2 => {
            Ok([require_extent(&a[0])?, require_extent(&a[1])?])
        }
        _ => Err(type_mismatch(
            &format!("{what}: int or [h, w] pair"),
            "other",
        )),
    }
}

/// The Rust handl for `op "tensor.max_pool_backward" : Value... -> Value`.
///
/// The adjoint [`DERIV_PAIRS`](crate::ops::DERIV_PAIRS) registers for
/// `tensor.max_pool` — `max_pool_backward(input, grad_out, kernel,
/// stride?, padding?)`: the input's gradient, scattered to each output
/// element's argmax position (first maximum in row-major scan order —
/// the primal's tie rule). Returns `{"input": dx}`.
#[derive(Debug, Clone, Copy)]
pub struct MaxPoolBackward;
impl Function for MaxPoolBackward {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        require_arg_range(args, 3, 5, "tensor.max_pool_backward")?;
        let t = require_tensor(args, 0)?;
        let g = require_tensor(args, 1)?;
        let [k_h, k_w] = require_extent_pair(&args[2], "kernel")?;
        let [s_h, s_w] = opt_extent_pair(args, 3, [k_h, k_w], "tensor.max_pool_backward: stride")?;
        let [p_h, p_w] = opt_extent_pair(args, 4, [0, 0], "tensor.max_pool_backward: padding")?;
        if k_h == 0 || k_w == 0 {
            return Err(EvalError::Other(format!(
                "tensor.max_pool_backward: kernel must be positive, got [{k_h}, {k_w}]"
            )));
        }
        if s_h == 0 || s_w == 0 {
            return Err(EvalError::Other(format!(
                "tensor.max_pool_backward: stride must be positive, got [{s_h}, {s_w}]"
            )));
        }
        if t.shape.len() != 4 || g.shape.len() != 4 {
            return Err(EvalError::Other(format!(
                "tensor.max_pool_backward: expected rank-4 input and grad_out, got {} and {}",
                t.shape.len(),
                g.shape.len()
            )));
        }
        let (n, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
        let Some(h_out_num) = (h + 2 * p_h).checked_sub(k_h) else {
            return Err(EvalError::Other(format!(
                "tensor.max_pool_backward: kernel height {k_h} exceeds padded input height {}",
                h + 2 * p_h
            )));
        };
        let Some(w_out_num) = (w + 2 * p_w).checked_sub(k_w) else {
            return Err(EvalError::Other(format!(
                "tensor.max_pool_backward: kernel width {k_w} exceeds padded input width {}",
                w + 2 * p_w
            )));
        };
        let h_out = h_out_num / s_h + 1;
        let w_out = w_out_num / s_w + 1;
        if g.shape != [n, c, h_out, w_out] {
            return Err(EvalError::Other(format!(
                "tensor.max_pool_backward: grad_out shape {:?} != max_pool output {:?}",
                g.shape,
                [n, c, h_out, w_out]
            )));
        }
        let mut dx = vec![0.0; n * c * h * w];
        for bn in 0..n {
            for ch in 0..c {
                for oy in 0..h_out {
                    for ox in 0..w_out {
                        let gv = g.data[((bn * c + ch) * h_out + oy) * w_out + ox];
                        // Re-derive the argmax exactly as the primal
                        // scanned: first maximum in (ky, kx) order.
                        let mut best = f64::NEG_INFINITY;
                        let mut best_off = 0usize;
                        for ky in 0..k_h {
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
                                let off = ((bn * c + ch) * h + iy) * w + ix;
                                if t.data[off] > best {
                                    best = t.data[off];
                                    best_off = off;
                                }
                            }
                        }
                        dx[best_off] += gv;
                    }
                }
            }
        }
        Ok(json!({
            "input": encode_tensor(Tensor { shape: t.shape, data: dx }),
        }))
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

    // ── conv2d_backward ───────────────────────────────────────────────

    #[test]
    fn conv2d_backward_1x1() {
        // input [1,2,1,2] = ch0 [1,2], ch1 [3,4]; weight [2,2,1,1] =
        // co0 [1,2], co1 [-1,0]; grad all-ones [1,2,1,2].
        // dx[ci] = Σ_co g·w[co,ci] → ch0: 1 + (−1) = 0, ch1: 2 + 0 = 2.
        // dw[co,ci] = Σ x[ci]·g = 3, 7 for both co; db = Σ g = 2 per co.
        let input = tensor(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let weight = tensor(&[2, 2, 1, 1], &[1.0, 2.0, -1.0, 0.0]);
        let grad = tensor(&[1, 2, 1, 2], &[1.0; 4]);
        let v = call_fn(
            &Conv2dBackward,
            &[input.clone(), weight.clone(), grad.clone()],
        )
        .unwrap();
        assert_eq!(v["input"], tensor(&[1, 2, 1, 2], &[0.0, 0.0, 2.0, 2.0]));
        assert_eq!(v["weight"], tensor(&[2, 2, 1, 1], &[3.0, 7.0, 3.0, 7.0]));
        assert!(v["bias"].is_null(), "no bias arg → no bias key");

        // With bias present the db key appears.
        let v = call_fn(
            &Conv2dBackward,
            &[input, weight, grad, tensor(&[2], &[0.0, 0.0])],
        )
        .unwrap();
        assert_eq!(v["bias"], tensor(&[2], &[2.0, 2.0]));
    }

    #[test]
    fn conv2d_backward_3x3_padding_scatter() {
        // input [1,1,2,2] = 1..4; all-ones 3x3 kernel, pad 1, stride 1:
        // every output sums all four cells (as the forward test showed),
        // so with grad all-ones each input cell is covered by all four
        // output windows → dx = [4; 4], and dw[ky,kx] = Σ over the
        // windows where tap (ky,kx) lands on a real cell — corners see
        // one cell ([1],[4]), edges two, center all four (10).
        let input = tensor(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let weight = tensor(&[1, 1, 3, 3], &[1.0; 9]);
        let grad = tensor(&[1, 1, 2, 2], &[1.0; 4]);
        let v = call_fn(
            &Conv2dBackward,
            &[input, weight, grad, Value::Null, Value::Null, json!(1)],
        )
        .unwrap();
        assert_eq!(v["input"], tensor(&[1, 1, 2, 2], &[4.0; 4]));
        assert_eq!(
            v["weight"],
            tensor(&[1, 1, 3, 3], &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0])
        );
    }

    #[test]
    fn conv2d_backward_rejects_bad_args() {
        let input = tensor(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let weight = tensor(&[1, 1, 1, 1], &[1.0]);
        let grad = tensor(&[1, 1, 2, 2], &[1.0; 4]);
        // grad_out shape must be the conv output, not the input.
        assert!(
            call_fn(
                &Conv2dBackward,
                &[input.clone(), weight.clone(), tensor(&[1, 1, 2, 3], &[0.0; 6])]
            )
            .is_err()
        );
        // non-rank-4 grad.
        assert!(
            call_fn(
                &Conv2dBackward,
                &[input.clone(), weight.clone(), tensor(&[4], &[0.0; 4])]
            )
            .is_err()
        );
        // stride 0 / groups not dividing / arity.
        assert!(
            call_fn(
                &Conv2dBackward,
                &[input.clone(), weight.clone(), grad.clone(), Value::Null, json!(0)]
            )
            .is_err()
        );
        assert!(
            call_fn(
                &Conv2dBackward,
                &[
                    input.clone(),
                    weight.clone(),
                    grad.clone(),
                    Value::Null,
                    Value::Null,
                    Value::Null,
                    json!(2)
                ]
            )
            .is_err()
        );
        assert!(call_fn(&Conv2dBackward, &[input, weight]).is_err());
    }

    // ── max_pool_backward ─────────────────────────────────────────────

    #[test]
    fn max_pool_backward_scatters_to_argmax() {
        // [1,1,4,4] = 1..16, 2x2 non-overlap, g = 1,2,3,4 → each grad
        // lands on its window's max (bottom-right corners).
        let input = tensor(
            &[1, 1, 4, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );
        let grad = tensor(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let v = call_fn(
            &MaxPoolBackward,
            &[input, grad, json!(2)],
        )
        .unwrap();
        assert_eq!(
            v["input"],
            tensor(
                &[1, 1, 4, 4],
                &[
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0,
                    4.0,
                ]
            )
        );
    }

    #[test]
    fn max_pool_backward_tie_and_errors() {
        // Tie: first max in scan order wins (the primal's rule).
        let v = call_fn(
            &MaxPoolBackward,
            &[
                tensor(&[1, 1, 2, 2], &[5.0, 5.0, 1.0, 1.0]),
                tensor(&[1, 1, 1, 1], &[7.0]),
                json!(2),
            ],
        )
        .unwrap();
        assert_eq!(v["input"], tensor(&[1, 1, 2, 2], &[7.0, 0.0, 0.0, 0.0]));
        // grad shape must be the pool output.
        assert!(
            call_fn(
                &MaxPoolBackward,
                &[
                    tensor(&[1, 1, 2, 2], &[0.0; 4]),
                    tensor(&[1, 1, 2, 2], &[0.0; 4]),
                    json!(2)
                ]
            )
            .is_err()
        );
        // kernel required; stride zero rejected.
        assert!(call_fn(&MaxPoolBackward, &[tensor(&[1,1,2,2],&[0.0;4]), tensor(&[1,1,1,1],&[0.0])]).is_err());
        assert!(
            call_fn(
                &MaxPoolBackward,
                &[
                    tensor(&[1, 1, 2, 2], &[0.0; 4]),
                    tensor(&[1, 1, 1, 1], &[0.0]),
                    json!(2),
                    json!(0)
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
}
