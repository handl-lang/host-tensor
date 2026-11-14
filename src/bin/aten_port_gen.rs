//! `aten-port-gen` — generate host-tensor op ports from ATen schemas.
//!
//! ```text
//! cargo run --bin aten-port-gen [--schemas tools/aten_schemas.json]
//!     [--list tools/aten_port_list.txt] [--include GLOB] [--exclude GLOB]
//!     [--emit-spec [PATH]] [--check]
//! ```
//!
//! Input is the offline schema dump written by `tools/aten_extract.py`
//! (runtime `torch._C._jit_get_all_schemas()` introspection — no torch
//! source needed). The curated list (`tools/aten_port_list.txt`) pins
//! `tensor.<name> = aten::<name>[.<overload>]` pairs; `# seam:` entries
//! declare ops whose kernels are not yet handwritten — they emit a
//! stub `Function` that fails loudly rather than faking behavior.
//!
//! Outputs (all marker-delimited, safe to regenerate):
//!
//! * `src/ops/aten_generated.rs` — one `Function` per op (whole file is
//!   generated; header carries provenance).
//! * `src/index.hnl` — `cap`/`op`/`handle` decls + `export tensor`
//!   record entries inside `--- ATEN-PORT-GEN:*` marker regions.
//! * `src/ops/mod.rs` — `TENSOR_OPS` names + `tensor_op_registry()`
//!   `register` lines inside `// ATEN-PORT-GEN:*` marker regions.
//!
//! `--emit-spec` prints the resolved generation spec (schema, decoded
//! params, kernel call, status) as JSON without touching the tree —
//! the probe-then-hand-tune dry run, in the spirit of ext-sdkgen.
//!
//! Schema features handled: optional/nullable args (decoders select
//! ATen defaults), `int[]`/`int[N]`/`SymInt[]` lists, kwonly args
//! (positional on our surface), multi-return schemas (JSON record
//! returns), tensor lists. Features rejected → seam or dropped:
//! `ScalarType`/`Layout`/`Device`/`MemoryFormat`/`Generator`/`Dimname`/
//! `str`/`complex` types, `Tensor(a!)` mutation, `.out`/`*_out`/inplace
//! (`*_`) variants, `...` varargs, `Tensor?[]` variadic tensors.

// Schema-model fields (kwonly, ty) and Kind variants are kept for
// completeness of the model even where the current pipeline doesn't
// read them.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Schema model (mirrors tools/aten_extract.py output)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Arg {
    name: String,
    ty: String,
    default: Option<String>,
    kwonly: bool,
    mutated: bool,
}

#[derive(Debug, Clone)]
struct Schema {
    name: String,
    overload: String,
    args: Vec<Arg>,
    ret: String,
    schema: String,
}

#[derive(Debug)]
struct Inventory {
    torch: String,
    schemas: Vec<Schema>,
}

fn load_inventory(path: &Path) -> Result<Inventory, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| format!("parse {}: {e}", path.display()))?;
    let torch = v["torch"].as_str().unwrap_or("unknown").to_string();
    let mut schemas = Vec::new();
    for s in v["schemas"].as_array().cloned().unwrap_or_default() {
        let args = s["args"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .map(|a| Arg {
                name: a["name"].as_str().unwrap_or("").to_string(),
                ty: a["type"].as_str().unwrap_or("").to_string(),
                default: a["default"].as_str().map(|d| d.to_string()),
                kwonly: a["kwonly"].as_bool().unwrap_or(false),
                mutated: a["mutated"].as_bool().unwrap_or(false),
            })
            .collect();
        schemas.push(Schema {
            name: s["name"].as_str().unwrap_or("").to_string(),
            overload: s["overload"].as_str().unwrap_or("").to_string(),
            args,
            ret: s["ret"].as_str().unwrap_or("").to_string(),
            schema: s["schema"].as_str().unwrap_or("").to_string(),
        });
    }
    Ok(Inventory { torch, schemas })
}

// ---------------------------------------------------------------------------
// Port list
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Entry {
    hnl: String,      // "relu" → op "tensor.relu"
    query: String,    // "aten::relu" or "aten::sum.dim_IntList"
    seam: bool,
    note: String,
}

fn load_list(path: &Path) -> Result<Vec<Entry>, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (ln, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        // Split off a trailing "# seam: ..." — handled above since `#`
        // starts the comment; seam marker is `seam:` inside the comment.
        let seam = raw.contains("seam:");
        let note = raw
            .split('#')
            .nth(1)
            .unwrap_or("")
            .trim()
            .to_string();
        let (lhs, rhs) = line
            .split_once('=')
            .ok_or_else(|| format!("{}:{}: expected `tensor.x = aten::y`", path.display(), ln + 1))?;
        let hnl = lhs.trim().trim_start_matches("tensor.").to_string();
        let query = rhs.trim().to_string();
        if !query.starts_with("aten::") {
            return Err(format!(
                "{}:{}: query must start with aten:: — got {query:?}",
                path.display(),
                ln + 1
            ));
        }
        out.push(Entry { hnl, query, seam, note });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Schema → decode plan
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Tensor,
    TensorOpt,
    TensorList,
    Int,
    IntOpt,
    IntList,
    IntListOpt,
    Float,
    FloatOpt,
    FloatList,
    Bool,
    BoolOpt,
    /// Dim-spec positions (`int[]? dim`, `int[1] dim=[]`) decoded via
    /// `opt_dimlist`/`req_dimlist` — accept bare int OR list.
    DimList,
    DimListReq,
    Unsupported,
}

fn strip_nullable(ty: &str) -> (&str, bool) {
    if let Some(t) = ty.strip_suffix('?') {
        (t, true)
    } else {
        (ty, false)
    }
}

fn classify(ty: &str) -> (Kind, bool) {
    let (mut base, nullable) = strip_nullable(ty);
    // JIT alias annotations: `Tensor(a)`, `Tensor(a -> *)`, `Tensor(a!)`.
    // The `!` (mutation) cases are filtered earlier by is_out_or_inplace;
    // here every parenthesized alias collapses to its head type.
    if let Some(i) = base.find('(') {
        if base.ends_with(')') {
            base = &base[..i];
        }
    }
    let kind = match base {
        "Tensor" => Kind::Tensor,
        "Tensor[]" => Kind::TensorList,
        "int" | "SymInt" => Kind::Int,
        "float" | "Scalar" => Kind::Float,
        "bool" => Kind::Bool,
        "float[]" | "Scalar[]" => Kind::FloatList,
        _ if is_int_list(base) => Kind::IntList,
        _ => Kind::Unsupported,
    };
    // Fold nullability into Opt kinds.
    let kind = if nullable {
        match kind {
            Kind::Tensor => Kind::TensorOpt,
            Kind::Int => Kind::IntOpt,
            Kind::Float => Kind::FloatOpt,
            Kind::Bool => Kind::BoolOpt,
            Kind::IntList => Kind::IntListOpt,
            Kind::TensorList => Kind::Unsupported, // Tensor[]? not needed
            other => other,
        }
    } else {
        kind
    };
    (kind, nullable)
}

fn is_int_list(base: &str) -> bool {
    // int[], SymInt[], int[1], SymInt[2], int[1]? handled after strip.
    base == "int[]" || base == "SymInt[]" || is_fixed_int_list(base)
}

fn is_fixed_int_list(base: &str) -> bool {
    for pre in ["int[", "SymInt["] {
        if let Some(rest) = base.strip_prefix(pre) {
            if rest.ends_with(']') && rest[..rest.len() - 1].chars().all(|c| c.is_ascii_digit()) {
                return true;
            }
        }
    }
    false
}

/// Parse an ATen default literal into Rust tokens for the opt_* calls.
#[derive(Debug, Clone)]
enum Default_ {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Ints(Vec<i64>),
    Other,
}

fn parse_default(s: Option<&str>) -> Default_ {
    match s {
        None => Default_::None,
        Some("None") | Some("null") => Default_::None,
        Some("True") | Some("true") => Default_::Bool(true),
        Some("False") | Some("false") => Default_::Bool(false),
        Some(t) if t.starts_with('[') && t.ends_with(']') => {
            let inner = &t[1..t.len() - 1];
            let nums: Option<Vec<i64>> = inner
                .split(',')
                .map(|p| p.trim().parse::<i64>().ok())
                .collect();
            match nums {
                Some(v) => Default_::Ints(v),
                None => Default_::Other,
            }
        }
        Some(t) => {
            if let Ok(i) = t.parse::<i64>() {
                return Default_::Int(i);
            }
            // ATen prints floats like `1.`, `1e-5`, `-1`, `0.5`.
            let cleaned = t.trim_end_matches('.');
            if let Ok(f) = cleaned.parse::<f64>() {
                if t.contains('.') || t.contains('e') || t.contains('E') {
                    return Default_::Float(f);
                }
                return Default_::Int(f as i64);
            }
            Default_::Other
        }
    }
}

impl Default_ {
    fn rust_i64(&self) -> Option<String> {
        match self {
            Default_::Int(i) => Some(i.to_string()),
            Default_::Float(f) if f.fract() == 0.0 => Some((*f as i64).to_string()),
            _ => None,
        }
    }
    fn rust_f64(&self) -> Option<String> {
        match self {
            Default_::Int(i) => Some(format!("{i}.0")),
            Default_::Float(f) => {
                // Rust `Display` prints `1.0` as `1` — force a float
                // literal so `opt_f64(.., 1)` can't infer `i64`... it
                // can't anyway, but keep the emitted token honest.
                let s = format!("{f}");
                if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") {
                    Some(s)
                } else {
                    Some(format!("{s}.0"))
                }
            }
            _ => None,
        }
    }
    fn rust_bool(&self) -> Option<String> {
        match self {
            Default_::Bool(b) => Some(b.to_string()),
            _ => None,
        }
    }
    fn rust_ints(&self) -> Option<String> {
        match self {
            Default_::Ints(v) => Some(format!(
                "&[{}]",
                v.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
            )),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel map — the handwritten half this generator targets.
// ---------------------------------------------------------------------------

/// How the generated `call` body is produced.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Emit {
    /// Decode args per schema; expr evaluates to `Result<Tensor>` and is
    /// wrapped in `encode_tensor`.
    Tensor,
    /// Decode args per schema; expr evaluates to `Result<Value>`.
    Value,
    /// No decoding — expr gets `args: &[Value]` and returns `Result<Value>`.
    Raw,
}

struct MapEntry {
    hnl: &'static str,
    emit: Emit,
    /// Expr template; `{v0}`…`{v9}` are decode bindings, `{args}` the raw
    /// slice (Raw mode only). Tensor mode: must be `Result<Tensor>`.
    call: &'static str,
    /// Positions decoded via `opt_dimlist`/`req_dimlist` instead of the
    /// generic int-list decoders (accepts bare int OR list).
    dims: &'static [usize],
    /// Positions decoded via `req_dimlist` (required, no default).
    dims_req: &'static [usize],
    /// Override `(min,max)` positional arity when the pinned schema's
    /// arity doesn't cover merged sibling overloads (e.g. `where`
    /// condition-only vs 3-arg, `arange` 1–4 args).
    arity: Option<(usize, usize)>,
    /// Override the emitted HNL signature (merged ops whose return type
    /// depends on which sibling overload the caller used).
    sig: Option<&'static str>,
}

const MAP: &[MapEntry] = &[
    // --- unary ---------------------------------------------------------
    m("abs", Emit::Tensor, "aten_kernels::k_abs(&v0)"),
    m("neg", Emit::Tensor, "aten_kernels::k_neg(&v0)"),
    m("exp", Emit::Tensor, "aten_kernels::k_exp(&v0)"),
    m("exp2", Emit::Tensor, "aten_kernels::k_exp2(&v0)"),
    m("expm1", Emit::Tensor, "aten_kernels::k_expm1(&v0)"),
    m("log", Emit::Tensor, "aten_kernels::k_log(&v0)"),
    m("log1p", Emit::Tensor, "aten_kernels::k_log1p(&v0)"),
    m("log2", Emit::Tensor, "aten_kernels::k_log2(&v0)"),
    m("log10", Emit::Tensor, "aten_kernels::k_log10(&v0)"),
    m("sqrt", Emit::Tensor, "aten_kernels::k_sqrt(&v0)"),
    m("rsqrt", Emit::Tensor, "aten_kernels::k_rsqrt(&v0)"),
    m("reciprocal", Emit::Tensor, "aten_kernels::k_reciprocal(&v0)"),
    m("square", Emit::Tensor, "aten_kernels::k_square(&v0)"),
    m("sin", Emit::Tensor, "aten_kernels::k_sin(&v0)"),
    m("cos", Emit::Tensor, "aten_kernels::k_cos(&v0)"),
    m("tan", Emit::Tensor, "aten_kernels::k_tan(&v0)"),
    m("asin", Emit::Tensor, "aten_kernels::k_asin(&v0)"),
    m("acos", Emit::Tensor, "aten_kernels::k_acos(&v0)"),
    m("atan", Emit::Tensor, "aten_kernels::k_atan(&v0)"),
    m("sinh", Emit::Tensor, "aten_kernels::k_sinh(&v0)"),
    m("cosh", Emit::Tensor, "aten_kernels::k_cosh(&v0)"),
    m("tanh", Emit::Tensor, "aten_kernels::k_tanh(&v0)"),
    m("asinh", Emit::Tensor, "aten_kernels::k_asinh(&v0)"),
    m("acosh", Emit::Tensor, "aten_kernels::k_acosh(&v0)"),
    m("atanh", Emit::Tensor, "aten_kernels::k_atanh(&v0)"),
    m("floor", Emit::Tensor, "aten_kernels::k_floor(&v0)"),
    m("ceil", Emit::Tensor, "aten_kernels::k_ceil(&v0)"),
    m("round", Emit::Tensor, "aten_kernels::k_round(&v0)"),
    m("trunc", Emit::Tensor, "aten_kernels::k_trunc(&v0)"),
    m("frac", Emit::Tensor, "aten_kernels::k_frac(&v0)"),
    m("sign", Emit::Tensor, "aten_kernels::k_sign(&v0)"),
    m("sigmoid", Emit::Tensor, "aten_kernels::k_sigmoid(&v0)"),
    m("silu", Emit::Tensor, "aten_kernels::k_silu(&v0)"),
    m("mish", Emit::Tensor, "aten_kernels::k_mish(&v0)"),
    MapEntry { hnl: "gelu", emit: Emit::Raw, call: "aten_kernels::drv_gelu(args)", dims: &[], dims_req: &[], arity: Some((1, 2)), sig: Some("Value... -> Tensor") },
    m("relu", Emit::Tensor, "aten_kernels::k_relu(&v0)"),
    m("selu", Emit::Tensor, "aten_kernels::k_selu(&v0)"),
    m("elu", Emit::Tensor, "aten_kernels::k_elu(&v0, v1, v2, v3)"),
    m("celu", Emit::Tensor, "aten_kernels::k_celu(&v0, v1)"),
    m("leaky_relu", Emit::Tensor, "aten_kernels::k_leaky_relu(&v0, v1)"),
    m("prelu", Emit::Tensor, "aten_kernels::k_prelu(&v0, &v1)"),
    m("softplus", Emit::Tensor, "aten_kernels::k_softplus(&v0, v1, v2)"),
    m("softshrink", Emit::Tensor, "aten_kernels::k_softshrink(&v0, v1)"),
    m("hardshrink", Emit::Tensor, "aten_kernels::k_hardshrink(&v0, v1)"),
    m("hardtanh", Emit::Tensor, "aten_kernels::k_hardtanh(&v0, v1, v2)"),
    m("threshold", Emit::Tensor, "aten_kernels::k_threshold(&v0, v1, v2)"),
    m("erf", Emit::Tensor, "aten_kernels::k_erf(&v0)"),
    m("erfc", Emit::Tensor, "aten_kernels::k_erfc(&v0)"),
    m("sinc", Emit::Tensor, "aten_kernels::k_sinc(&v0)"),
    m("deg2rad", Emit::Tensor, "aten_kernels::k_degrees_to_radians(&v0)"),
    m("rad2deg", Emit::Tensor, "aten_kernels::k_radians_to_degrees(&v0)"),
    m("logit", Emit::Tensor, "aten_kernels::k_logit(&v0, v1)"),
    m("nan_to_num", Emit::Tensor, "aten_kernels::k_nan_to_num(&v0, v1, v2, v3)"),
    m("isfinite", Emit::Tensor, "aten_kernels::k_isfinite(&v0)"),
    m("isinf", Emit::Tensor, "aten_kernels::k_isinf(&v0)"),
    m("isnan", Emit::Tensor, "aten_kernels::k_isnan(&v0)"),
    m("isposinf", Emit::Tensor, "aten_kernels::k_isposinf(&v0)"),
    m("isneginf", Emit::Tensor, "aten_kernels::k_isneginf(&v0)"),
    m("signbit", Emit::Tensor, "aten_kernels::k_signbit(&v0)"),
    m("logical_not", Emit::Tensor, "aten_kernels::k_logical_not(&v0)"),
    // --- binary ---------------------------------------------------------
    m("atan2", Emit::Tensor, "aten_kernels::k_atan2(&v0, &v1)"),
    m("hypot", Emit::Tensor, "aten_kernels::k_hypot(&v0, &v1)"),
    m("logaddexp", Emit::Tensor, "aten_kernels::k_logaddexp(&v0, &v1)"),
    m("logaddexp2", Emit::Tensor, "aten_kernels::k_logaddexp2(&v0, &v1)"),
    m("maximum", Emit::Tensor, "aten_kernels::k_maximum(&v0, &v1)"),
    m("minimum", Emit::Tensor, "aten_kernels::k_minimum(&v0, &v1)"),
    m("nextafter", Emit::Tensor, "aten_kernels::k_nextafter(&v0, &v1)"),
    m("heaviside", Emit::Tensor, "aten_kernels::k_heaviside(&v0, &v1)"),
    m("logical_and", Emit::Tensor, "aten_kernels::k_logical_and(&v0, &v1)"),
    m("logical_or", Emit::Tensor, "aten_kernels::k_logical_or(&v0, &v1)"),
    m("logical_xor", Emit::Tensor, "aten_kernels::k_logical_xor(&v0, &v1)"),
    // Scalar-or-Tensor rhs variants → raw drivers.
    MapEntry { hnl: "fmod", emit: Emit::Raw, call: "aten_kernels::drv_bin_ss(args, \"tensor.fmod\", aten_kernels::f_fmod)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "floor_divide", emit: Emit::Raw, call: "aten_kernels::drv_bin_ss(args, \"tensor.floor_divide\", aten_kernels::f_floor_divide)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "copysign", emit: Emit::Raw, call: "aten_kernels::drv_bin_ss(args, \"tensor.copysign\", aten_kernels::f_copysign)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "remainder", emit: Emit::Raw, call: "aten_kernels::k_remainder(args)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "pow", emit: Emit::Raw, call: "aten_kernels::k_pow(args)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "float_power", emit: Emit::Raw, call: "aten_kernels::k_pow(args)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "eq", emit: Emit::Raw, call: "aten_kernels::k_compare(args, aten_kernels::CmpKind::Eq)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "ne", emit: Emit::Raw, call: "aten_kernels::k_compare(args, aten_kernels::CmpKind::Ne)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "lt", emit: Emit::Raw, call: "aten_kernels::k_compare(args, aten_kernels::CmpKind::Lt)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "le", emit: Emit::Raw, call: "aten_kernels::k_compare(args, aten_kernels::CmpKind::Le)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "gt", emit: Emit::Raw, call: "aten_kernels::k_compare(args, aten_kernels::CmpKind::Gt)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "ge", emit: Emit::Raw, call: "aten_kernels::k_compare(args, aten_kernels::CmpKind::Ge)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "where", emit: Emit::Raw, call: "aten_kernels::k_where(args)", dims: &[], dims_req: &[], arity: Some((1, 3)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "clamp", emit: Emit::Raw, call: "aten_kernels::k_clamp(args)", dims: &[], dims_req: &[], arity: Some((1, 3)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "clamp_min", emit: Emit::Raw, call: "aten_kernels::k_clamp_minmax(args, true)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "clamp_max", emit: Emit::Raw, call: "aten_kernels::k_clamp_minmax(args, false)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "lerp", emit: Emit::Raw, call: "aten_kernels::k_lerp(args)", dims: &[], dims_req: &[], arity: Some((3, 3)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "xlogy", emit: Emit::Raw, call: "aten_kernels::k_xlogy(args)", dims: &[], dims_req: &[], arity: Some((2, 2)), sig: Some("Value... -> Tensor")},
    // --- reductions -------------------------------------------------------
    MapEntry { hnl: "sum", emit: Emit::Tensor, call: "aten_kernels::reduce_sum(&v0, v1, v2)", dims: &[1], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "prod", emit: Emit::Tensor, call: "aten_kernels::reduce_prod(&v0, v1, v2)", dims: &[1], dims_req: &[], arity: Some((1, 3)), sig: None },
    MapEntry { hnl: "mean", emit: Emit::Tensor, call: "aten_kernels::reduce_mean(&v0, v1, v2)", dims: &[1], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "amax", emit: Emit::Tensor, call: "aten_kernels::reduce_amax(&v0, v1, v2)", dims: &[1], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "amin", emit: Emit::Tensor, call: "aten_kernels::reduce_amin(&v0, v1, v2)", dims: &[1], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "argmax", emit: Emit::Tensor, call: "aten_kernels::reduce_arg(&v0, v1, v2, true, \"argmax\")", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "argmin", emit: Emit::Tensor, call: "aten_kernels::reduce_arg(&v0, v1, v2, false, \"argmin\")", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "logsumexp", emit: Emit::Tensor, call: "aten_kernels::reduce_logsumexp(&v0, v1, v2)", dims: &[], dims_req: &[1], arity: None, sig: None },
    MapEntry { hnl: "var", emit: Emit::Raw, call: "aten_kernels::k_var_std(args, false)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "std", emit: Emit::Raw, call: "aten_kernels::k_var_std(args, true)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "all", emit: Emit::Raw, call: "aten_kernels::k_all_any(args, true)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "any", emit: Emit::Raw, call: "aten_kernels::k_all_any(args, false)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "cumsum", emit: Emit::Tensor, call: "aten_kernels::cum_along(&v0, v1, false, \"cumsum\")", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "cumprod", emit: Emit::Tensor, call: "aten_kernels::cum_along(&v0, v1, true, \"cumprod\")", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "softmax", emit: Emit::Tensor, call: "aten_kernels::softmax_dim(&v0, v1, false, \"softmax\")", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "log_softmax", emit: Emit::Tensor, call: "aten_kernels::softmax_dim(&v0, v1, true, \"log_softmax\")", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "layer_norm", emit: Emit::Raw, call: "aten_kernels::drv_layer_norm(args)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "rms_norm", emit: Emit::Raw, call: "aten_kernels::drv_rms_norm(args)", dims: &[], dims_req: &[], arity: None, sig: None },
    m("glu", Emit::Tensor, "aten_kernels::k_glu(&v0, v1)"),
    // --- sort / select -----------------------------------------------------
    MapEntry { hnl: "sort", emit: Emit::Raw, call: "aten_kernels::k_sort(args, false)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "argsort", emit: Emit::Raw, call: "aten_kernels::k_sort(args, true)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "topk", emit: Emit::Raw, call: "aten_kernels::k_topk(args)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "nonzero", emit: Emit::Tensor, call: "aten_kernels::nonzero_impl(&v0)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "argwhere", emit: Emit::Tensor, call: "aten_kernels::nonzero_impl(&v0)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "squeeze", emit: Emit::Raw, call: "aten_kernels::k_squeeze(args)", dims: &[], dims_req: &[], arity: Some((1, 2)), sig: Some("Value... -> Tensor")},
    m("index_select", Emit::Tensor, "aten_kernels::k_index_select(&v0, v1, &v2)"),
    m("gather", Emit::Tensor, "aten_kernels::k_gather(&v0, v1, &v2)"),
    m("index_add", Emit::Tensor, "aten_kernels::k_index_add(&v0, v1, &v2, &v3, v4)"),
    m("index_fill", Emit::Tensor, "aten_kernels::k_index_fill(&v0, v1, &v2, v3)"),
    m("take", Emit::Tensor, "aten_kernels::k_take(&v0, &v1)"),
    // --- shape --------------------------------------------------------------
    m("unsqueeze", Emit::Tensor, "aten_kernels::k_unsqueeze(&v0, v1)"),
    m("flatten", Emit::Tensor, "aten_kernels::k_flatten(&v0, v1, v2)"),
    MapEntry { hnl: "expand", emit: Emit::Raw, call: "aten_kernels::drv_expand(args)", dims: &[], dims_req: &[], arity: Some((2, 16)), sig: Some("Value... -> Tensor") },
    m("expand_as", Emit::Tensor, "aten_kernels::k_expand_as(&v0, &v1)"),
    m("narrow", Emit::Tensor, "aten_kernels::k_narrow(&v0, v1, v2, v3)"),
    m("slice", Emit::Tensor, "aten_kernels::k_slice_opt(&v0, v1, v2, v3, v4)"),
    MapEntry { hnl: "chunk", emit: Emit::Raw, call: "aten_kernels::k_chunk(args)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "split", emit: Emit::Raw, call: "aten_kernels::k_split(args)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "stack", emit: Emit::Raw, call: "aten_kernels::k_stack(args)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "unbind", emit: Emit::Raw, call: "aten_kernels::k_unbind(args)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "repeat", emit: Emit::Raw, call: "aten_kernels::drv_repeat(args)", dims: &[], dims_req: &[], arity: Some((2, 16)), sig: Some("Value... -> Tensor") },
    m("flip", Emit::Tensor, "aten_kernels::k_flip(&v0, &v1)"),
    m("triu", Emit::Tensor, "aten_kernels::k_triu_tril(&v0, v1, true)"),
    m("tril", Emit::Tensor, "aten_kernels::k_triu_tril(&v0, v1, false)"),
    m("diag", Emit::Tensor, "aten_kernels::k_diag(&v0, v1)"),
    m("diagonal", Emit::Tensor, "aten_kernels::k_diagonal(&v0, v1, v2, v3)"),
    m("trace", Emit::Tensor, "aten_kernels::k_trace(&v0)"),
    m("t", Emit::Tensor, "aten_kernels::k_t(&v0)"),
    m("mT", Emit::Tensor, "aten_kernels::k_mT(&v0)"),
    m("swapaxes", Emit::Tensor, "aten_kernels::k_swapaxes(&v0, v1, v2)"),
    MapEntry { hnl: "moveaxis", emit: Emit::Tensor, call: "aten_kernels::k_moveaxis(&v0, v1.as_deref().unwrap_or(&[]), v2.as_deref().unwrap_or(&[]))", dims: &[1, 2], dims_req: &[], arity: None, sig: None },
    m("rot90", Emit::Tensor, "aten_kernels::k_rot90(&v0, v1, v2.as_deref().unwrap_or(&[0, 1]))"),
    m("contiguous", Emit::Tensor, "aten_kernels::k_identity(&v0)"),
    m("clone", Emit::Tensor, "aten_kernels::k_identity(&v0)"),
    m("detach", Emit::Tensor, "aten_kernels::k_identity(&v0)"),
    m("reshape", Emit::Tensor, "aten_kernels::k_reshape(&v0, &v1)"),
    MapEntry { hnl: "view", emit: Emit::Raw, call: "aten_kernels::drv_view(args)", dims: &[], dims_req: &[], arity: Some((2, 16)), sig: Some("Value... -> Tensor") },
    m("pixel_shuffle", Emit::Tensor, "aten_kernels::k_pixel_shuffle(&v0, v1)"),
    m("pixel_unshuffle", Emit::Tensor, "aten_kernels::k_pixel_unshuffle(&v0, v1)"),
    m("constant_pad_nd", Emit::Tensor, "aten_kernels::k_constant_pad_nd(&v0, &v1, v2)"),
    // --- linalg -------------------------------------------------------------
    m("mm", Emit::Tensor, "aten_kernels::k_mm(&v0, &v1)"),
    m("bmm", Emit::Tensor, "aten_kernels::k_bmm(&v0, &v1)"),
    MapEntry { hnl: "addmm", emit: Emit::Raw, call: "aten_kernels::k_addmm(args, false, false)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "addbmm", emit: Emit::Raw, call: "aten_kernels::k_addmm(args, true, false)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "baddbmm", emit: Emit::Raw, call: "aten_kernels::k_addmm(args, true, true)", dims: &[], dims_req: &[], arity: None, sig: None },
    m("mv", Emit::Tensor, "aten_kernels::k_mv(&v0, &v1)"),
    m("dot", Emit::Tensor, "aten_kernels::k_dot(&v0, &v1)"),
    m("vdot", Emit::Tensor, "aten_kernels::k_dot(&v0, &v1)"),
    m("ger", Emit::Tensor, "aten_kernels::k_ger(&v0, &v1)"),
    m("outer", Emit::Tensor, "aten_kernels::k_ger(&v0, &v1)"),
    MapEntry { hnl: "addr", emit: Emit::Raw, call: "aten_kernels::k_addr(args)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "addmv", emit: Emit::Raw, call: "aten_kernels::k_addmv(args)", dims: &[], dims_req: &[], arity: None, sig: None },
    // --- constructors / metadata --------------------------------------------
    m("full", Emit::Tensor, "aten_kernels::k_full(&v0, v1)"),
    m("full_like", Emit::Tensor, "aten_kernels::k_full_like(&v0, v1)"),
    m("zeros_like", Emit::Tensor, "aten_kernels::k_zeros_like(&v0)"),
    m("ones_like", Emit::Tensor, "aten_kernels::k_ones_like(&v0)"),
    MapEntry { hnl: "arange", emit: Emit::Raw, call: "aten_kernels::k_arange(args)", dims: &[], dims_req: &[], arity: Some((1, 4)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "linspace", emit: Emit::Raw, call: "aten_kernels::k_linspace(args, false)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "logspace", emit: Emit::Raw, call: "aten_kernels::k_linspace(args, true)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "eye", emit: Emit::Raw, call: "aten_kernels::k_eye(args)", dims: &[], dims_req: &[], arity: Some((1, 2)), sig: Some("Value... -> Tensor")},
    MapEntry { hnl: "size", emit: Emit::Raw, call: "aten_kernels::k_size(args)", dims: &[], dims_req: &[], arity: Some((1, 2)), sig: Some("Value... -> Value")},
    MapEntry { hnl: "stride", emit: Emit::Value, call: "aten_kernels::k_stride(&v0)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "numel", emit: Emit::Value, call: "Ok(aten_kernels::k_numel(&v0))", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "dim", emit: Emit::Value, call: "Ok(aten_kernels::k_dim(&v0))", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "item", emit: Emit::Value, call: "aten_kernels::k_item(&v0)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "is_contiguous", emit: Emit::Value, call: "Ok(aten_kernels::k_is_contiguous(&v0))", dims: &[], dims_req: &[], arity: None, sig: None },
    // --- pooling --------------------------------------------------------------
    MapEntry { hnl: "avg_pool2d", emit: Emit::Raw, call: "aten_kernels::drv_avg_pool2d(args)", dims: &[], dims_req: &[], arity: None, sig: None },
    // --- seam closures (formerly `# seam` entries) -----------------------------
    // einsum's `str equation` and to.dtype's `ScalarType dtype` are required
    // args of unsupported types — Raw drivers read them from `args` directly.
    MapEntry { hnl: "einsum", emit: Emit::Raw, call: "aten_kernels::k_einsum(args)", dims: &[], dims_req: &[], arity: Some((2, 3)), sig: Some("Value... -> Tensor") },
    m("tensordot", Emit::Tensor, "aten_kernels::k_tensordot(&v0, &v1, &v2, &v3)"),
    m("kron", Emit::Tensor, "aten_kernels::k_kron(&v0, &v1)"),
    m("cross", Emit::Tensor, "aten_kernels::k_cross(&v0, &v1, v2)"),
    m("norm", Emit::Tensor, "aten_kernels::k_norm(&v0, v1)"),
    m("renorm", Emit::Tensor, "aten_kernels::k_renorm(&v0, v1, v2, v3)"),
    MapEntry { hnl: "normalize", emit: Emit::Tensor, call: "aten_kernels::k_normalize(&v0, v1, v2.as_deref(), v3)", dims: &[2], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "median", emit: Emit::Raw, call: "aten_kernels::k_median(args)", dims: &[], dims_req: &[], arity: Some((1, 3)), sig: Some("Value... -> Value") },
    MapEntry { hnl: "mode", emit: Emit::Value, call: "aten_kernels::k_mode(&v0, v1, v2)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "kthvalue", emit: Emit::Value, call: "aten_kernels::k_kthvalue(&v0, v1, v2, v3)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "quantile", emit: Emit::Raw, call: "aten_kernels::k_quantile(args)", dims: &[], dims_req: &[], arity: None, sig: Some("Value... -> Tensor") },
    MapEntry { hnl: "searchsorted", emit: Emit::Raw, call: "aten_kernels::k_searchsorted(args)", dims: &[], dims_req: &[], arity: None, sig: Some("Value... -> Tensor") },
    MapEntry { hnl: "bincount", emit: Emit::Tensor, call: "aten_kernels::k_bincount(&v0, v1.as_ref(), v2)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "meshgrid", emit: Emit::Value, call: "aten_kernels::k_meshgrid(&v0)", dims: &[], dims_req: &[], arity: None, sig: None },
    m("corrcoef", Emit::Tensor, "aten_kernels::k_corrcoef(&v0)"),
    MapEntry { hnl: "to", emit: Emit::Raw, call: "aten_kernels::k_to(args)", dims: &[], dims_req: &[], arity: None, sig: Some("Value... -> Tensor") },
    // --- promotions: trivially supported by existing kernels ---------------
    MapEntry { hnl: "max", emit: Emit::Raw, call: "aten_kernels::k_max_min(args, true)", dims: &[], dims_req: &[], arity: Some((1, 3)), sig: Some("Value... -> Value") },
    MapEntry { hnl: "min", emit: Emit::Raw, call: "aten_kernels::k_max_min(args, false)", dims: &[], dims_req: &[], arity: Some((1, 3)), sig: Some("Value... -> Value") },
    m("count_nonzero", Emit::Tensor, "aten_kernels::k_count_nonzero(&v0, v1)"),
    m("inner", Emit::Tensor, "aten_kernels::k_dot(&v0, &v1)"),
    m("dist", Emit::Tensor, "aten_kernels::k_dist(&v0, &v1, v2)"),
    MapEntry { hnl: "allclose", emit: Emit::Value, call: "aten_kernels::k_allclose(&v0, &v1, v2, v3, v4)", dims: &[], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "is_nonzero", emit: Emit::Value, call: "aten_kernels::k_is_nonzero(&v0)", dims: &[], dims_req: &[], arity: None, sig: None },
    m("msort", Emit::Tensor, "aten_kernels::k_msort(&v0)"),
    MapEntry { hnl: "nansum", emit: Emit::Tensor, call: "aten_kernels::k_nansum(&v0, v1.as_deref(), v2)", dims: &[1], dims_req: &[], arity: None, sig: None },
    MapEntry { hnl: "nanmean", emit: Emit::Tensor, call: "aten_kernels::k_nanmean(&v0, v1.as_deref(), v2)", dims: &[1], dims_req: &[], arity: None, sig: None },
    m("cdist", Emit::Tensor, "aten_kernels::k_cdist(&v0, &v1, v2, v3)"),
    m("pdist", Emit::Tensor, "aten_kernels::k_pdist(&v0, v1)"),
];

const fn m(hnl: &'static str, emit: Emit, call: &'static str) -> MapEntry {
    MapEntry { hnl, emit, call, dims: &[], dims_req: &[], arity: None, sig: None }
}

fn map_for(hnl: &str) -> Option<&'static MapEntry> {
    MAP.iter().find(|e| e.hnl == hnl)
}

// ---------------------------------------------------------------------------
// Resolution + emission
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Status {
    Ported,
    Seam,
    Missing,
    Unsupported,
}

struct Resolved {
    entry: Entry,
    map: Option<&'static MapEntry>,
    schema: Option<Schema>,
    status: Status,
    reason: String,
    /// Decode bindings: (pos, kind, rust_let_line, binding_name, rust_type)
    bindings: Vec<Binding>,
    min_args: usize,
    max_args: usize,
    ret_hnl: String,
    sig: String,
}

struct Binding {
    pos: usize,
    name: String, // v{pos}
    let_line: String,
    ty: &'static str, // Rust type for doc purposes
}

/// Normalize JIT alias annotations in a return type: `Tensor(a)` →
/// `Tensor`, `Tensor(a)[ ]` → `Tensor[]`. Tuple returns like
/// `(Tensor values, Tensor indices)` are untouched.
fn norm_ret(ret: &str) -> String {
    let mut out = ret.to_string();
    while let Some(i) = out.find("Tensor(") {
        if let Some(j) = out[i..].find(')') {
            out.replace_range(i..i + j + 1, "Tensor");
        } else {
            break;
        }
    }
    out
}

fn ret_hnl(ret: &str) -> &'static str {
    match norm_ret(ret).as_str() {
        "Tensor" => "Tensor",
        "Tensor[]" => "Array",
        "int" => "Int",
        "int[]" | "SymInt[]" => "(List Int)",
        "bool" => "Bool",
        "float" => "Float",
        _ => "Value",
    }
}

fn is_out_or_inplace(s: &Schema) -> bool {
    s.overload == "out"
        || s.schema.contains(".out(")
        || s.name.ends_with('_')
        || s.name.starts_with('_')
        || s.args.iter().any(|a| a.mutated || a.ty.contains('!'))
        || s.ret.contains('!')
}

fn resolve(
    inv: &Inventory,
    e: &Entry,
    map: Option<&'static MapEntry>,
) -> Resolved {
    // Schema lookup: "aten::name" or "aten::name.overload".
    let body = e.query.trim_start_matches("aten::");
    let (name, overload) = match body.split_once('.') {
        Some((n, o)) => (n.to_string(), Some(o.to_string())),
        None => (body.to_string(), None),
    };
    let mut cands: Vec<&Schema> = inv
        .schemas
        .iter()
        .filter(|s| s.name == name)
        .filter(|s| overload.as_deref().map_or(true, |o| s.overload == o))
        .collect();
    // Prefer canonical (non-out, non-mutated, overload "" first, then
    // shortest schema string for determinism).
    cands.sort_by_key(|s| {
        (
            is_out_or_inplace(s),
            !s.overload.is_empty(),
            s.schema.len(),
            s.schema.clone(),
        )
    });
    let chosen = cands.first().copied();

    let mut r = Resolved {
        entry: Entry { hnl: e.hnl.clone(), query: e.query.clone(), seam: e.seam, note: e.note.clone() },
        map,
        schema: chosen.cloned(),
        status: Status::Missing,
        reason: String::new(),
        bindings: Vec::new(),
        min_args: 0,
        max_args: 0,
        ret_hnl: "Value".into(),
        sig: "Value... -> Value".into(),
    };

    let Some(s) = chosen else {
        r.status = Status::Missing;
        r.reason = format!("no aten:: schema matched {}", e.query);
        return r;
    };
    r.ret_hnl = ret_hnl(&s.ret).to_string();

    // Declared seams win over feature-gate verdicts — the stub is
    // honest either way; the reason records why.
    if e.seam || map.is_none() {
        r.status = Status::Seam;
        r.reason = if e.seam {
            format!("seam: {}", e.note)
        } else {
            "no kernel mapping in KERNEL_MAP".into()
        };
        return r;
    }
    if is_out_or_inplace(s) {
        r.status = Status::Unsupported;
        r.reason = "out/inplace/mutated variant".into();
        return r;
    }

    // Feature gate on args. Raw drivers take `args: &[Value]` verbatim —
    // unsupported arg types (`str equation`, `ScalarType dtype`) are the
    // kernel's problem, not the decoder's, so the gate is skipped for
    // them (and optional unsupported args are NOT "dropped" — the raw
    // driver still sees them).
    let raw_emit = map.is_some_and(|m| m.emit == Emit::Raw);
    let mut unsupported_required: Vec<String> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    if !raw_emit {
        for a in &s.args {
            if a.ty.contains("...") || a.ty.contains("Tensor?[]") {
                unsupported_required.push(format!("{}: {}", a.name, a.ty));
                continue;
            }
            let (kind, _nullable) = classify(&a.ty);
            if kind == Kind::Unsupported {
                let optional = a.default.is_some() || strip_nullable(&a.ty).1;
                if optional {
                    dropped.push(format!("{}:{}", a.name, a.ty));
                } else {
                    unsupported_required.push(format!("{}: {}", a.name, a.ty));
                }
            }
        }
        if !unsupported_required.is_empty() {
            r.status = Status::Unsupported;
            r.reason = format!("unsupported required arg types: {}", unsupported_required.join(", "));
            return r;
        }
    }

    // Decode plan (skipped for Raw emit).
    let map = map.unwrap();
    let mut last_required = 0usize;
    let mut sig_params: Vec<String> = Vec::new();
    let mut all_required = true;
    for (i, a) in s.args.iter().enumerate() {
        let optional = a.default.is_some() || strip_nullable(&a.ty).1;
        if !optional {
            last_required = i + 1;
        }
        let (kind, _) = classify(&a.ty);
        if kind == Kind::Unsupported {
            continue; // dropped optional
        }
        if optional {
            all_required = false;
        }
        let hnl_ty = match kind {
            Kind::Tensor | Kind::TensorOpt => "Tensor",
            Kind::TensorList => "Array",
            Kind::Int | Kind::IntOpt | Kind::IntList | Kind::IntListOpt | Kind::DimList | Kind::DimListReq => "Int",
            Kind::Float | Kind::FloatOpt | Kind::FloatList => "Float",
            Kind::Bool | Kind::BoolOpt => "Bool",
            Kind::Unsupported => unreachable!(),
        };
        // Optional params can't be expressed in a fixed tuple → Value...
        if optional {
            sig_params.push("Value".into());
        } else {
            sig_params.push(
                match kind {
                    Kind::Tensor => "Tensor".into(),
                    Kind::TensorList => "Array".into(),
                    Kind::Int | Kind::IntOpt | Kind::IntList | Kind::IntListOpt | Kind::DimList | Kind::DimListReq => {
                        if matches!(kind, Kind::IntList | Kind::IntListOpt | Kind::DimList | Kind::DimListReq) {
                            "(List Int)".into()
                        } else {
                            "Int".into()
                        }
                    }
                    Kind::Float | Kind::FloatOpt | Kind::FloatList => "Float".into(),
                    Kind::Bool | Kind::BoolOpt => "Bool".into(),
                    _ => hnl_ty.into(),
                },
            );
        }
        if map.emit != Emit::Raw {
            if let Some(b) = decode_binding(i, a, map) {
                r.bindings.push(b);
            }
        }
    }
    r.min_args = last_required;
    r.max_args = s.args.len();
    if let Some((lo, hi)) = map.arity {
        r.min_args = lo;
        r.max_args = hi;
    }

    // Signature: precise tuple only when every decoded param is required
    // and there are at most 4 of them; otherwise Value... (the conv2d
    // convention — `null`/missing selects defaults).
    r.sig = if let Some(sig) = map.sig {
        sig.to_string()
    } else if all_required && sig_params.len() <= 4 && !sig_params.is_empty() {
        if sig_params.len() == 1 {
            format!("{} -> {}", sig_params[0], r.ret_hnl)
        } else {
            format!("({}) -> {}", sig_params.join(" * "), r.ret_hnl)
        }
    } else {
        format!("Value... -> {}", r.ret_hnl)
    };
    if !dropped.is_empty() {
        r.reason = format!("dropped unsupported optional args: {}", dropped.join(", "));
    }
    r.status = Status::Ported;
    r
}

fn decode_binding(pos: usize, a: &Arg, map: &MapEntry) -> Option<Binding> {
    let name = format!("v{pos}");
    let d = parse_default(a.default.as_deref());
    let is_dim = map.dims.contains(&pos);
    let is_dim_req = map.dims_req.contains(&pos);
    let (kind, _n) = classify(&a.ty);
    let optional = a.default.is_some() || strip_nullable(&a.ty).1;
    let line = match kind {
        Kind::Tensor => format!("let {name} = require_tensor(args, {pos})?;"),
        Kind::TensorOpt => format!("let {name} = opt_tensor(args, {pos})?;"),
        Kind::TensorList => format!(
            "let {name} = aten_kernels::req_tensors(aten_kernels::arg_at(args, {pos})?)?;"
        ),
        Kind::Int => {
            if is_dim {
                // Dim-like `int` arg — accept int OR list, `None` → all.
                format!("let {name} = aten_kernels::opt_dimlist(args.get({pos}), None)?;")
            } else if optional {
                let dv = d.rust_i64().unwrap_or("0".into());
                format!("let {name} = opt_i64(args.get({pos}), {dv})?;")
            } else if is_dim_req {
                format!("let {name} = aten_kernels::req_dimlist(aten_kernels::arg_at(args, {pos})?)?;")
            } else {
                format!("let {name} = req_i64(aten_kernels::arg_at(args, {pos})?)?;")
            }
        }
        Kind::IntOpt => format!("let {name} = opt_i64n(args.get({pos}))?;"),
        Kind::IntList | Kind::IntListOpt => {
            if is_dim {
                // `int[1] dim=[]` default means "all dims" → None.
                let dv = match &d {
                    Default_::Ints(v) if v.is_empty() => "None".to_string(),
                    Default_::Ints(v) => format!("Some(&[{}])", v.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")),
                    Default_::None => "None".to_string(),
                    _ => "None".to_string(),
                };
                format!("let {name} = aten_kernels::opt_dimlist(args.get({pos}), {dv})?;")
            } else if is_dim_req {
                format!("let {name} = aten_kernels::req_dimlist(aten_kernels::arg_at(args, {pos})?)?;")
            } else if optional {
                let dv = d
                    .rust_ints()
                    .map(|s| format!("Some({s})"))
                    .unwrap_or("None".into());
                format!("let {name} = opt_i64s(args.get({pos}), {dv})?;")
            } else {
                format!("let {name} = req_i64s(aten_kernels::arg_at(args, {pos})?)?;")
            }
        }
        Kind::Float => {
            if optional {
                let dv = d.rust_f64().unwrap_or("0.0".into());
                format!("let {name} = opt_f64(args.get({pos}), {dv})?;")
            } else {
                format!("let {name} = req_f64(aten_kernels::arg_at(args, {pos})?)?;")
            }
        }
        Kind::FloatOpt => format!("let {name} = opt_f64n(args.get({pos}))?;"),
        Kind::FloatList => format!(
            "let {name} = aten_kernels::req_f64s(aten_kernels::arg_at(args, {pos})?)?;"
        ),
        Kind::Bool => {
            if optional {
                let dv = d.rust_bool().unwrap_or("false".into());
                format!("let {name} = opt_bool(args.get({pos}), {dv})?;")
            } else {
                format!("let {name} = req_bool(aten_kernels::arg_at(args, {pos})?)?;")
            }
        }
        Kind::BoolOpt => format!("let {name} = opt_booln(args.get({pos}))?;"),
        Kind::DimList | Kind::DimListReq => unreachable!(),
        Kind::Unsupported => return None,
    };
    let ty = match kind {
        Kind::Tensor => "Tensor",
        Kind::TensorOpt => "Option<Tensor>",
        Kind::TensorList => "Vec<Tensor>",
        Kind::Int | Kind::IntOpt => "i64",
        Kind::IntList | Kind::IntListOpt | Kind::DimList | Kind::DimListReq => "Vec<i64>",
        Kind::Float | Kind::FloatOpt => "f64",
        Kind::FloatList => "Vec<f64>",
        Kind::Bool | Kind::BoolOpt => "bool",
        Kind::Unsupported => unreachable!(),
    };
    Some(Binding { pos, name, let_line: line, ty })
}

// ---------------------------------------------------------------------------
// File emission
// ---------------------------------------------------------------------------

/// Escape a string for embedding in a Rust `"…"` literal (schema strings
/// can contain `"` defaults like `interpolation="linear"`).
fn rust_lit(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn camel(hnl: &str) -> String {
    let mut out = String::new();
    for (i, part) in hnl.split('_').enumerate() {
        if i > 0 {
            out.push_str(&part[..1].to_uppercase());
            out.push_str(&part[1..]);
        } else {
            out.push_str(&part[..1].to_uppercase());
            out.push_str(&part[1..]);
        }
    }
    out
}

fn op_name(hnl: &str) -> String {
    format!("tensor.{hnl}")
}

fn gen_rs(resolved: &[Resolved], inv: &Inventory) -> String {
    let mut s = String::new();
    writeln!(s, "//! GENERATED FILE — do not edit.").unwrap();
    writeln!(s, "//!").unwrap();
    writeln!(s, "//! Generated by `cargo run --bin aten-port-gen` from").unwrap();
    writeln!(s, "//! `tools/aten_port_list.txt` × `tools/aten_schemas.json`").unwrap();
    writeln!(s, "//! ({} aten:: schemas, torch {}).", inv.schemas.len(), inv.torch).unwrap();
    writeln!(s, "//! Kernel semantics live in `super::aten_kernels`; this file").unwrap();
    writeln!(s, "//! is only the decode → call glue. `# seam` entries emit").unwrap();
    writeln!(s, "//! stubs that fail loudly.").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "#![allow(unused_imports, unused_variables)] // decoder set is a superset; some decoded optionals are intentionally ignored").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "use handl_platform::{{EvalContext, EvalError, Function}};").unwrap();
    writeln!(s, "use serde_json::Value;").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "use super::aten_kernels;").unwrap();
    // Scalar/list decoders live in `aten_kernels`; tensor/codec helpers
    // in `codec`.
    writeln!(s, "use super::aten_kernels::{{opt_bool, opt_booln, opt_f64, opt_f64n, opt_i64,").unwrap();
    writeln!(s, "    opt_i64n, opt_i64s, req_bool, req_f64, req_i64, req_i64s}};").unwrap();
    writeln!(s, "use super::codec::{{encode_tensor, opt_tensor, require_arg_range, require_tensor}};").unwrap();
    writeln!(s).unwrap();

    for r in resolved {
        let hnl_full = op_name(&r.entry.hnl);
        let struct_name = camel(&r.entry.hnl);
        let schema_str = r
            .schema
            .as_ref()
            .map(|s| s.schema.as_str())
            .unwrap_or("<no schema>");
        writeln!(s, "/// `{hnl_full}` — ATen port of `{schema_str}`.").unwrap();
        if r.status == Status::Seam {
            writeln!(s, "/// SEAM: {}", r.reason).unwrap();
        } else if !r.reason.is_empty() {
            writeln!(s, "/// {}", r.reason).unwrap();
        }
        writeln!(s, "pub(crate) struct {struct_name};").unwrap();
        writeln!(s, "impl Function for {struct_name} {{").unwrap();
        writeln!(s, "    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {{").unwrap();
        match r.status {
            Status::Seam | Status::Missing | Status::Unsupported => {
                writeln!(s, "        let _ = args; // seam stub — no decode").unwrap();
                writeln!(
                    s,
                    "        Err(EvalError::Other(\"{hnl_full}: ATen port seam — {} — TODO: handwritten kernel required\".into()))",
                    rust_lit(schema_str)
                )
                .unwrap();
            }
            Status::Ported => {
                let map = r.map.unwrap();
                writeln!(
                    s,
                    "        require_arg_range(args, {}, {}, \"{hnl_full}\")?;",
                    r.min_args, r.max_args
                )
                .unwrap();
                if map.emit != Emit::Raw {
                    for b in &r.bindings {
                        writeln!(s, "        {}", b.let_line).unwrap();
                    }
                }
                let call = map.call.replace("{args}", "args");
                let call = (0..10).fold(call, |c, i| c.replace(&format!("{{v{i}}}"), &format!("v{i}")));
                match map.emit {
                    Emit::Tensor => {
                        writeln!(s, "        Ok(encode_tensor({call}?))").unwrap();
                    }
                    Emit::Value | Emit::Raw => {
                        writeln!(s, "        {call}").unwrap();
                    }
                }
            }
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
        writeln!(s).unwrap();
    }
    s
}

fn hnl_block(resolved: &[Resolved], kind: &str) -> String {
    // kind: "caps" | "ops" | "handles" | "export"
    let mut s = String::new();
    for r in resolved {
        let full = op_name(&r.entry.hnl);
        match kind {
            "caps" => writeln!(s, "cap \"{full}\"").unwrap(),
            "ops" => {
                writeln!(s, "--- {}", r.schema.as_ref().map(|x| x.schema.as_str()).unwrap_or("<seam>"))
                    .unwrap();
                writeln!(s, "op \"{full}\" : {}", r.sig).unwrap();
                writeln!(s, "  guarantee: True").unwrap();
            }
            "handles" => {
                writeln!(s, "handle \"{full}\" with rust_{}", r.entry.hnl.replace('-', "_"))
                    .unwrap();
            }
            "export" => {
                writeln!(s, "  {}: \"{full}\",", r.entry.hnl).unwrap();
            }
            _ => unreachable!(),
        }
    }
    // export record: last entry must not have a trailing comma? Existing
    // file ends `derive: "tensor.derive"` without comma — keep commas,
    // HNL tolerates trailing commas? Safer: strip trailing comma on last.
    if kind == "export" && s.ends_with(",\n") {
        s.pop();
        s.pop();
        s.push('\n');
    }
    s
}

fn patch_marked(file: &Path, begin: &str, end: &str, body: &str) -> Result<bool, String> {
    let text = fs::read_to_string(file).map_err(|e| format!("read {}: {e}", file.display()))?;
    let b = format!("{begin}\n");
    let e = format!("{end}\n");
    let Some(bs) = text.find(&b) else {
        return Err(format!("{}: missing marker {begin}", file.display()));
    };
    let Some(es) = text.find(&e) else {
        return Err(format!("{}: missing marker {end}", file.display()));
    };
    if es < bs {
        return Err(format!("{}: markers out of order", file.display()));
    }
    let new = format!("{}{}{}", &text[..bs + b.len()], body, &text[es..]);
    if new == text {
        return Ok(false);
    }
    fs::write(file, new).map_err(|e| format!("write {}: {e}", file.display()))?;
    Ok(true)
}

fn spec_json(resolved: &[Resolved], inv: &Inventory) -> Value {
    json!({
        "torch": inv.torch,
        "schema_count": inv.schemas.len(),
        "ops": resolved.iter().map(|r| json!({
            "op": op_name(&r.entry.hnl),
            "schema": r.schema.as_ref().map(|s| s.schema.clone()),
            "status": format!("{:?}", r.status),
            "reason": r.reason,
            "signature": r.sig,
            "min_args": r.min_args,
            "max_args": r.max_args,
            "bindings": r.bindings.iter().map(|b| json!({"pos": b.pos, "name": b.name, "let": b.let_line})).collect::<Vec<_>>(),
            "call": r.map.map(|m| m.call),
        })).collect::<Vec<_>>(),
    })
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn glob_match(pat: &str, s: &str) -> bool {
    // Minimal glob: `*` matches any run, `?` one char.
    let p: Vec<char> = pat.chars().collect();
    let t: Vec<char> = s.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = pi;
            mark = ti;
            pi += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn main() {
    let mut schemas_path = PathBuf::from("tools/aten_schemas.json");
    let mut list_path = PathBuf::from("tools/aten_port_list.txt");
    let mut emit_spec: Option<Option<PathBuf>> = None;
    let mut check = false;
    let mut includes: Vec<String> = Vec::new();
    let mut excludes: Vec<String> = Vec::new();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--schemas" => {
                i += 1;
                schemas_path = PathBuf::from(&args[i]);
            }
            "--list" => {
                i += 1;
                list_path = PathBuf::from(&args[i]);
            }
            "--include" => {
                i += 1;
                includes.push(args[i].clone());
            }
            "--exclude" => {
                i += 1;
                excludes.push(args[i].clone());
            }
            "--emit-spec" => {
                // Optional path: --emit-spec or --emit-spec PATH
                emit_spec = Some(if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    Some(PathBuf::from(&args[i]))
                } else {
                    None
                });
            }
            "--check" => check = true,
            "-h" | "--help" => {
                println!("aten-port-gen [--schemas P] [--list P] [--include G]... [--exclude G]... [--emit-spec [P]] [--check]");
                return;
            }
            other => {
                eprintln!("aten-port-gen: unknown arg {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let inv = match load_inventory(&schemas_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("aten-port-gen: {e}");
            std::process::exit(1);
        }
    };
    let entries = match load_list(&list_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("aten-port-gen: {e}");
            std::process::exit(1);
        }
    };

    // Filters match the hnl op name OR the schema query independently —
    // `--include '*sum*'` catches `tensor.sum` and `aten::sum.*` alike.
    let entries: Vec<Entry> = entries
        .into_iter()
        .filter(|e| {
            let op = op_name(&e.hnl);
            let q = e.query.as_str();
            includes.iter().all(|p| glob_match(p, &op) || glob_match(p, q))
                && !excludes.iter().any(|p| glob_match(p, &op) || glob_match(p, q))
        })
        .collect();

    let mut resolved: Vec<Resolved> = Vec::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for e in &entries {
        let map = map_for(&e.hnl);
        let r = resolve(&inv, e, map);
        *counts.entry(format!("{:?}", r.status)).or_default() += 1;
        if r.status == Status::Unsupported {
            eprintln!("unsupported: {} — {}", op_name(&e.hnl), r.reason);
        } else if r.status == Status::Missing {
            eprintln!("missing: {} — {}", op_name(&e.hnl), r.reason);
        }
        resolved.push(r);
    }

    if let Some(spec_path) = emit_spec {
        let spec = spec_json(&resolved, &inv);
        let text = serde_json::to_string_pretty(&spec).unwrap() + "\n";
        match spec_path {
            Some(p) => {
                if let Err(e) = fs::write(&p, &text) {
                    eprintln!("aten-port-gen: write {}: {e}", p.display());
                    std::process::exit(1);
                }
                eprintln!("wrote spec → {}", p.display());
            }
            None => print!("{text}"),
        }
        return;
    }

    // Emit files. `--check` verifies every output is up-to-date without
    // writing — the CI drift gate.
    let rs = gen_rs(&resolved, &inv);
    let rs_path = Path::new("src/ops/aten_generated.rs");
    let hnl_path = Path::new("src/index.hnl");
    let mod_path = Path::new("src/ops/mod.rs");
    let patches: [(&Path, &str, &str, String); 6] = [
        (hnl_path, "--- ATEN-PORT-GEN:BEGIN caps", "--- ATEN-PORT-GEN:END caps", hnl_block(&resolved, "caps")),
        (hnl_path, "--- ATEN-PORT-GEN:BEGIN ops", "--- ATEN-PORT-GEN:END ops", hnl_block(&resolved, "ops")),
        (hnl_path, "--- ATEN-PORT-GEN:BEGIN handles", "--- ATEN-PORT-GEN:END handles", hnl_block(&resolved, "handles")),
        (hnl_path, "--- ATEN-PORT-GEN:BEGIN export", "--- ATEN-PORT-GEN:END export", hnl_block(&resolved, "export")),
        (mod_path, "// ATEN-PORT-GEN:BEGIN ops", "// ATEN-PORT-GEN:END ops", {
            resolved.iter().map(|r| format!("    \"{}\",\n", op_name(&r.entry.hnl))).collect()
        }),
        (mod_path, "// ATEN-PORT-GEN:BEGIN registry", "// ATEN-PORT-GEN:END registry", {
            resolved
                .iter()
                .map(|r| {
                    format!(
                        "    registry.register(\"{}\", Arc::new(aten_generated::{}));\n",
                        op_name(&r.entry.hnl),
                        camel(&r.entry.hnl)
                    )
                })
                .collect()
        }),
    ];
    if check {
        let mut stale = Vec::new();
        let cur = fs::read_to_string(rs_path).unwrap_or_default();
        if cur != rs {
            stale.push("src/ops/aten_generated.rs");
        }
        for (path, begin, end, body) in &patches {
            let text = fs::read_to_string(path).unwrap_or_default();
            let region = format!("{begin}\n{body}{end}\n");
            if !text.contains(&region) {
                stale.push(path.to_str().unwrap());
            }
        }
        if !stale.is_empty() {
            eprintln!("aten-port-gen: --check FAILED — stale: {}", stale.join(", "));
            eprintln!("  regenerate with: cargo run --bin aten_port_gen");
            std::process::exit(1);
        }
        eprintln!("aten-port-gen: --check OK — all generated surfaces up-to-date");
    } else {
        fs::write(rs_path, &rs)
            .map_err(|e| format!("write {}: {e}", rs_path.display()))
            .unwrap();
        for (path, begin, end, body) in &patches {
            if let Err(e) = patch_marked(path, begin, end, body) {
                eprintln!("aten-port-gen: {e}");
                std::process::exit(1);
            }
        }
        eprintln!("aten-port-gen: wrote src/ops/aten_generated.rs + marker regions");
    }

    eprintln!(
        "aten-port-gen: {} entries — {:?}",
        entries.len(),
        counts.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ")
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_covers_aten_types() {
        assert_eq!(classify("Tensor").0, Kind::Tensor);
        assert_eq!(classify("Tensor?").0, Kind::TensorOpt);
        assert_eq!(classify("Tensor[]").0, Kind::TensorList);
        assert_eq!(classify("int").0, Kind::Int);
        assert_eq!(classify("SymInt?").0, Kind::IntOpt);
        assert_eq!(classify("int[]").0, Kind::IntList);
        assert_eq!(classify("int[1]?").0, Kind::IntListOpt);
        assert_eq!(classify("SymInt[2]").0, Kind::IntList);
        assert_eq!(classify("float").0, Kind::Float);
        assert_eq!(classify("Scalar?").0, Kind::FloatOpt);
        assert_eq!(classify("bool").0, Kind::Bool);
        // Alias annotations collapse to the head type.
        assert_eq!(classify("Tensor(a)").0, Kind::Tensor);
        assert_eq!(classify("Tensor(a -> *)").0, Kind::Tensor);
        // Unsupported dtype/device/layout surface.
        assert_eq!(classify("ScalarType?").0, Kind::Unsupported);
        assert_eq!(classify("Device?").0, Kind::Unsupported);
        assert_eq!(classify("str").0, Kind::Unsupported);
        assert_eq!(classify("Dimname").0, Kind::Unsupported);
    }

    #[test]
    fn defaults_parse() {
        assert!(matches!(parse_default(Some("False")), Default_::Bool(false)));
        assert!(matches!(parse_default(Some("1e-5")), Default_::Float(_)));
        assert!(matches!(parse_default(Some("[1, 1]")), Default_::Ints(_)));
        assert!(matches!(parse_default(Some("None")), Default_::None));
        assert_eq!(parse_default(Some("1")).rust_f64().as_deref(), Some("1.0"));
        // `1.` must render as a float literal, not `1`.
        let f = parse_default(Some("1.")).rust_f64().unwrap();
        assert!(f.contains('.'));
    }

    #[test]
    fn ret_aliases_normalize() {
        assert_eq!(ret_hnl("Tensor"), "Tensor");
        assert_eq!(ret_hnl("Tensor(a)"), "Tensor");
        assert_eq!(ret_hnl("Tensor(a)[]"), "Array");
        assert_eq!(ret_hnl("(Tensor values, Tensor indices)"), "Value");
        assert_eq!(ret_hnl("int[]"), "(List Int)");
        assert_eq!(ret_hnl("bool"), "Bool");
    }

    #[test]
    fn glob_matching() {
        assert!(glob_match("aten::sum*", "aten::sum.dim_IntList"));
        assert!(glob_match("tensor.relu", "tensor.relu"));
        assert!(!glob_match("aten::conv*", "aten::relu"));
        assert!(glob_match("*", "anything"));
    }

    #[test]
    fn list_parses_seams_and_queries() {
        let dir = std::env::temp_dir().join("aten_port_list_test.txt");
        std::fs::write(
            &dir,
            "tensor.relu = aten::relu\n# comment\ntensor.einsum = aten::einsum # seam: eq parser\n",
        )
        .unwrap();
        let entries = load_list(&dir).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(!entries[0].seam);
        assert!(entries[1].seam);
        assert_eq!(entries[1].hnl, "einsum");
        let _ = std::fs::remove_file(&dir);
    }
}
