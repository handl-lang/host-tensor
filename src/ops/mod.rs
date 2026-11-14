//! Rust `Function` impls for the `tensor` module — the handls (handlers)
//! for the `op`/`handle` declarations in `src/index.hnl`.
//!
//! All 28 functions are FFI atoms: a tensor has no literal syntax and no
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
//! | [`conv`] | `conv2d`, `max_pool` + their `*_backward` (NCHW) | `conv.hnl` |
//! | [`norm`] | `batch_norm`, `batch_norm_backward` | `norm.hnl` |
//! | [`shape_ops`] | `reshape`, `transpose`, `concat` + their `*_backward` | `shape_ops.hnl` |
//! | [`derive`] | `derive` + `DERIV_PAIRS` (11 pairs) | `norm.hnl`'s `derive` |
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

pub mod aten_generated;
pub mod aten_kernels;
pub mod codec;
pub mod conv;
pub mod derive;
pub mod elementwise;
pub mod matmul;
pub mod norm;
pub mod shape_ops;

pub use codec::{FromList, Ones, Shape, ToList, Zeros};
pub use conv::{Conv2d, Conv2dBackward, MaxPool, MaxPoolBackward};
pub use derive::{DERIV_PAIRS, Derive};
pub use elementwise::{Add, AddBackward, Div, DivBackward, Mul, MulBackward, Sub, SubBackward};
pub use matmul::{Linear, Matmul, MatmulBackward};
pub use norm::{BatchNorm, BatchNormBackward};
pub use shape_ops::{
    Concat, ConcatBackward, Reshape, ReshapeBackward, Select, Transpose, TransposeBackward,
    Unsqueeze,
};

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
    "tensor.linear",
    "tensor.conv2d",
    "tensor.conv2d_backward",
    "tensor.max_pool",
    "tensor.max_pool_backward",
    "tensor.batch_norm",
    "tensor.reshape",
    "tensor.reshape_backward",
    "tensor.transpose",
    "tensor.transpose_backward",
    "tensor.unsqueeze",
    "tensor.select",
    "tensor.concat",
    "tensor.concat_backward",
    "tensor.batch_norm_backward",
    "tensor.derive",
    // ATEN-PORT-GEN:BEGIN ops
    "tensor.abs",
    "tensor.neg",
    "tensor.exp",
    "tensor.exp2",
    "tensor.expm1",
    "tensor.log",
    "tensor.log1p",
    "tensor.log2",
    "tensor.log10",
    "tensor.sqrt",
    "tensor.rsqrt",
    "tensor.reciprocal",
    "tensor.square",
    "tensor.sin",
    "tensor.cos",
    "tensor.tan",
    "tensor.asin",
    "tensor.acos",
    "tensor.atan",
    "tensor.sinh",
    "tensor.cosh",
    "tensor.tanh",
    "tensor.asinh",
    "tensor.acosh",
    "tensor.atanh",
    "tensor.floor",
    "tensor.ceil",
    "tensor.round",
    "tensor.trunc",
    "tensor.frac",
    "tensor.sign",
    "tensor.sigmoid",
    "tensor.silu",
    "tensor.mish",
    "tensor.gelu",
    "tensor.relu",
    "tensor.selu",
    "tensor.elu",
    "tensor.celu",
    "tensor.leaky_relu",
    "tensor.prelu",
    "tensor.softplus",
    "tensor.softshrink",
    "tensor.hardshrink",
    "tensor.hardtanh",
    "tensor.threshold",
    "tensor.erf",
    "tensor.erfc",
    "tensor.sinc",
    "tensor.deg2rad",
    "tensor.rad2deg",
    "tensor.logit",
    "tensor.nan_to_num",
    "tensor.heaviside",
    "tensor.isfinite",
    "tensor.isinf",
    "tensor.isnan",
    "tensor.isposinf",
    "tensor.isneginf",
    "tensor.signbit",
    "tensor.logical_not",
    "tensor.atan2",
    "tensor.hypot",
    "tensor.logaddexp",
    "tensor.logaddexp2",
    "tensor.maximum",
    "tensor.minimum",
    "tensor.nextafter",
    "tensor.fmod",
    "tensor.floor_divide",
    "tensor.copysign",
    "tensor.remainder",
    "tensor.pow",
    "tensor.float_power",
    "tensor.eq",
    "tensor.ne",
    "tensor.lt",
    "tensor.le",
    "tensor.gt",
    "tensor.ge",
    "tensor.where",
    "tensor.clamp",
    "tensor.clamp_min",
    "tensor.clamp_max",
    "tensor.lerp",
    "tensor.xlogy",
    "tensor.logical_and",
    "tensor.logical_or",
    "tensor.logical_xor",
    "tensor.sum",
    "tensor.prod",
    "tensor.mean",
    "tensor.var",
    "tensor.std",
    "tensor.amax",
    "tensor.amin",
    "tensor.argmax",
    "tensor.argmin",
    "tensor.logsumexp",
    "tensor.all",
    "tensor.any",
    "tensor.cumsum",
    "tensor.cumprod",
    "tensor.softmax",
    "tensor.log_softmax",
    "tensor.layer_norm",
    "tensor.rms_norm",
    "tensor.glu",
    "tensor.sort",
    "tensor.argsort",
    "tensor.topk",
    "tensor.nonzero",
    "tensor.argwhere",
    "tensor.squeeze",
    "tensor.index_select",
    "tensor.gather",
    "tensor.index_add",
    "tensor.index_fill",
    "tensor.take",
    "tensor.flatten",
    "tensor.expand",
    "tensor.expand_as",
    "tensor.narrow",
    "tensor.slice",
    "tensor.chunk",
    "tensor.split",
    "tensor.stack",
    "tensor.unbind",
    "tensor.repeat",
    "tensor.flip",
    "tensor.triu",
    "tensor.tril",
    "tensor.diag",
    "tensor.diagonal",
    "tensor.trace",
    "tensor.t",
    "tensor.mT",
    "tensor.swapaxes",
    "tensor.moveaxis",
    "tensor.rot90",
    "tensor.contiguous",
    "tensor.clone",
    "tensor.detach",
    "tensor.view",
    "tensor.pixel_shuffle",
    "tensor.pixel_unshuffle",
    "tensor.constant_pad_nd",
    "tensor.mm",
    "tensor.bmm",
    "tensor.addmm",
    "tensor.addbmm",
    "tensor.baddbmm",
    "tensor.mv",
    "tensor.dot",
    "tensor.vdot",
    "tensor.ger",
    "tensor.outer",
    "tensor.addr",
    "tensor.addmv",
    "tensor.full",
    "tensor.full_like",
    "tensor.zeros_like",
    "tensor.ones_like",
    "tensor.arange",
    "tensor.linspace",
    "tensor.logspace",
    "tensor.eye",
    "tensor.size",
    "tensor.stride",
    "tensor.numel",
    "tensor.dim",
    "tensor.item",
    "tensor.is_contiguous",
    "tensor.avg_pool2d",
    "tensor.einsum",
    "tensor.tensordot",
    "tensor.kron",
    "tensor.cross",
    "tensor.norm",
    "tensor.renorm",
    "tensor.normalize",
    "tensor.median",
    "tensor.mode",
    "tensor.kthvalue",
    "tensor.quantile",
    "tensor.searchsorted",
    "tensor.bincount",
    "tensor.meshgrid",
    "tensor.corrcoef",
    "tensor.empty",
    "tensor.rand",
    "tensor.normal",
    "tensor.to",
    "tensor.unique",
    "tensor.nonzero_dim",
// ATEN-PORT-GEN:END ops
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
    registry.register("tensor.linear", Arc::new(Linear));
    registry.register("tensor.conv2d", Arc::new(Conv2d));
    registry.register("tensor.conv2d_backward", Arc::new(Conv2dBackward));
    registry.register("tensor.max_pool", Arc::new(MaxPool));
    registry.register("tensor.max_pool_backward", Arc::new(MaxPoolBackward));
    registry.register("tensor.batch_norm", Arc::new(BatchNorm));
    registry.register("tensor.reshape", Arc::new(Reshape));
    registry.register("tensor.reshape_backward", Arc::new(ReshapeBackward));
    registry.register("tensor.transpose", Arc::new(Transpose));
    registry.register("tensor.transpose_backward", Arc::new(TransposeBackward));
    registry.register("tensor.unsqueeze", Arc::new(Unsqueeze));
    registry.register("tensor.select", Arc::new(Select));
    registry.register("tensor.concat", Arc::new(Concat));
    registry.register("tensor.concat_backward", Arc::new(ConcatBackward));
    registry.register("tensor.batch_norm_backward", Arc::new(BatchNormBackward));
    registry.register("tensor.derive", Arc::new(Derive));
    // ATEN-PORT-GEN:BEGIN registry
    registry.register("tensor.abs", Arc::new(aten_generated::Abs));
    registry.register("tensor.neg", Arc::new(aten_generated::Neg));
    registry.register("tensor.exp", Arc::new(aten_generated::Exp));
    registry.register("tensor.exp2", Arc::new(aten_generated::Exp2));
    registry.register("tensor.expm1", Arc::new(aten_generated::Expm1));
    registry.register("tensor.log", Arc::new(aten_generated::Log));
    registry.register("tensor.log1p", Arc::new(aten_generated::Log1p));
    registry.register("tensor.log2", Arc::new(aten_generated::Log2));
    registry.register("tensor.log10", Arc::new(aten_generated::Log10));
    registry.register("tensor.sqrt", Arc::new(aten_generated::Sqrt));
    registry.register("tensor.rsqrt", Arc::new(aten_generated::Rsqrt));
    registry.register("tensor.reciprocal", Arc::new(aten_generated::Reciprocal));
    registry.register("tensor.square", Arc::new(aten_generated::Square));
    registry.register("tensor.sin", Arc::new(aten_generated::Sin));
    registry.register("tensor.cos", Arc::new(aten_generated::Cos));
    registry.register("tensor.tan", Arc::new(aten_generated::Tan));
    registry.register("tensor.asin", Arc::new(aten_generated::Asin));
    registry.register("tensor.acos", Arc::new(aten_generated::Acos));
    registry.register("tensor.atan", Arc::new(aten_generated::Atan));
    registry.register("tensor.sinh", Arc::new(aten_generated::Sinh));
    registry.register("tensor.cosh", Arc::new(aten_generated::Cosh));
    registry.register("tensor.tanh", Arc::new(aten_generated::Tanh));
    registry.register("tensor.asinh", Arc::new(aten_generated::Asinh));
    registry.register("tensor.acosh", Arc::new(aten_generated::Acosh));
    registry.register("tensor.atanh", Arc::new(aten_generated::Atanh));
    registry.register("tensor.floor", Arc::new(aten_generated::Floor));
    registry.register("tensor.ceil", Arc::new(aten_generated::Ceil));
    registry.register("tensor.round", Arc::new(aten_generated::Round));
    registry.register("tensor.trunc", Arc::new(aten_generated::Trunc));
    registry.register("tensor.frac", Arc::new(aten_generated::Frac));
    registry.register("tensor.sign", Arc::new(aten_generated::Sign));
    registry.register("tensor.sigmoid", Arc::new(aten_generated::Sigmoid));
    registry.register("tensor.silu", Arc::new(aten_generated::Silu));
    registry.register("tensor.mish", Arc::new(aten_generated::Mish));
    registry.register("tensor.gelu", Arc::new(aten_generated::Gelu));
    registry.register("tensor.relu", Arc::new(aten_generated::Relu));
    registry.register("tensor.selu", Arc::new(aten_generated::Selu));
    registry.register("tensor.elu", Arc::new(aten_generated::Elu));
    registry.register("tensor.celu", Arc::new(aten_generated::Celu));
    registry.register("tensor.leaky_relu", Arc::new(aten_generated::LeakyRelu));
    registry.register("tensor.prelu", Arc::new(aten_generated::Prelu));
    registry.register("tensor.softplus", Arc::new(aten_generated::Softplus));
    registry.register("tensor.softshrink", Arc::new(aten_generated::Softshrink));
    registry.register("tensor.hardshrink", Arc::new(aten_generated::Hardshrink));
    registry.register("tensor.hardtanh", Arc::new(aten_generated::Hardtanh));
    registry.register("tensor.threshold", Arc::new(aten_generated::Threshold));
    registry.register("tensor.erf", Arc::new(aten_generated::Erf));
    registry.register("tensor.erfc", Arc::new(aten_generated::Erfc));
    registry.register("tensor.sinc", Arc::new(aten_generated::Sinc));
    registry.register("tensor.deg2rad", Arc::new(aten_generated::Deg2rad));
    registry.register("tensor.rad2deg", Arc::new(aten_generated::Rad2deg));
    registry.register("tensor.logit", Arc::new(aten_generated::Logit));
    registry.register("tensor.nan_to_num", Arc::new(aten_generated::NanToNum));
    registry.register("tensor.heaviside", Arc::new(aten_generated::Heaviside));
    registry.register("tensor.isfinite", Arc::new(aten_generated::Isfinite));
    registry.register("tensor.isinf", Arc::new(aten_generated::Isinf));
    registry.register("tensor.isnan", Arc::new(aten_generated::Isnan));
    registry.register("tensor.isposinf", Arc::new(aten_generated::Isposinf));
    registry.register("tensor.isneginf", Arc::new(aten_generated::Isneginf));
    registry.register("tensor.signbit", Arc::new(aten_generated::Signbit));
    registry.register("tensor.logical_not", Arc::new(aten_generated::LogicalNot));
    registry.register("tensor.atan2", Arc::new(aten_generated::Atan2));
    registry.register("tensor.hypot", Arc::new(aten_generated::Hypot));
    registry.register("tensor.logaddexp", Arc::new(aten_generated::Logaddexp));
    registry.register("tensor.logaddexp2", Arc::new(aten_generated::Logaddexp2));
    registry.register("tensor.maximum", Arc::new(aten_generated::Maximum));
    registry.register("tensor.minimum", Arc::new(aten_generated::Minimum));
    registry.register("tensor.nextafter", Arc::new(aten_generated::Nextafter));
    registry.register("tensor.fmod", Arc::new(aten_generated::Fmod));
    registry.register("tensor.floor_divide", Arc::new(aten_generated::FloorDivide));
    registry.register("tensor.copysign", Arc::new(aten_generated::Copysign));
    registry.register("tensor.remainder", Arc::new(aten_generated::Remainder));
    registry.register("tensor.pow", Arc::new(aten_generated::Pow));
    registry.register("tensor.float_power", Arc::new(aten_generated::FloatPower));
    registry.register("tensor.eq", Arc::new(aten_generated::Eq));
    registry.register("tensor.ne", Arc::new(aten_generated::Ne));
    registry.register("tensor.lt", Arc::new(aten_generated::Lt));
    registry.register("tensor.le", Arc::new(aten_generated::Le));
    registry.register("tensor.gt", Arc::new(aten_generated::Gt));
    registry.register("tensor.ge", Arc::new(aten_generated::Ge));
    registry.register("tensor.where", Arc::new(aten_generated::Where));
    registry.register("tensor.clamp", Arc::new(aten_generated::Clamp));
    registry.register("tensor.clamp_min", Arc::new(aten_generated::ClampMin));
    registry.register("tensor.clamp_max", Arc::new(aten_generated::ClampMax));
    registry.register("tensor.lerp", Arc::new(aten_generated::Lerp));
    registry.register("tensor.xlogy", Arc::new(aten_generated::Xlogy));
    registry.register("tensor.logical_and", Arc::new(aten_generated::LogicalAnd));
    registry.register("tensor.logical_or", Arc::new(aten_generated::LogicalOr));
    registry.register("tensor.logical_xor", Arc::new(aten_generated::LogicalXor));
    registry.register("tensor.sum", Arc::new(aten_generated::Sum));
    registry.register("tensor.prod", Arc::new(aten_generated::Prod));
    registry.register("tensor.mean", Arc::new(aten_generated::Mean));
    registry.register("tensor.var", Arc::new(aten_generated::Var));
    registry.register("tensor.std", Arc::new(aten_generated::Std));
    registry.register("tensor.amax", Arc::new(aten_generated::Amax));
    registry.register("tensor.amin", Arc::new(aten_generated::Amin));
    registry.register("tensor.argmax", Arc::new(aten_generated::Argmax));
    registry.register("tensor.argmin", Arc::new(aten_generated::Argmin));
    registry.register("tensor.logsumexp", Arc::new(aten_generated::Logsumexp));
    registry.register("tensor.all", Arc::new(aten_generated::All));
    registry.register("tensor.any", Arc::new(aten_generated::Any));
    registry.register("tensor.cumsum", Arc::new(aten_generated::Cumsum));
    registry.register("tensor.cumprod", Arc::new(aten_generated::Cumprod));
    registry.register("tensor.softmax", Arc::new(aten_generated::Softmax));
    registry.register("tensor.log_softmax", Arc::new(aten_generated::LogSoftmax));
    registry.register("tensor.layer_norm", Arc::new(aten_generated::LayerNorm));
    registry.register("tensor.rms_norm", Arc::new(aten_generated::RmsNorm));
    registry.register("tensor.glu", Arc::new(aten_generated::Glu));
    registry.register("tensor.sort", Arc::new(aten_generated::Sort));
    registry.register("tensor.argsort", Arc::new(aten_generated::Argsort));
    registry.register("tensor.topk", Arc::new(aten_generated::Topk));
    registry.register("tensor.nonzero", Arc::new(aten_generated::Nonzero));
    registry.register("tensor.argwhere", Arc::new(aten_generated::Argwhere));
    registry.register("tensor.squeeze", Arc::new(aten_generated::Squeeze));
    registry.register("tensor.index_select", Arc::new(aten_generated::IndexSelect));
    registry.register("tensor.gather", Arc::new(aten_generated::Gather));
    registry.register("tensor.index_add", Arc::new(aten_generated::IndexAdd));
    registry.register("tensor.index_fill", Arc::new(aten_generated::IndexFill));
    registry.register("tensor.take", Arc::new(aten_generated::Take));
    registry.register("tensor.flatten", Arc::new(aten_generated::Flatten));
    registry.register("tensor.expand", Arc::new(aten_generated::Expand));
    registry.register("tensor.expand_as", Arc::new(aten_generated::ExpandAs));
    registry.register("tensor.narrow", Arc::new(aten_generated::Narrow));
    registry.register("tensor.slice", Arc::new(aten_generated::Slice));
    registry.register("tensor.chunk", Arc::new(aten_generated::Chunk));
    registry.register("tensor.split", Arc::new(aten_generated::Split));
    registry.register("tensor.stack", Arc::new(aten_generated::Stack));
    registry.register("tensor.unbind", Arc::new(aten_generated::Unbind));
    registry.register("tensor.repeat", Arc::new(aten_generated::Repeat));
    registry.register("tensor.flip", Arc::new(aten_generated::Flip));
    registry.register("tensor.triu", Arc::new(aten_generated::Triu));
    registry.register("tensor.tril", Arc::new(aten_generated::Tril));
    registry.register("tensor.diag", Arc::new(aten_generated::Diag));
    registry.register("tensor.diagonal", Arc::new(aten_generated::Diagonal));
    registry.register("tensor.trace", Arc::new(aten_generated::Trace));
    registry.register("tensor.t", Arc::new(aten_generated::T));
    registry.register("tensor.mT", Arc::new(aten_generated::MT));
    registry.register("tensor.swapaxes", Arc::new(aten_generated::Swapaxes));
    registry.register("tensor.moveaxis", Arc::new(aten_generated::Moveaxis));
    registry.register("tensor.rot90", Arc::new(aten_generated::Rot90));
    registry.register("tensor.contiguous", Arc::new(aten_generated::Contiguous));
    registry.register("tensor.clone", Arc::new(aten_generated::Clone));
    registry.register("tensor.detach", Arc::new(aten_generated::Detach));
    registry.register("tensor.view", Arc::new(aten_generated::View));
    registry.register("tensor.pixel_shuffle", Arc::new(aten_generated::PixelShuffle));
    registry.register("tensor.pixel_unshuffle", Arc::new(aten_generated::PixelUnshuffle));
    registry.register("tensor.constant_pad_nd", Arc::new(aten_generated::ConstantPadNd));
    registry.register("tensor.mm", Arc::new(aten_generated::Mm));
    registry.register("tensor.bmm", Arc::new(aten_generated::Bmm));
    registry.register("tensor.addmm", Arc::new(aten_generated::Addmm));
    registry.register("tensor.addbmm", Arc::new(aten_generated::Addbmm));
    registry.register("tensor.baddbmm", Arc::new(aten_generated::Baddbmm));
    registry.register("tensor.mv", Arc::new(aten_generated::Mv));
    registry.register("tensor.dot", Arc::new(aten_generated::Dot));
    registry.register("tensor.vdot", Arc::new(aten_generated::Vdot));
    registry.register("tensor.ger", Arc::new(aten_generated::Ger));
    registry.register("tensor.outer", Arc::new(aten_generated::Outer));
    registry.register("tensor.addr", Arc::new(aten_generated::Addr));
    registry.register("tensor.addmv", Arc::new(aten_generated::Addmv));
    registry.register("tensor.full", Arc::new(aten_generated::Full));
    registry.register("tensor.full_like", Arc::new(aten_generated::FullLike));
    registry.register("tensor.zeros_like", Arc::new(aten_generated::ZerosLike));
    registry.register("tensor.ones_like", Arc::new(aten_generated::OnesLike));
    registry.register("tensor.arange", Arc::new(aten_generated::Arange));
    registry.register("tensor.linspace", Arc::new(aten_generated::Linspace));
    registry.register("tensor.logspace", Arc::new(aten_generated::Logspace));
    registry.register("tensor.eye", Arc::new(aten_generated::Eye));
    registry.register("tensor.size", Arc::new(aten_generated::Size));
    registry.register("tensor.stride", Arc::new(aten_generated::Stride));
    registry.register("tensor.numel", Arc::new(aten_generated::Numel));
    registry.register("tensor.dim", Arc::new(aten_generated::Dim));
    registry.register("tensor.item", Arc::new(aten_generated::Item));
    registry.register("tensor.is_contiguous", Arc::new(aten_generated::IsContiguous));
    registry.register("tensor.avg_pool2d", Arc::new(aten_generated::AvgPool2d));
    registry.register("tensor.einsum", Arc::new(aten_generated::Einsum));
    registry.register("tensor.tensordot", Arc::new(aten_generated::Tensordot));
    registry.register("tensor.kron", Arc::new(aten_generated::Kron));
    registry.register("tensor.cross", Arc::new(aten_generated::Cross));
    registry.register("tensor.norm", Arc::new(aten_generated::Norm));
    registry.register("tensor.renorm", Arc::new(aten_generated::Renorm));
    registry.register("tensor.normalize", Arc::new(aten_generated::Normalize));
    registry.register("tensor.median", Arc::new(aten_generated::Median));
    registry.register("tensor.mode", Arc::new(aten_generated::Mode));
    registry.register("tensor.kthvalue", Arc::new(aten_generated::Kthvalue));
    registry.register("tensor.quantile", Arc::new(aten_generated::Quantile));
    registry.register("tensor.searchsorted", Arc::new(aten_generated::Searchsorted));
    registry.register("tensor.bincount", Arc::new(aten_generated::Bincount));
    registry.register("tensor.meshgrid", Arc::new(aten_generated::Meshgrid));
    registry.register("tensor.corrcoef", Arc::new(aten_generated::Corrcoef));
    registry.register("tensor.empty", Arc::new(aten_generated::Empty));
    registry.register("tensor.rand", Arc::new(aten_generated::Rand));
    registry.register("tensor.normal", Arc::new(aten_generated::Normal));
    registry.register("tensor.to", Arc::new(aten_generated::To));
    registry.register("tensor.unique", Arc::new(aten_generated::Unique));
    registry.register("tensor.nonzero_dim", Arc::new(aten_generated::NonzeroDim));
// ATEN-PORT-GEN:END registry
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

    /// ATen-port smoke: a representative slice of the `aten_generated`
    /// surface — unary, dim-reduction, tuple-returning sort, linalg, a
    /// constructor, and a declared seam failing loudly.
    #[test]
    fn aten_generated_ports_smoke() {
        use super::testutil::call_fn;
        use serde_json::json;

        // Unary activation.
        let out = call_fn(
            &aten_generated::Relu,
            &[tensor(&[3], &[-1.0, 0.0, 2.0])],
        )
        .unwrap();
        assert_eq!(out, tensor(&[3], &[0.0, 0.0, 2.0]));

        // `sum` over dim 0 (bare int accepted for `int[1]? dim`).
        let out = call_fn(
            &aten_generated::Sum,
            &[
                tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                json!(0),
                json!(false),
            ],
        )
        .unwrap();
        assert_eq!(out, tensor(&[2], &[4.0, 6.0]));

        // `sort` → `{values, indices}` record of wire tensors.
        let out = call_fn(
            &aten_generated::Sort,
            &[tensor(&[3], &[3.0, 1.0, 2.0])],
        )
        .unwrap();
        assert_eq!(out["values"], tensor(&[3], &[1.0, 2.0, 3.0]));
        assert_eq!(out["indices"], tensor(&[3], &[1.0, 2.0, 0.0]));

        // `mm` rank-2.
        let out = call_fn(
            &aten_generated::Mm,
            &[
                tensor(&[1, 2], &[3.0, 4.0]),
                tensor(&[2, 1], &[5.0, 6.0]),
            ],
        )
        .unwrap();
        assert_eq!(out, tensor(&[1, 1], &[39.0]));

        // `nonzero` → (N, rank) index tensor.
        let out = call_fn(
            &aten_generated::Nonzero,
            &[tensor(&[3], &[0.0, 2.0, 0.0])],
        )
        .unwrap();
        assert_eq!(out, tensor(&[1, 1], &[1.0]));

        // `arange` constructor (raw driver, 1–3 numeric args).
        let out = call_fn(&aten_generated::Arange, &[json!(0), json!(5), json!(2)]).unwrap();
        assert_eq!(out, tensor(&[3], &[0.0, 2.0, 4.0]));

        // A declared seam fails loudly rather than faking behavior.
        assert!(call_fn(&aten_generated::Einsum, &[]).is_err());
    }
}
