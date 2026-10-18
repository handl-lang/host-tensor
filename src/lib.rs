//! `host-tensor` — host-side tensor functions for handl runtimes.
//!
//! `src/index.hnl` is the single source of truth for the `tensor.*`
//! interface (18 `op` declarations); `ops.rs` holds the Rust atoms behind
//! them. A runtime (catopt is the first consumer) composes the component
//! by building a `HandlModule` with the op registry:
//!
//! ```ignore
//! let module = HandlModule::new("tensor", TENSOR_HNL, &[])
//!     .with_ops(TENSOR_OPS, Arc::new(tensor_op_registry()));
//! ```
//!
//! after which `.hnl` `tensor.*` calls dispatch through the module chain
//! (`call_host("tensor.matmul")` → `ModuleLookup` → the `OpFunction` → the
//! registered Rust impl).
//!
//! The tensor value is data on the `serde_json::Value` boundary —
//! `{"__tensor__": {"shape": [d, ...], "data": [x, ...], "dtype": "f64"}}`
//! — so no engine or language change is required to carry it.

/// The handl source declaring the `tensor` interface.
pub const TENSOR_HNL: &str = include_str!("index.hnl");

pub mod ops;

pub use ops::{TENSOR_OPS, tensor_op_registry};
