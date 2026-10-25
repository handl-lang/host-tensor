# Explanation

## Why this crate exists

Tensor code is the first real numerical workload on the platform, and it
belongs *outside* the language: a host composes the tensor component only if it
needs it, and the `.hnl` surface stays the same whether the backend is Rust f64
arrays or something faster later. Declaring the surface in `.hnl` keeps the
documentation and the implementation from drifting apart.

## Where it sits

This crate belongs to the `engine-*` family. Plugins, kits and hosts reach it
through the kit's operation slot, so the engine can be swapped without touching a
language definition.

See the [reference](/host-tensor/reference/) for the API and the
[how-to guides](/host-tensor/how-to/) for tasks.
