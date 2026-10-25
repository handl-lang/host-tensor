# How-to guides

Task recipes. Each assumes sibling checkouts of this repo and of the platform
repos it patches.

## Add a tensor op

Declare it in `src/index.hnl` first, then implement it in `src/ops/` and add its backward pass — the declaration is the contract the runtime loads.

## Wire tensors into a host

Compose the plugin crate into the host's plugin list; `.hnl` code then calls `tensor.*` like any other host function.
