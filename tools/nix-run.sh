#!/bin/bash
# Runs binaries from the locally cached OpenLane2 container image layers
# without a container runtime (podman/docker unusable in this sandbox).
# Usage: tools/nix-run.sh <abs-path-to-nix-binary> [args...]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIFF=/home/indhirajith/.local/share/containers/storage/overlay/25193cff65d059b5dc1e2950ea4fbfdda43b1fc5e24c2e35bdefea77fc832326/diff
LOADER=$DIFF/nix/store/k7zgvzp2r31zkg9xqgjim7mbknryv6bs-glibc-2.39-52/lib/ld-linux-x86-64.so.2
LP=$(cat "$ROOT/tools/nix-libpath.txt")
# yosys plugins/abc resolve via PATH; expose the tool bins too.
export PATH="$DIFF/nix/store/6l3aj1gi4lja8z5s4lan2k2cfz9mqkrz-yosys-abc/bin:$DIFF/nix/store/9r0bh7sp051dpm8km8bqlb028anpd3v3-yosys/bin:$PATH"
# Embedded interpreters expect a real /nix/store; point them at the layer copy.
export PYTHONHOME="$DIFF/nix/store/7hnr99nxrd2aw6lghybqdmkckq60j6l9-python3-3.11.9"
export PYTHONPATH="$PYTHONHOME/lib/python3.11:$PYTHONHOME/lib/python3.11/lib-dynload"
exec "$LOADER" --library-path "$LP" "$@"
