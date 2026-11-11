#!/usr/bin/env bash
# Build pinned llama.cpp and its RSS governor / token replay preload libraries.
#   libllama_budget.so  keeps llama-server's RSS inside the RAM budget (madvise + fadvise eviction of mmap'd weights)
#   libllama_replay.so  forced-output replay: every system decodes the same frozen output tokens
#
#   baselines/llama_cpp/build.sh [OUT]        # default OUT = build/llama.cpp
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=${1:-$ROOT/build/llama.cpp}
COMMIT=67a17c17caa95742186f8b1ecadd1b5abd6d5ebb

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
[ -d "$OUT/src/.git" ] || git clone https://github.com/ggml-org/llama.cpp "$OUT/src"
git -C "$OUT/src" checkout --detach "$COMMIT"
CUDA_OPTIONS=(-DGGML_CUDA="${GGML_CUDA:-ON}")
if [ "${GGML_CUDA:-ON}" != OFF ]; then
    CUDA_OPTIONS+=(-DCMAKE_CUDA_ARCHITECTURES="${TIDEWAY_CUDA_ARCH:-native}")
fi
cmake -S "$OUT/src" -B "$OUT/cmake" "${CUDA_OPTIONS[@]}" -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$OUT/cmake" --target llama-server llama-fit-params -j "${JOBS:-4}"
mkdir -p "$OUT/bin"
cp "$OUT/cmake/bin/llama-server" "$OUT/cmake/bin/llama-fit-params" "$OUT/bin/"

gcc -shared -fPIC -O2 -Wall -Wextra "$HERE/llama_budget.c" -o "$OUT/libllama_budget.so" -ldl -pthread
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"$OUT/src/include" -I"$OUT/src/ggml/include" \
    -shared -fPIC "$HERE/llama_replay_probe.cpp" -ldl -o "$OUT/libllama_replay.so"
echo "llama.cpp $COMMIT and preload libraries in $OUT"
