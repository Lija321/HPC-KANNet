#!/usr/bin/env bash
# Ponovljiva regeneracija: data/ + outputs/ (seed=0), zatim benchmark-i.
# Iz python_implementation: relativne putanje ../data i ../outputs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/python_implementation"
RS="$ROOT/rust_implementation"
SEED=0
RUNS="${RUNS:-30}"

cd "$ROOT"
rm -rf data/* outputs/*
rm -rf par_visualization_* seq_visualization_* 2>/dev/null || true

cd "$PY"
# Veličine: benchmark_sizes_kernels (default) + strong/weak skaliranje, ali limit do 512
# Weak (base=256, workers=1,2,4) daje velicine: 256, 360, 512.
SIZES=(16 24 32 40 48 56 64 80 88 96 112 128)
for n in "${SIZES[@]}"; do
  python3 generate_input.py --out-dir "$ROOT/data" --H "$n" --W "$n" --seed "$SEED"
done

for k in 2 3 5 7; do
  python3 generate_params.py --out-dir "$ROOT/data" --kernel "$k" --seed "$SEED"
done

echo "== benchmark_sizes_kernels (Python) =="
python3 benchmark_sizes_kernels.py --data-dir "$ROOT/data" --params-base "$ROOT/data" \
  --out-dir "$ROOT/outputs/sizes_kernels/python/workers{workers}"

echo "== benchmark_sizes_kernels (Rust, --release) =="
cd "$RS"
cargo run --release -q --bin benchmark_sizes_kernels -- \
  --data-dir "$ROOT/data" --params-base "$ROOT/data" \
  --out-dir "$ROOT/outputs/sizes_kernels/rust/workers{threads}"

PARAMS_DIR="$ROOT/data/paper_kan_params_in9_out8_G5_k3_kernel3"
PARAMS_JSON="$PARAMS_DIR/params.json"

run_scaling () {
  local strong="$1"
  local weakbase="$2"
  local tag="strong${strong}_weakbase${weakbase}"
  echo "== benchmark_scaling $tag (Python, runs=$RUNS) =="
  cd "$PY"
  python3 benchmark_scaling.py \
    --data-dir "$ROOT/data" \
    --params-dir "$PARAMS_DIR" \
    --kernel 3 --runs "$RUNS" \
    --workers "1,2,4" \
    --strong-input "input_matrix_${strong}.csv" \
    --weak-base "$weakbase" \
    --out-dir "$ROOT/outputs/scaling/kernel3/strong{strong}/weakbase{weakbase}"

  echo "== benchmark_scaling $tag (Rust, runs=$RUNS) =="
  cd "$RS"
  cargo run --release -q --bin benchmark_scaling -- \
    --data-dir "$ROOT/data" \
    --params "$PARAMS_JSON" \
    --kernel 3 --runs "$RUNS" \
    --workers "1,2,4" \
    --strong-size "$strong" \
    --weak-base "$weakbase" \
    --out-dir "$ROOT/outputs/scaling/kernel3/strong{strong_size}/weakbase{weak_base}"
}

run_scaling 128 64
run_scaling 512 256

# Plotting podrazumevano čita ove fajlove — kopija poslednjeg (najveći ulaz) ili glavnog seta.
# Za NTP grafike tipično je dovoljan jedan set; koristimo strong128/weakbase64 kao podrazumevani.
cd "$ROOT"
cp -f "outputs/scaling/kernel3/strong128/weakbase64/python_scaling.csv" "outputs/scaling_results.csv"
cp -f "outputs/scaling/kernel3/strong128/weakbase64/rust_scaling.csv" "outputs/scaling_results_rust.csv"

echo "== compare_rust_python (sanity) =="
cd "$PY"
python3 compare_rust_python.py \
  --input "$ROOT/data/input_matrix_128.csv" \
  --params-dir "$PARAMS_DIR" \
  --workdir "$ROOT/outputs/tmp_compare"

echo "== kannet_plotting =="
cd "$ROOT/plotting"
cargo run --release -q -- \
  --rust-input "$ROOT/outputs/scaling_results_rust.csv" \
  --python-input "$ROOT/outputs/scaling_results.csv" \
  --out-dir "$ROOT/outputs/plots_rust"

echo "Gotovo. Podaci: $ROOT/data, rezultati: $ROOT/outputs"
# Napomena: log ne ide u outputs/ jer se na početku briše outputs/*.
