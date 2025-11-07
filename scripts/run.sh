#!/usr/bin/env bash
# Ejecuta uno o varios scripts del proyecto.
# Uso:
#   ./scripts/run.sh            # Ejecuta todos
#   ./scripts/run.sh 01         # Ejecuta 01-wine
#   ./scripts/run.sh 03-grid    # Ejecuta 03-house-prices-grid
#   ./scripts/run.sh 03,04,06   # Ejecuta varios (coma separada)
# Variables de entorno:
#   EPOCHS_OVERRIDE   Número de épocas (default 1)
#   PRED_SAMPLE_LIMIT Número de predicciones de ejemplo (default 5)
# Salidas:
#   quickruns/logs/<alias>.txt
#   quickruns/logs/summary.tsv acumulado
#   quickruns/logs/_failures.txt lista fallos

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/quickruns/logs"
mkdir -p "${LOG_DIR}"
SUMMARY_FILE="${LOG_DIR}/summary.tsv"
FAIL_FILE="${LOG_DIR}/_failures.txt"
: > "${SUMMARY_FILE}" || true
: > "${FAIL_FILE}" || true

EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-1}"
PRED_SAMPLE_LIMIT="${PRED_SAMPLE_LIMIT:-5}"

# Mapa alias -> ruta
declare -A SCRIPTS=(
  [01]="${PROJECT_ROOT}/01-wine/01-wine.py"
  [02]="${PROJECT_ROOT}/02-mushrooms/02-mushrooms.py"
  [03]="${PROJECT_ROOT}/03-house-prices/03-kingcounty-house-prices.py"
  [03-grid]="${PROJECT_ROOT}/03-house-prices/03-kingcounty-house-prices-grid.py"
  [03-scikit]="${PROJECT_ROOT}/03-house-prices/03-kingcounty-house-prices-scikit.py"
  [04]="${PROJECT_ROOT}/04-videogames/04-videogames.py"
  [05]="${PROJECT_ROOT}/05-sq-tri/05-sq-tri-keras.py"
  [06]="${PROJECT_ROOT}/06-bolts-nuts/06-bolts-nuts.py"
  [07]="${PROJECT_ROOT}/07-new-houses-london/07-new-houses-london.py"
  [08]="${PROJECT_ROOT}/08-temperature/08-temperature.py"
)

list_all() {
  echo "Scripts disponibles:" >&2
  for k in "${!SCRIPTS[@]}"; do echo "  $k -> ${SCRIPTS[$k]}" >&2; done | sort -n
}

run_one() {
  local alias="$1"; shift
  local path="${SCRIPTS[$alias]}"
  local base
  base="$(basename "${path}" .py)"  # 01-wine.py -> 01-wine
  local log="${LOG_DIR}/${base}.txt"
  echo "== Ejecutando ${alias} (${path}) -> log: ${base}.txt =="
  if EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE}" PRED_SAMPLE_LIMIT="${PRED_SAMPLE_LIMIT}" python "${path}" > "${log}" 2>&1; then
    echo -e "${alias}\tOK" >> "${SUMMARY_FILE}"
  else
    echo -e "${alias}\tFAIL" >> "${SUMMARY_FILE}"
    echo "${alias}" >> "${FAIL_FILE}"
  fi
}

resolve_targets() {
  local input="$1"
  if [[ -z "${input}" ]]; then
    for k in "${!SCRIPTS[@]}"; do
      echo "$k"
    done
    return
  fi
  # separar por comas
  IFS=',' read -r -a parts <<< "$input"
  for p in "${parts[@]}"; do
    if [[ -n "${SCRIPTS[$p]:-}" ]]; then
      echo "$p"
    else
      echo "[warn] Alias desconocido: $p" >&2
    fi
  done
}

main() {
  local arg="${1:-}"
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    echo "Uso: $0 [alias|alias1,alias2,...]" >&2
    list_all
    exit 0
  fi
  local targets
  targets=$(resolve_targets "${arg}") || true
  if [[ -z "${targets}" ]]; then
    echo "No hay objetivos válidos." >&2
    exit 1
  fi
  echo "EPOCHS_OVERRIDE=${EPOCHS_OVERRIDE} PRED_SAMPLE_LIMIT=${PRED_SAMPLE_LIMIT}" >&2
  for t in ${targets}; do
    [[ "$t" == \[* ]] && continue  # saltar mensajes [skip]/[warn]
    run_one "$t"
  done
  echo -e "\nResumen:" >&2
  column -t -s $'\t' "${SUMMARY_FILE}" || cat "${SUMMARY_FILE}"
  if [[ -s "${FAIL_FILE}" ]]; then
    echo "\nFallos detectados (ver ${FAIL_FILE})." >&2
    exit 1
  fi
}

main "$@"
