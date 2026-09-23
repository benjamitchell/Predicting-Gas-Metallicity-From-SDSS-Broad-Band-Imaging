#!/bin/bash
# ---------------------------------------------------------------------------
# Pasos 3 y 4: entrenamiento y evaluacion.
#
#   sbatch cluster/entrenar.slurm                       # resnet34, por defecto
#   sbatch cluster/entrenar.slurm cnn_simple baseline   # arquitectura y nombre
#
# AJUSTAR antes de usar: --partition, --account, --gres y el modulo de Python.
# ---------------------------------------------------------------------------
#SBATCH --job-name=gasmet-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

ARQUITECTURA="${1:-resnet34}"
NOMBRE="${2:-$ARQUITECTURA}"

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Los datos pesados van a scratch, no al home (que suele tener cuota chica).
# AJUSTAR: la ruta de scratch depende del cluster.
export GASMET_DATA_DIR="${GASMET_DATA_DIR:-/scratch/$USER/gasmet-data}"
mkdir -p "$GASMET_DATA_DIR"
echo "Datos en: $GASMET_DATA_DIR"

module load python/3.11 || echo "sin modules, se usa el python del entorno"
module load cuda/12.1 || echo "sin modulo de cuda"
source .venv/bin/activate

echo "Nodo: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-?} | inicio: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/03_entrenar.py --arquitectura "$ARQUITECTURA" --nombre "$NOMBRE"
python scripts/04_evaluar.py  --arquitectura "$ARQUITECTURA" --nombre "$NOMBRE"

echo "Fin: $(date)"
