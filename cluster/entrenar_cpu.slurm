#!/bin/bash
# ---------------------------------------------------------------------------
# Variante en CPU de entrenar.slurm, para cuando la particion de GPU esta
# copada. Util sobre todo para la cnn_simple, que tiene 0.1M parametros contra
# los 21.3M de la ResNet-34 y no necesita GPU para ser viable.
#
#   sbatch cluster/entrenar_cpu.slurm cnn_simple baseline
#
# La particion general tiene 48 nodos y rara vez esta llena, asi que entra de
# inmediato. A cambio es varias veces mas lento que una V100.
#
# NO usar esto para resnet34: 21M de parametros sobre 70k imagenes en CPU
# tomaria muchas horas. Para eso, esperar la particion v100.
# ---------------------------------------------------------------------------
#SBATCH --job-name=gasmet-cpu
#SBATCH --partition=general
#SBATCH --account=users
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/traincpu_%j.out
#SBATCH --error=logs/traincpu_%j.err

set -euo pipefail

ARQUITECTURA="${1:-cnn_simple}"
NOMBRE="${2:-$ARQUITECTURA}"

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export GASMET_DATA_DIR="${GASMET_DATA_DIR:-$HOME/gasmet-data}"

# Que torch use todos los cores reservados y ni uno mas: sobresuscribir hilos
# en un nodo compartido degrada el rendimiento propio y el de los vecinos.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gasmet

echo "Nodo: $(hostname) | CPUs: ${SLURM_CPUS_PER_TASK:-?} | inicio: $(date)"
echo "Datos en: $GASMET_DATA_DIR"
echo "Sin GPU: el entrenamiento sera notoriamente mas lento."

python scripts/03_entrenar.py --arquitectura "$ARQUITECTURA" --nombre "$NOMBRE"
python scripts/04_evaluar.py  --arquitectura "$ARQUITECTURA" --nombre "$NOMBRE"

echo "Fin: $(date)"
