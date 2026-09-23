#!/bin/bash
# ---------------------------------------------------------------------------
# Pasos 1 y 2: catalogo + descarga de cutouts.
#
# Va en una particion de CPU: no usa GPU y esta limitado por red, no por
# computo. Para 100k galaxias con 8 workers toma varias horas.
#
#   sbatch cluster/descargar_datos.slurm
#
# AJUSTAR antes de usar: --partition, --account y el modulo de Python, que
# dependen del cluster.
# ---------------------------------------------------------------------------
#SBATCH --job-name=gasmet-datos
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/datos_%j.out
#SBATCH --error=logs/datos_%j.err

set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Los datos pesados van a scratch, no al home (que suele tener cuota chica).
# AJUSTAR: la ruta de scratch depende del cluster.
export GASMET_DATA_DIR="${GASMET_DATA_DIR:-/scratch/$USER/gasmet-data}"
mkdir -p "$GASMET_DATA_DIR"
echo "Datos en: $GASMET_DATA_DIR"

module load python/3.11 || echo "sin modules, se usa el python del entorno"
source .venv/bin/activate

echo "Nodo: $(hostname) | inicio: $(date)"

python scripts/01_construir_catalogo.py
python scripts/02_descargar_imagenes.py

echo "Fin: $(date)"
