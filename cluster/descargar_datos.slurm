#!/bin/bash
# ---------------------------------------------------------------------------
# Pasos 1 y 2: catalogo + descarga de cutouts.
#
#   sbatch cluster/descargar_datos.slurm
#
# OJO: estos dos pasos necesitan salida a internet, no GPU. En muchos clusters
# los nodos de computo no tienen red. Si este job falla con errores de conexion,
# corre los dos scripts directamente en el login node dentro de tmux:
#
#     tmux new -s datos
#     export GASMET_DATA_DIR=$HOME/gasmet-data
#     conda activate gasmet
#     python scripts/01_construir_catalogo.py
#     python scripts/02_descargar_imagenes.py
#     # Ctrl+B, luego D para salir sin matarlo
#
# Configurado para NLHPC (Leftraru): particion general, cuenta "users".
# ---------------------------------------------------------------------------
#SBATCH --job-name=gasmet-datos
#SBATCH --partition=general
#SBATCH --account=users
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/datos_%j.out
#SBATCH --error=logs/datos_%j.err

set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export GASMET_DATA_DIR="${GASMET_DATA_DIR:-$HOME/gasmet-data}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gasmet

echo "Nodo: $(hostname) | inicio: $(date)"
echo "Datos en: $GASMET_DATA_DIR"

# Falla temprano y con un mensaje claro si el nodo no tiene salida a internet.
if ! curl -sS -o /dev/null --max-time 20 https://data.sdss.org/sas/dr8/common/sdss-spectro/redux/; then
    echo "ERROR: este nodo no tiene salida a internet." >&2
    echo "Corre los pasos 1 y 2 en el login node con tmux (ver cabecera)." >&2
    exit 1
fi

python scripts/01_construir_catalogo.py
python scripts/02_descargar_imagenes.py

echo "Fin: $(date)"
