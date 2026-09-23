#!/bin/bash
# ---------------------------------------------------------------------------
# Pasos 3 y 4: entrenamiento y evaluacion.
#
#   sbatch cluster/entrenar.slurm                       # resnet34, por defecto
#   sbatch cluster/entrenar.slurm cnn_simple baseline   # arquitectura y nombre
#
# Configurado para NLHPC (Leftraru): particion v100, cuenta "users".
# En otro cluster hay que ajustar --partition, --account y --gres.
#
# Se usa la particion v100 y no mi210: esas son AMD Instinct y necesitan
# PyTorch compilado con ROCm en vez de CUDA.
# ---------------------------------------------------------------------------
#SBATCH --job-name=gasmet-train
#SBATCH --partition=v100
#SBATCH --account=users
# --gres=gpu:1 y no gpu:v100:1 - el filtro de submit de NLHPC parsea el tipo
# y extrae el "100" de "v100", rechazando el job por pedir 100 GPUs.
#SBATCH --gres=gpu:1
# Cuidado con la razon memoria/CPU: NLHPC reasigna a largemem los jobs que la
# tienen alta, y largemem NO TIENE GPU, asi que este job moriria ahi. 48G sobre
# 12 CPUs daba 4096 MB/CPU, que es el valor que disparo el filtro en el job de
# datos. 24G da 2048 MB/CPU y alcanza de sobra: las imagenes se leen por memmap
# y el modelo vive en la GPU.
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

ARQUITECTURA="${1:-resnet34}"
NOMBRE="${2:-$ARQUITECTURA}"

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Los datos pesados viven fuera del repo. El home tiene cuota de 200 GB, que
# alcanza de sobra para los ~6.5 GB del proyecto.
export GASMET_DATA_DIR="${GASMET_DATA_DIR:-$HOME/gasmet-data}"

# Conda en vez de "module load python": los modulos estan compilados por
# arquitectura (zen4 / skylake) y los nodos GPU son Skylake, asi que cargar el
# modulo equivocado revienta con "Illegal instruction". Los paquetes de conda
# son binarios x86_64 portables y no tienen ese problema.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gasmet

echo "Nodo: $(hostname) | inicio: $(date)"
echo "Datos en: $GASMET_DATA_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python scripts/03_entrenar.py --arquitectura "$ARQUITECTURA" --nombre "$NOMBRE"
python scripts/04_evaluar.py  --arquitectura "$ARQUITECTURA" --nombre "$NOMBRE"

echo "Fin: $(date)"
