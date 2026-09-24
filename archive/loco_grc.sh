
set -e
echo "=== EXTRACT $(date) ==="
"/home/users/ar29/.conda/envs/solar-siting-ml/bin/python" Solar-Siting/extract_pixels.py --exclude-iso3 GRC --out "Solar-Siting/artifacts/stats/px_loco_grc.npz"
echo "=== TRAIN $(date) ==="
"/home/users/ar29/.conda/envs/solar-siting-ml/bin/python" Solar-Siting/train_rf.py --pool "Solar-Siting/artifacts/stats/px_loco_grc.npz" --out "Solar-Siting/artifacts/models/rf30_loco_grc.joblib"
echo "=== DONE $(date) ==="
