"""
Crea un subconjunto balanceado del dataset PKLot (formato Roboflow YOLOv8):
- Acepta múltiples extensiones de imagen (.jpg, .jpeg, .png, .bmp, .webp).
- Detecta clases automáticamente desde data.yaml (o desde los .txt si no hay yaml).
- Permite elegir las particiones de origen (train / valid / test).
- Balancea por clase hasta un máximo (--per-class) si hay suficientes ejemplos.
- Hace split 80/20 (o el que indiques) sobre el conjunto unido de origen.
- Copia imágenes y etiquetas y genera un data.yaml con rutas relativas.

Uso:
python src/utils/create_subset_pklot.py \
  --root data/PKLot.v2-640.yolov8 \
  --out data/mini_PKLot.v2-640.yolov8 \
  --per-class 1000 \
  --splits train valid \
  --val-size 0.2 \
  --seed 42
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_image_for_label(images_dir: Path, stem: str) -> Optional[Path]:
    """Devuelve la ruta de imagen existente para un label dado (cualquier extensión soportada)."""
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_classes_from_yaml(root: Path) -> Optional[List[str]]:
    """Intenta leer las clases desde root/data.yaml (formato Roboflow)."""
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        return None
    names: List[str] = []
    try:
        # lectura mínima sin dependencias externas
        txt = yaml_path.read_text(encoding="utf-8")
        # Busca la línea names: ['a','b'] o names: ["a","b"]
        import re

        m = re.search(r"(?m)^\s*names\s*:\s*\[(.*?)\]\s*$", txt)
        if m:
            content = m.group(1)
            parts = re.findall(r"""(?x)
                '([^']+)'|   # 'clase'
                "([^"]+)"    # "clase"
            """, content)
            for p in parts:
                names.append(p[0] or p[1])
            return names if names else None
    except Exception:
        return None
    return None


def detect_classes_from_labels(labels_dirs: Sequence[Path]) -> List[int]:
    """Escanea .txt y devuelve ids de clase encontrados (ordenados)."""
    seen = set()
    for d in labels_dirs:
        if not d.exists():
            continue
        for lf in d.glob("*.txt"):
            try:
                for line in lf.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    cid = int(line.split()[0])
                    seen.add(cid)
            except Exception:
                continue
    return sorted(seen)


def collect_pairs(
    root: Path, splits: Sequence[str]
) -> Tuple[Dict[int, List[Tuple[Path, Path]]], List[int], List[str]]:
    """
    Recolecta parejas (imagen, label) agrupadas por class_id desde las particiones dadas.
    Retorna (pairs_by_class, class_ids, class_names)
    """
    # Detecta class_names desde yaml, si existen:
    class_names = read_classes_from_yaml(root) or []

    # Para detectar ids desde labels:
    labels_dirs = []
    for sp in splits:
        labels_dirs.append(root / sp / "labels")
    class_ids = detect_classes_from_labels(labels_dirs)

    if not class_ids:
        raise ValueError("No se encontraron etiquetas válidas en las particiones indicadas.")

    # Si no hay names en yaml, genera placeholders
    if not class_names:
        class_names = [f"class_{i}" for i in range(max(class_ids) + 1)]

    pairs_by_class: Dict[int, List[Tuple[Path, Path]]] = defaultdict(list)

    for sp in splits:
        images_dir = root / sp / "images"
        labels_dir = root / sp / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        for lf in labels_dir.glob("*.txt"):
            stem = lf.stem
            img_path = find_image_for_label(images_dir, stem)
            if img_path is None:
                continue

            # clasifica por primera clase que aparezca en el archivo (suficiente para balancear)
            try:
                lines = [ln.strip() for ln in lf.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if not lines:
                    continue
                first_cid = int(lines[0].split()[0])
            except Exception:
                continue

            pairs_by_class[first_cid].append((img_path, lf))

    # Limpia clases vacías (por si estaban en yaml pero no en labels)
    class_ids = [cid for cid in class_ids if cid in pairs_by_class and len(pairs_by_class[cid]) > 0]
    if not class_ids:
        raise ValueError("Se encontraron etiquetas, pero ninguna pareja imagen/label válida.")

    return pairs_by_class, class_ids, class_names


def write_yaml(out_base: Path, class_names: List[str]) -> None:
    """Escribe data.yaml con rutas relativas y nombres de clases proporcionados."""
    y = (
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    (out_base / "data.yaml").write_text(y, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Crea subset balanceado de PKLot (Roboflow YOLOv8).")
    ap.add_argument("--root", required=True, help="Carpeta raíz del dataset (con train/ valid/ test/).")
    ap.add_argument("--out", required=True, help="Carpeta salida para el mini-dataset.")
    ap.add_argument("--per-class", type=int, default=1000, help="Máximo de ejemplos por clase.")
    ap.add_argument("--splits", nargs="+", default=["train", "valid"], help="Particiones de origen.")
    ap.add_argument("--val-size", type=float, default=0.2, help="Proporción para validación (0–1).")
    ap.add_argument("--seed", type=int, default=42, help="Semilla para aleatoriedad.")
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root)
    out_base = Path(args.out)

    # Crear carpetas destino
    for split in ("train", "val"):
        (out_base / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_base / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Recolectar pares por clase
    print("Escaneando dataset...")
    pairs_by_class, class_ids, class_names = collect_pairs(root, args.splits)
    print(f"Clases detectadas (ids): {class_ids}")
    print(f"Nombres: {class_names}")

    # Construir subconjunto balanceado (hasta per-class por id)
    selected: List[Tuple[Path, Path]] = []
    for cid in class_ids:
        pool = pairs_by_class[cid]
        random.shuffle(pool)
        take = min(args.per_class, len(pool))
        selected.extend(pool[:take])

    if not selected:
        raise ValueError("No se pudo seleccionar ningún ejemplo. Revisa parámetros y datos.")

    random.shuffle(selected)

    # Split train/val
    n = len(selected)
    n_val = max(1, int(n * args.val_size))
    val_set = selected[:n_val]
    train_set = selected[n_val:]

    def copy_pairs(pairs: List[Tuple[Path, Path]], split: str) -> None:
        for img, lbl in pairs:
            shutil.copy2(img, out_base / "images" / split / img.name)
            shutil.copy2(lbl, out_base / "labels" / split / lbl.name)

    print(f"📦 Copiando: train={len(train_set)}, val={len(val_set)} (total={n})")
    copy_pairs(train_set, "train")
    copy_pairs(val_set, "val")

    # Escribir data.yaml
    # Ajusta class_names al tamaño máximo de los ids
    if len(class_names) <= max(class_ids):
        # asegura longitud suficiente
        max_id = max(class_ids)
        class_names = (class_names + [f"class_{i}" for i in range(max_id + 1 - len(class_names))])[: max_id + 1]
    write_yaml(out_base, class_names)

    print("✅ ¡Listo!")
    print(f"   Carpeta destino: {out_base.resolve()}")
    print(f"   data.yaml:       {(out_base / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
