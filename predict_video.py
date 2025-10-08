"""
Inferencia de video con YOLOv8 (Ultralytics) para conteo de plazas de parqueo.

- Acepta archivo de video o webcam.
- Dibuja boxes y muestra/guarda el video anotado.
- Cuenta por frame clases "libres" y "ocupadas" detectándolas por nombre
  (funciona con nombres como: empty/free/libre/vacant y occupied/ocupado/busy).

"""

from __future__ import annotations

import argparse
import os
import sys
import platform
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
from ultralytics import YOLO


# ------------------------- utilidades de clases -------------------------
def _normalize_class_map(names: Dict[int, str]) -> Dict[int, str]:
    """Devuelve {id: nombre_minusculas}."""
    return {int(i): str(n).lower() for i, n in names.items()}


def _find_class_id(
    names_lower: Dict[int, str],
    keywords: Iterable[str],
) -> Optional[int]:
    """Retorna el id de clase cuyo nombre contenga alguna palabra clave, o None si no hay coincidencia."""
    for i, n in names_lower.items():
        if any(k in n for k in keywords):
            return i
    return None


def resolve_empty_occupied_ids(names: Dict[int, str]) -> Tuple[int, int]:
    """
    Intenta deducir los IDs para 'empty' y 'occupied' con varios alias.
    Si no puede, usa fallback (si hay >=2 clases: 0/1 ordenados; si hay 1: repite el mismo).
    """
    lower = _normalize_class_map(names)
    idx_empty = _find_class_id(lower, ["empty", "free", "libre", "vacant"])
    idx_occ = _find_class_id(lower, ["occupied", "ocupado", "busy", "occu"])

    if idx_empty is None and idx_occ is None:
        ids_sorted = sorted(int(i) for i in names.keys())
        if len(ids_sorted) >= 2:
            return ids_sorted[0], ids_sorted[1]
        if len(ids_sorted) == 1:
            return ids_sorted[0], ids_sorted[0]
        # Sin clases (no debería pasar en YOLO)
        return 0, 1

    if idx_empty is None:
        ids_sorted = [i for i in sorted(int(i) for i in names.keys()) if i != idx_occ]
        idx_empty = ids_sorted[0] if ids_sorted else idx_occ
    if idx_occ is None:
        ids_sorted = [i for i in sorted(int(i) for i in names.keys()) if i != idx_empty]
        idx_occ = ids_sorted[0] if ids_sorted else idx_empty

    return int(idx_empty), int(idx_occ)


def choose_fourcc() -> int:
    """Elige un FOURCC razonable según el SO."""
    system = platform.system().lower()
    if "windows" in system:
        return cv2.VideoWriter_fourcc(*"mp4v")
    if "darwin" in system or "mac" in system:
        return cv2.VideoWriter_fourcc(*"avc1")
    # Linux/otros
    return cv2.VideoWriter_fourcc(*"mp4v")


def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


# ------------------------- inferencia de video -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predicción de video con YOLOv8.")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Ruta de video o índice de webcam (ej. '0').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_PATH", "models/Modelo_yolov8_pklot2.pt"),
        help="Ruta al modelo .pt entrenado.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1).")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU para NMS.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen de entrada.")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo: 'cpu' o 'cuda:0'.")
    parser.add_argument("--show", action="store_true", help="Muestra ventana con previsualización (q para salir).")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Ruta de salida del video anotado (mp4). Si no se indica, se genera automáticamente.",
    )
    parser.add_argument("--nosave", action="store_true", help="No guarda el video anotado en disco.")
    parser.add_argument("--out-dir", type=str, default=".", help="Directorio de salida si no se especifica --save.")
    parser.add_argument("--save-fps", type=float, default=None, help="FPS forzado para el writer (opcional).")
    parser.add_argument("--max-frames", type=int, default=None, help="Procesa solo N frames (debug).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validar modelo
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        sys.exit(1)

    # Cargar modelo
    model = YOLO(str(model_path))
    names = model.names
    idx_empty, idx_occ = resolve_empty_occupied_ids(names)

    print("Modelo cargado:", model_path.name)
    print("Clases del modelo:", names)
    print(f"Mapeo clases → libres: {idx_empty}, ocupadas: {idx_occ}")

    # Preparar fuente
    is_webcam = False
    source_for_yolo = args.source
    cap_for_meta: cv2.VideoCapture

    if is_int(args.source):
        is_webcam = True
        cam_index = int(args.source)
        cap_for_meta = cv2.VideoCapture(cam_index)
        source_for_yolo = cam_index
    else:
        src_path = Path(args.source)
        if not src_path.exists():
            print(f"❌ Fuente de video no encontrada: {src_path}")
            sys.exit(1)
        cap_for_meta = cv2.VideoCapture(str(src_path))

    if not cap_for_meta.isOpened():
        print("❌ No se pudo abrir la fuente de video.")
        sys.exit(1)

    # Metadatos de video para el writer
    fps = args.save_fps or (cap_for_meta.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap_for_meta.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap_for_meta.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    cap_for_meta.release()

    # Preparar writer
    writer = None
    out_path: Optional[Path] = None
    if not args.nosave:
        if args.save:
            out_path = Path(args.save)
        else:
            base = "webcam" if is_webcam else Path(str(args.source)).stem
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{base}_pred.mp4"

        fourcc = choose_fourcc()
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            print("No se pudo crear el archivo de salida. Continuando sin guardado…")
            writer = None
        else:
            print(f"💾 Guardando salida en: {out_path}")

    # Inferencia en streaming
    print("▶️ Iniciando inferencia… (q para salir si --show)")
    frame_count = 0
    try:
        for result in model.predict(
            source=source_for_yolo,
            stream=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        ):
            # Imagen anotada por Ultralytics
            frame = result.plot()

            # Conteos por frame
            cls_ids = result.boxes.cls.int().tolist() if result.boxes is not None else []
            libres = sum(1 for cid in cls_ids if cid == idx_empty)
            ocupadas = sum(1 for cid in cls_ids if cid == idx_occ)

            # Overlay de contadores
            top = 30
            cv2.rectangle(frame, (0, 0), (260, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Libres:   {libres}", (10, top), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Ocupadas: {ocupadas}", (10, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            if args.show:
                cv2.imshow("YOLOv8 - Predicción", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer is not None:
                writer.write(frame)

            frame_count += 1
            if args.max_frames is not None and frame_count >= args.max_frames:
                print(f"⏹ Fin por límite de frames: {args.max_frames}")
                break
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print("✅ Listo.")


if __name__ == "__main__":
    main()
