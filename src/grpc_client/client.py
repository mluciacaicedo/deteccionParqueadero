# src/grpc_client/client.py
"""
Cliente gRPC: envía una imagen al servidor y recibe detecciones YOLO.

- Usa el servicio ParkingDetector y el RPC Predict(ImageRequest) -> PredictResponse.
- Permite ajustar conf/imgsz y guardar la imagen anotada que envía el servidor.
"""

from __future__ import annotations

import argparse
import os
import cv2
import grpc
from src.grpc_api import parking_pb2 as pb, parking_pb2_grpc as pb_grpc


def run(image_path: str,
        host: str = "localhost",
        port: int = 50051,
        conf: float = 0.25,
        imgsz: int = 640,
        save_path: str | None = None) -> None:
    """Envía una imagen al servidor gRPC y muestra/guarda resultados."""

    # 1) Cargar imagen y codificar a JPEG
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen a JPEG")
    img_bytes = buf.tobytes()

    # 2) Conectar y crear stub
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = pb_grpc.ParkingDetectorStub(channel)

    # 3) Construir request según tu .proto
    req = pb.ImageRequest(image=img_bytes, conf=float(conf), imgsz=int(imgsz))

    # 4) Llamar RPC
    resp: pb.PredictResponse = stub.Predict(req)

    # 5) Mostrar resultados
    print(f"det: {len(resp.boxes)}, libres={resp.free_count}, ocupadas={resp.occupied_count}")
    for i, b in enumerate(resp.boxes, 1):
        print(f"#{i} {b.name} conf={b.conf:.2f} [{b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f}]")

    # 6) Guardar imagen anotada si viene en la respuesta
    if save_path and resp.annotated:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(resp.annotated)
        print(f"💾 Guardado: {save_path}")


def main():
    ap = argparse.ArgumentParser(description="Cliente gRPC para inferencia YOLO.")
    ap.add_argument("--image", required=True, help="Ruta a la imagen de entrada (JPG/PNG).")
    ap.add_argument("--host", default="localhost", help="Host del servidor gRPC.")
    ap.add_argument("--port", type=int, default=50051, help="Puerto del servidor gRPC.")
    ap.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza.")
    ap.add_argument("--imgsz", type=int, default=640, help="Tamaño de entrada del modelo.")
    ap.add_argument("--save", type=str, default=None, help="Ruta para guardar la imagen anotada (opcional).")
    args = ap.parse_args()

    run(args.image, args.host, args.port, args.conf, args.imgsz, args.save)


if __name__ == "__main__":
    main()
