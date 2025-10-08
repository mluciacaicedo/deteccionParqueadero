# src/grpc_client/client.py
"""
Cliente gRPC: envía una imagen al servidor y recibe detecciones YOLO.
- Detecta automáticamente el nombre del Stub en parking_pb2_grpc.
- O permite fijarlo con --service.
"""

import argparse, grpc, cv2
from src.grpc_api import parking_pb2 as pb, parking_pb2_grpc as pb_grpc

def run(image_path, host="localhost", port=50051):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("No se pudo codificar JPEG")
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = pb_grpc.ParkingDetectorStub(channel)            # <– NOMBRE DEL SERVICIO
    req = pb.PredictRequest(image=buf.tobytes(), conf=0.25, imgsz=640)  # <– NOMBRES DEL MENSAJE
    resp = stub.Predict(req)                                # <– NOMBRE DEL MÉTODO
    print(f"det: {len(resp.boxes)}, libres={resp.free_count}, ocupadas={resp.occupied_count}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=50051)
    args = ap.parse_args()
    run(args.image, args.host, args.port)