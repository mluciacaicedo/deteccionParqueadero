"""
gRPC Inference Server para YOLOv8.
- Carga el modelo una vez.
- Expone el método Predict(ImageRequest) -> PredictResponse.
- Devuelve boxes, conteo de libres/ocupadas y la imagen anotada (JPG bytes).
"""

from concurrent import futures
import os
import io
import cv2
import numpy as np
import grpc
from ultralytics import YOLO

from src.grpc_api import parking_pb2, parking_pb2_grpc


# ---------- utilidades para mapear clases ----------
def _normalize_class_map(names):
    return {i: str(n).lower() for i, n in names.items()}

def _find_class_id(names_lower, keywords):
    for i, n in names_lower.items():
        if any(k in n for k in keywords):
            return i
    return None

def resolve_empty_occupied_ids(names):
    lower = _normalize_class_map(names)
    idx_empty = _find_class_id(lower, ["empty", "free", "libre", "vacant"])
    idx_occ   = _find_class_id(lower, ["occupied", "ocupado", "busy", "occu"])

    if idx_empty is None and idx_occ is None:
        ids_sorted = sorted(list(names.keys()))
        if len(ids_sorted) >= 2:
            return ids_sorted[0], ids_sorted[1]
        elif len(ids_sorted) == 1:
            return ids_sorted[0], ids_sorted[0]
        return 0, 1

    if idx_empty is None:
        ids_sorted = [i for i in sorted(names) if i != idx_occ]
        idx_empty = ids_sorted[0] if ids_sorted else idx_occ
    if idx_occ is None:
        ids_sorted = [i for i in sorted(names) if i != idx_empty]
        idx_occ = ids_sorted[0] if ids_sorted else idx_empty
    return idx_empty, idx_occ


class ParkingDetectorServicer(parking_pb2_grpc.ParkingDetectorServicer):
    def __init__(self):
        # Ruta al modelo (puedes cambiarla con la variable de entorno MODEL_PATH)
        model_path = os.getenv("MODEL_PATH", "models/Modelo_yolov8_pklot2.pt")
        if not os.path.exists(model_path):
            # Fallback a un modelo base si no está tu entrenado
            model_path = os.getenv("MODEL_FALLBACK", "models/yolov8n.pt")

        self.model = YOLO(model_path)
        self.names = self.model.names
        self.idx_empty, self.idx_occ = resolve_empty_occupied_ids(self.names)

    def Predict(self, request, context):
        # Leer bytes -> imagen OpenCV BGR
        if not request.image:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty image payload")
            return parking_pb2.PredictResponse()

        img_arr = np.frombuffer(request.image, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid image bytes")
            return parking_pb2.PredictResponse()

        conf = request.conf if request.conf > 0 else 0.25
        imgsz = request.imgsz if request.imgsz > 0 else 640

        # Inferencia
        result = self.model.predict(
            source=img,
            conf=conf,
            imgsz=imgsz,
            device="cpu",    # cambia a "cuda:0" si tienes GPU disponible
            verbose=False
        )[0]

        boxes_msg = []
        cls_ids = result.boxes.cls.int().tolist() if result.boxes is not None else []
        confs   = result.boxes.conf.tolist()      if result.boxes is not None else []
        xyxy    = result.boxes.xyxy.tolist()      if result.boxes is not None else []

        for (x1, y1, x2, y2), cid, cf in zip(xyxy, cls_ids, confs):
            name = str(self.names.get(int(cid), str(cid)))
            boxes_msg.append(parking_pb2.Box(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                cls=int(cid), conf=float(cf), name=name
            ))

        libres   = sum(1 for cid in cls_ids if cid == self.idx_empty)
        ocupadas = sum(1 for cid in cls_ids if cid == self.idx_occ)

        annotated = result.plot()  # BGR
        ok, enc = cv2.imencode(".jpg", annotated)
        annotated_bytes = enc.tobytes() if ok else b""

        return parking_pb2.PredictResponse(
            boxes=boxes_msg,
            free_count=int(libres),
            occupied_count=int(ocupadas),
            annotated=annotated_bytes
        )


def serve(port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    parking_pb2_grpc.add_ParkingDetectorServicer_to_server(ParkingDetectorServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    print(f"gRPC YOLO Inference Server listo en 0.0.0.0:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
