import json
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import keras
from pydantic import BaseModel, Field, ValidationError, GetCoreSchemaHandler
from pydantic_core import core_schema
from typing import Dict, Optional, List, Any, Annotated
from datetime import datetime, timezone
from bson import ObjectId
import os
import uuid
import asyncio
import pandas as pd
from pyproj import Transformer
import tensorflow as tf
import numpy as np

try:
    from pymongo import MongoClient, ASCENDING
except Exception as exc:  # pragma: no cover - allows file to load without deps installed
    MongoClient = None  # type: ignore
    ASCENDING = 1  # type: ignore


# ----------------------
# Configuration
# ----------------------
MODEL_PATH = "MachineLearning/rd9_epoch100_h1n350_ffn150_final_model.keras"
PREDICTION_HORIZON = 60          # dt(t+1) in seconds (you can change per-request if wanted)
CPA_THRESHOLD = 500.0            # meters -> below this is considered risk/collision
MAX_TCPA = 3600.0                # seconds; ignore TCPA beyond this (safety limit)
UTM_EPSG = 32614                 # UTM zone 14N (EPSG:32614)

@keras.saving.register_keras_serializable()
def dist_euclidean(y_true, y_pred):
    """
    Calculates the Euclidean distance between two tensors.
    """
    return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()
except Exception as e:
    # Raise on import so you notice early; in production handle better
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

print("\nLoading normalization parameters...")
with open("MachineLearning/rd9_epoch100_h1n350_ffn150_norm_param_mean_std.json", 'r') as f:
    norm_param = json.load(f)

    sc_x_mean = np.array(norm_param['sc_x_mean'])
    sc_x_std = np.array(norm_param['sc_x_std'])
    print(f"Loaded mean values: {sc_x_mean}")
    print(f"Loaded std dev values: {sc_x_std}")


to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
from_utm = Transformer.from_crs(f"EPSG:{UTM_EPSG}", "EPSG:4326", always_xy=True)

def lonlat_to_utm(lon: float, lat: float):
    """Return (x, y) in meters in UTM zone 14N."""
    try:
        x, y = to_utm.transform(lon, lat)
        return float(x), float(y)
    except Exception as e:
        print(f"error: {e}")

def utm_to_lonlat(x: float, y: float):
    try:
        """Return (lon, lat) in degrees from UTM coords."""
        lon, lat = from_utm.transform(x, y)
        return float(lon), float(lat)
    except Exception as e:
        print(f"error {e}")

def build_model_input(dt: float, dlon: float, dlat: float, dt_future: float):
    """
    Build raw input vector for the Keras model.
    If your model expects normalization (mean/std etc.), apply it here before returning.
    """
    try:
        print(f"ma")
        arr = np.array([[dt, dlon, dlat, dt_future]], dtype=np.float32)
        normalized_input_data = (arr - sc_x_mean) / sc_x_std
        print(f"is thisok {normalized_input_data}")
        return normalized_input_data
    except Exception as e:
        print(f"error {e}")

def predict_displacement(arr: np.ndarray):
    """Run model.predict and return predicted [dlon_future, dlat_future] as floats (meters)."""
    pred = model.predict(arr, batch_size = 1)
    # model.predict returns shape (1, 2) ideally
    # pred = np.asarray(pred).reshape(-1)
    # TODO: de-normalize if model used normalization
    dlon_mean = sc_x_mean[1]
    dlon_std = sc_x_std[1]
    dlat_mean = sc_x_mean[2]
    dlat_std = sc_x_std[2]

    predictions_denormalized = pred.copy()
    predictions_denormalized[:,:,0] = (pred[:,:,0] * dlon_std) + dlon_mean
    predictions_denormalized[:,:,1] = (pred[:,:,1] * dlat_std) + dlat_mean
    return float(predictions_denormalized[0, 0, 0]), float(predictions_denormalized[0, 0, 1])
    # return float(pred[0]), float(pred[1])

def compute_cpa_tcpa(own_x, own_y, own_vx, own_vy, tgt_x, tgt_y, tgt_vx, tgt_vy):
    """
    Given two vessels' positions (x,y in meters) and velocities (vx,vy m/s),
    compute CPA distance and TCPA.
    """
    dx = tgt_x - own_x
    dy = tgt_y - own_y
    dvx = tgt_vx - own_vx
    dvy = tgt_vy - own_vy

    dv2 = dvx**2 + dvy**2
    if dv2 < 1e-6:
        # Nearly parallel -> CPA is current distance, TCPA undefined
        return ( (dx**2 + dy**2)**0.5, None )

    tcpa = -(dx*dvx + dy*dvy) / dv2
    if tcpa < 0 or tcpa > MAX_TCPA:
        # Already diverging or too far into future
        return ( (dx**2 + dy**2)**0.5, tcpa )

    # Position at tcpa
    cx = dx + dvx*tcpa
    cy = dy + dvy*tcpa
    cpa = (cx**2 + cy**2)**0.5
    return (cpa, tcpa)

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.json_schema(
            core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(cls._validate_string),
                ])
            ])
        )
    
    @classmethod
    def _validate_string(cls, v: str) -> ObjectId:
        try:
            return ObjectId(v)
        except Exception as exc:
            raise ValueError("Invalid ObjectId") from exc
        
    

class VesselRecord(BaseModel):
    sourcemmsi: str
    navigationalstatus: Optional[int] = None
    rateofturn: Optional[float] = None
    speedoverground: Optional[float] = None
    courseoverground: Optional[float] = None
    trueheading: Optional[float] = None
    lon: float
    lat: float
    t: int  

class VesselBatch(BaseModel):
    previous: Dict[str, VesselRecord]
    current: Dict[str, VesselRecord]
    selected: str

class BatchPayload(BaseModel):
    previous: Dict[str, VesselRecord] = Field(default_factory=dict)
    current: Dict[str, VesselRecord]  = Field(default_factory=dict)
    prediction_horizon: Optional[int] = None  # optional override

class PlasticReportIn(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    description: Optional[str] = Field(default=None, max_length=1000)
    image_filename: Optional[str] = None


class PlasticReportOut(PlasticReportIn):
    id: str = Field(alias="_id")
    created_at: datetime

    class Config:
        populate_by_name = True
        from_attributes = True
        json_encoders = {ObjectId: str}


def create_app() -> FastAPI:
    app = FastAPI(title="TadomSea API", version="0.1.0")

    # CORS - Explicitly allow frontend origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # MongoDB client
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "tadomsea")

    # Mount static files for uploaded images
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=upload_dir), name="uploads")

    if MongoClient is None:
        client = None
        db = None
        collection = None
    else:
        client = MongoClient(mongodb_uri)
        db = client[db_name]
        collection = db["plastic_reports"]
        try:
            collection.create_index([("location", "2dsphere")])
            collection.create_index([("created_at", ASCENDING)])
        except Exception:
            pass

    @app.get("/")
    def root():
        return {"status": "ok", "service": "TadomSea API"}

    @app.get("/api/health")
    def health():
        return {"ok": True, "timestamp": datetime.now(timezone.utc).isoformat()}
    
    @app.get("/api/health/db")
    def health_db():
        if collection is None:
            return {"ok": False, "database": "not configured"}
        try:
            # Try to ping the database
            db.command('ping')
            return {"ok": True, "database": "connected", "timestamp": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            return {"ok": False, "database": "error", "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.post("/api/reports", response_model=PlasticReportOut)
    async def create_report(
        latitude: float = Form(...),
        longitude: float = Form(...),
        description: Optional[str] = Form(None),
        image: Optional[UploadFile] = File(None)
    ):
        try:
            if collection is None:
                raise HTTPException(status_code=500, detail="Database not configured")
            
            # Validate coordinates
            if not (-90 <= latitude <= 90):
                raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90")
            if not (-180 <= longitude <= 180):
                raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180")

            # Handle image upload
            image_filename = None
            if image:
                # Generate unique filename
                file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
                image_filename = f"plastic_report_{uuid.uuid4()}.{file_extension}"
                
                # Save image to local storage (in production, use cloud storage like S3)
                upload_dir = "uploads"
                os.makedirs(upload_dir, exist_ok=True)
                
                with open(os.path.join(upload_dir, image_filename), "wb") as buffer:
                    content = await image.read()
                    buffer.write(content)

            document = {
                "latitude": latitude,
                "longitude": longitude,
                "description": description,
                "image_filename": image_filename,
                # GeoJSON point for potential geo queries
                "location": {
                    "type": "Point",
                    "coordinates": [longitude, latitude],
                },
                "created_at": datetime.now(timezone.utc),
            }
            result = collection.insert_one(document)
            created = collection.find_one({"_id": result.inserted_id})
            assert created is not None
            
            # Convert ObjectId to string for JSON serialization
            created["_id"] = str(created["_id"])
            return PlasticReportOut(**created)
        except Exception as e:
            print(f"Error creating report: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create report: {str(e)}")

    @app.get("/api/reports", response_model=List[PlasticReportOut])
    def list_reports(limit: int = 100, since: Optional[datetime] = None):
        if collection is None:
            raise HTTPException(status_code=500, detail="Database not configured")

        query = {}
        if since is not None:
            query["created_at"] = {"$gte": since}
        cursor = collection.find(query).sort("created_at", -1).limit(max(1, min(limit, 1000)))
        
        # Convert ObjectIds to strings for JSON serialization
        reports = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            reports.append(PlasticReportOut(**doc))
        
        return reports
    
    #FOR VESSEL SIMULATION PART
    dfZone14 = pd.read_csv("ais_prepared_14N.csv")
    dfZone14 = dfZone14.sort_values("t")

    # Group by timestamp
    grouped = dfZone14.groupby("t")

    async def event_generator():
        for timestamp, group in grouped:
            # Convert to {sourcemmsi: {other fields}}
            batch = {
                str(row["sourcemmsi"]): {
                    k: v for k, v in row.items() if k != "sourcemmsi"
                }
                for _, row in group.iterrows()
            }
            data = {
                "timestamp": int(timestamp),
                "records": batch
            }
            # Send as one SSE event
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(3)  # fixed 60s wait

    @app.get("/start-simulation")
    async def start_simulation():
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    

    # @app.post("/collision-risk")
    # def compute_collision(batch: VesselBatch):
    #     results = []
    #     for mmsi, curr in batch.current.items():
    #         if mmsi in batch.previous:
    #             prev = batch.previous[mmsi]
    #             # Example: compute delta
    #             dlon = curr.lon - prev.lon
    #             dlat = curr.lat - prev.lat

               
    #     return {"risks": results}

    @app.post("/collision-risk")
    def compute_collision(batch: VesselBatch):
        """
        Batch-based collision predictor.
        - Expects previous and current dicts keyed by MMSI (string).
        - Returns predicted future positions (lon/lat) and risk entries (CPA/TCPA) for pairs.
        """

        try:
            prev = batch.previous
            curr = batch.current
            selected = batch.selected
            # print(curr)
            # print(prev)
            # print(selected)

            if selected not in curr or selected not in prev:
                raise HTTPException(status_code=400, detail="Selected vessel missing in previous or current data")

            # First pass: for each vessel with both prev+curr, compute UTM deltas, model input, predicted displacement & future pos.
            predicted = {}  # mmsi -> dict with utm/current/predicted positions
            for mmsi, c in curr.items():
                # print("im runin")
                if mmsi not in prev:
                    # no previous, cannot compute motion -> skip (client requested)
                    continue
                # print("im runin2")
                p = prev[mmsi]

                # ensure timestamps present
                if getattr(p, "t", None) is None or getattr(c, "t", None) is None:
                    # skip if missing timestamps
                    continue
                # print("im runin3")
                # Convert lon/lat to UTM (meters)
                prev_x, prev_y = lonlat_to_utm(p.lon, p.lat)
                curr_x, curr_y = lonlat_to_utm(c.lon, c.lat)
                # print("im runin33")

                # Compute deltas
                dt = float(c.t - p.t)  # seconds
                if dt <= 0:
                    # invalid time difference, skip
                    # print("check time")
                    # print(c.t)
                    # print(p.t)
                    # print(float(c.t - p.t))
                    continue
                # print("im runin333")

                dlon_m = curr_x - prev_x
                dlat_m = curr_y - prev_y

                # Build model input: [dt(t), dlon(t), dlat(t), dt(t+1)]
                model_input = build_model_input(dt, dlon_m, dlat_m, 60)

                # print("im runin4")

                # Predict displacement in meters for next dt_future
                try:
                    pred_dlon, pred_dlat = predict_displacement(model_input)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")
                
                # print("im runin5")
                # Predicted UTM future position at time curr.t + dt_future
                pred_x = curr_x + pred_dlon
                pred_y = curr_y + pred_dlat

                vx = dlon_m / dt
                vy = dlat_m / dt

                # Store computed info
                predicted[mmsi] = {
                    "mmsi": mmsi,
                    "curr": {"t": c.t, "utm_x": curr_x, "utm_y": curr_y},
                    "prediction": {"utm_x": pred_x, "utm_y": pred_y},
                    "velocity": {"vx": vx, "vy": vy},
                }
                # predicted[mmsi] = {
                #     "mmsi": mmsi,
                #     "prev": {"t": int(p.t), "utm_x": prev_x, "utm_y": prev_y, "lon": p.lon, "lat": p.lat},
                #     "curr": {"t": int(c.t), "utm_x": curr_x, "utm_y": curr_y, "lon": c.lon, "lat": c.lat},
                #     "motion": {"dt": dt, "dlon_m": dlon_m, "dlat_m": dlat_m},
                #     "prediction": {
                #         "dt_future": 60,
                #         "pred_dlon_m": pred_dlon,
                #         "pred_dlat_m": pred_dlat,
                #         "pred_utm_x": pred_x,
                #         "pred_utm_y": pred_y,
                #     }
                # }

            print("sometin")
            print(len(predicted))

            # Selected section
            sel = predicted.get(selected)
            if not sel:
                raise HTTPException(status_code=400, detail="Selected vessel cannot be predicted")
            
            sel_x, sel_y = sel["curr"]["utm_x"], sel["curr"]["utm_y"]
            sel_vx, sel_vy = sel["velocity"]["vx"], sel["velocity"]["vy"]

            #try to find the tcpa and cpa
            risks = []
            for mmsi, vessel in predicted.items():
                if mmsi == selected: continue

                cpa, tcpa = compute_cpa_tcpa(
                    sel_x, sel_y, sel_vx, sel_vy,
                    vessel["curr"]["utm_x"], vessel["curr"]["utm_y"],
                    vessel["velocity"]["vx"], vessel["velocity"]["vy"]
                )

                risk = cpa < CPA_THRESHOLD and (tcpa is None or 0 <= tcpa <= MAX_TCPA)

                risks.append({
                    "target_mmsi": mmsi,
                    "cpa": cpa,
                    "tcpa": tcpa,
                    "risk": risk
                })

            return {
                "selected": selected,
                "predicted": predicted[selected],
                "risks": risks
            }
        except Exception as e:
            print(f"Error bah: {e}")
        


    return app


app = create_app()
