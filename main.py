import json
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError, GetCoreSchemaHandler
from pydantic_core import core_schema
from typing import Optional, List, Any, Annotated
from datetime import datetime, timezone
from bson import ObjectId
import os
import uuid
import asyncio
import pandas as pd

try:
    from pymongo import MongoClient, ASCENDING
except Exception as exc:  # pragma: no cover - allows file to load without deps installed
    MongoClient = None  # type: ignore
    ASCENDING = 1  # type: ignore


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
            await asyncio.sleep(60)  # fixed 60s wait

    @app.get("/start-simulation")
    async def start_simulation():
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


app = create_app()
