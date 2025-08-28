#!/usr/bin/env python3
"""
Simple script to run the TadomSea API server
"""
import uvicorn
from main import app

if __name__ == "__main__":
    print("Starting TadomSea API server...")
    print("Frontend should be accessible at: http://localhost:3000")
    print("API will be available at: http://localhost:8000")
    print("Health check: http://localhost:8000/api/health")
    print("Database health: http://localhost:8000/api/health/db")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
