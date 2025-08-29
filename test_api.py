#!/usr/bin/env python3
"""
Simple test script to verify the TadomSea API is working
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test basic health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_db_health():
    """Test database health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/health/db")
        print(f"DB Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"DB Health check failed: {e}")
        return False

def test_cors():
    """Test CORS headers"""
    try:
        response = requests.options(f"{BASE_URL}/api/reports")
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers'),
        }
        print(f"CORS headers: {cors_headers}")
        return 'Access-Control-Allow-Origin' in response.headers
    except Exception as e:
        print(f"CORS test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing TadomSea API...")
    print("=" * 40)
    
    health_ok = test_health()
    db_ok = test_db_health()
    cors_ok = test_cors()
    
    print("=" * 40)
    print(f"Results:")
    print(f"  Health: {'✓' if health_ok else '✗'}")
    print(f"  Database: {'✓' if db_ok else '✗'}")
    print(f"  CORS: {'✓' if cors_ok else '✗'}")
    
    if all([health_ok, db_ok, cors_ok]):
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")

