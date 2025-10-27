#!/usr/bin/env python3
import asyncio
import httpx

async def test_endpoints():
    endpoints = [
        "http://localhost:9003",
        "http://localhost:9003/mcp", 
        "http://localhost:9003/health",
        "http://localhost:9003/api",
        "http://localhost:15501/v1"
    ]
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for url in endpoints:
            try:
                response = await client.get(url)
                print(f"✅ {url} - Status: {response.status_code}")
                if response.headers.get('content-type', '').startswith('application/json'):
                    print(f"   Response: {response.json()}")
                else:
                    print(f"   Response: {response.text[:100]}")
            except Exception as e:
                print(f"❌ {url} - Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_endpoints())