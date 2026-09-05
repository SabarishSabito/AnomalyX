import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.routes import router

app = FastAPI(
    title="AnomalyX Intelligence API",
    description="Enterprise-grade AI-powered real-time anomaly detection API",
    version="1.0.0"
)

# Enable CORS for local testing and cross-origin frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API routes under /api
app.include_router(router, prefix="/api")

# Mount frontend static files if frontend folder exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
