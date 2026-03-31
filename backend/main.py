# Load environment variables first
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Quick feature: Add basic request logging
REQUEST_LOG_FILE = "requests.log"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from routes import screenshot, generate_code, home, evals

app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

# Simple request logging middleware for new analytics feature
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Basic logging - could be improved with proper logging framework
    log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {request.method} {request.url} - {response.status_code} - {process_time:.2f}s\n"
    with open(REQUEST_LOG_FILE, "a") as f:
        f.write(log_entry)
    
    return response

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes
app.include_router(generate_code.router)
app.include_router(screenshot.router)
app.include_router(home.router)
app.include_router(evals.router)
