from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .routers import experiments, training, export, etl, ops
from .services.config_library import sync_config_library

# Creating the app
app = FastAPI(title="ML Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Including the routers
app.include_router(experiments.router, prefix="/api/experiments")
app.include_router(training.router, prefix="/api/training")
app.include_router(export.router, prefix="/api/export")
app.include_router(etl.router, prefix="/api/etl")
app.include_router(ops.router, prefix="/api/ops")

@app.on_event("startup")
def _startup():
    sync_config_library()

@app.get("/api/health")
def health():
    return {"status": "ok"}


# Mouting the UI
app.mount("/", StaticFiles(directory="ui/dist", html=True), name="ui")
