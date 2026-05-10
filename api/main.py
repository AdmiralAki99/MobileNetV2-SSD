from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import experiments, training, export

# Creating the app
app = FastAPI(title="ML Pipeline API", version="1.0.0")

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

# Including the routers
app.include_router(experiments.router,prefix="/api/experiments")
app.include_router(training.router,prefix="/api/training")
app.include_router(export.router,prefix="/api/export")

@app.get("/api/health")
def health():
    return {'status':'ok'}