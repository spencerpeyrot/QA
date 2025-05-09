# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from routes.manual_qa.router import router as manual_qa_router
from routes.automated_qa.router import router as automated_qa_router
from routes.automated_qa.ltv_router import router as ltv_router
from routes.automated_qa.tp_router import router as tp_router
from routes.automated_qa.slvb_router import router as slvb_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QA Platform API")

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "QA Platform API is running"}

# Include routers
app.include_router(manual_qa_router)
app.include_router(automated_qa_router)
app.include_router(ltv_router)
app.include_router(tp_router)
app.include_router(slvb_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 