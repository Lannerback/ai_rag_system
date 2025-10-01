"""
API file that create the FastAPI application and defines the endpoints.
"""
from typing import Dict
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel

from src.common.APIException import APIException 
from src.ai.rag_facade import RagFacade
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,  # or INFO
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="AI Assistant",)

@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    logging.debug(f"APIException: {exc.detail}, Code: {exc.code}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "code": exc.code  # custom code you added
        }
    )
    
rag_facade = RagFacade()

@app.on_event("startup")
async def startup_event():
    """Initialize the vector store when the application starts."""
    logging.info("Starting application...")

    start_time = time.perf_counter()

    logging.info("Initializing vector store...")
    rag_facade.initialize_vector_store()

    duration_sec = time.perf_counter() - start_time
    duration_min = duration_sec / 60

    logging.info(f"Vector store initialized in {duration_min:.2f} minutes "
                 f"({duration_sec:.2f} seconds).")
    logging.info("Application startup complete")

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    sources: list[Dict]

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """
        Ask a question and get an answer based on the documentation.
    """
    logging.info(f"Received question: {question.text}")
    
    try:
        response = rag_facade.answer_question(question.text)                                    
        return response
            
    except Exception as e:
        logger.exception("Unhandled exception in /ask")
        raise HTTPException(status_code=500, detail=str(e)) from e
