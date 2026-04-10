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
    level=logging.DEBUG, 
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("pypdf").setLevel(logging.WARNING)
logging.getLogger("unstructured").setLevel(logging.INFO)
from contextlib import asynccontextmanager
from src.startup import initialize_vector_store, initialize_rag_facade

rag_facade: RagFacade

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Setting up application...")
    try:
        initialize_vector_store()
    except Exception as e:
        logging.error(f"Error initializing vector store: {e}")
        logging.shutdown()
        exit(1) 
        
    logging.info("Vector store initialized successfully")
    global rag_facade
    rag_facade = initialize_rag_facade()
    yield


app = FastAPI(title="AI Assistant", lifespan=lifespan)

@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    logging.debug(f"APIException: {exc.detail}, Code: {exc.code}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "code": exc.code 
        }
    )
    


class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    sources: list[Dict]
    chunks: list[Dict]

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
