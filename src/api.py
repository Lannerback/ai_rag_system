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
from contextlib import asynccontextmanager
from src.ai.builder_dispatcher import BuilderDispatcher
from src.common.app_context import set_app_context
from src.ai.vector_store_service.faiss.faiss_vector_store_initializer import VectorStoreInitializer
from src.ai.vector_store_service.vector_store_facade import VectorStoreFacade

rag_facade = None

def startup_event():
    """Initialize the vector store when the application starts."""
    start_time = time.perf_counter()

    logging.info("Initializing vector store...")
    vector_store_initializer = VectorStoreInitializer()
    vector_store_initializer.initialize_vector_store()

    duration_sec = time.perf_counter() - start_time
    duration_min = duration_sec / 60

    logging.info(f"Vector store initialized in {duration_min:.2f} minutes "
                 f"({duration_sec:.2f} seconds).")
    logging.info("Application startup complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Setting up application...")
    # Set the environment and vector store
    # Make app globally accessible
    set_app_context(app)

    builder = BuilderDispatcher()
    app.state.llm = builder.get_llm()
    app.state.embedder = builder.get_embedder_store()
    global rag_facade
    rag_facade = RagFacade()
    
    startup_event()

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
