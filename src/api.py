"""
API file that create the FastAPI application and defines the endpoints.
"""
from typing import Dict
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from dotenv import load_dotenv

from src.common.APIException import APIException 
from src.ai.ai_facade import AiFacade
logger = logging.getLogger(__name__)

load_dotenv()

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
    
ai_facade = AiFacade()

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
        response = ai_facade.answer_question(question.text)                                    
        return response
            
    except Exception as e:
        logger.exception("Unhandled exception in /ask")
        raise HTTPException(status_code=500, detail=str(e)) from e
