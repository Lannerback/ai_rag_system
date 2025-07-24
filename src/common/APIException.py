from fastapi import HTTPException

class APIException(HTTPException):

    def __init__(self, status_code: int, detail: str, code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.code = code 
