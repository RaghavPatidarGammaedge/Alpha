from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from urllib import request

from .query2 import askGeminiAndGroq
app=FastAPI()

class Query(BaseModel) :
    question: str

@app.post('/askGeminiAndGroq')
def geminiAndGroq(query : Query):

    if "hack" in query.question.lower():
            return "Unsafe question detected!"



    result = askGeminiAndGroq(query.question)
    print(query.question)

    if "kill" in result.lower():
        return "[Content blocked by safety policy]"

    return result



