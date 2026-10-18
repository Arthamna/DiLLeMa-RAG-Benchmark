from fastapi import FastAPI
import uvicorn
from llm import conversation_chain, create_chain
from llm2 import get_stream_chain
from pydantic import BaseModel

app = FastAPI()

@app.get("/ping")
async def root():
    return {"message": "pong"}

class Question(BaseModel):
   question: str
   sid: str

chain_store = {
   "123a": create_chain(),
   "123b": create_chain(),
   "123c": get_stream_chain()
}

@app.post("/ask")
async def ask_question(item: Question):
    key = {'question': item.question}
    chain = chain_store[item.sid]
    return chain(key)

from typing import List
from sse_starlette.sse import EventSourceResponse

async def chain_generator(question: str, sid: str):
    async for chunk in chain_store[sid].astream(question):
        yield chunk

@app.post("/ask/stream")
async def generate_stream(item: Question):
    return EventSourceResponse(chain_generator(item.question, item.sid), media_type='text/event-stream')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)