from fastapi import FastAPI,HTTPException
import uvicorn 
from pydantic import BaseModel
from crew import Legaldraft

app=FastAPI()

class queryRequest(BaseModel):
    query_text:str

@app.get('/')
def root():
    return {"Hello": "World"}


@app.post("/submit_query")
def submit_query_endpoint(request:queryRequest):
    try:
        inputs = {'topic': request.query_text}
        result = Legaldraft().crew().kickoff(inputs=inputs)
        return {"drafted_document": result.raw}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__== "__main__":
    port=8000
    print(f"Running the fastAPI server on port {port}.")
    uvicorn.run(app,host="0.0.0.0",port=port)