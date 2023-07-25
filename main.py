from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

import pandas as pd

load_dotenv()
app = FastAPI()

# Configure CORS settings
origins = [
    "*"
    # Add more allowed origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# df = pd.read_csv("docs/SLPDirectory.csv")


agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    "docs/SLPDirectory.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# agent = create_pandas_dataframe_agent(ChatOpenAI(
#     temperature=0, model="gpt-3.5-turbo-16k"), df, verbose=True, max_iterations=5)

# Define a Pydantic model for the request payload


class InputData(BaseModel):
    question: str


def run_agent(query):
    return agent.run(query)


@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}


@app.post("/chat-stream")
def process_chat_stream(input_data: InputData):
    print("-----------", input_data)
    # Perform processing on the input string
    # Example: Convert to uppercase
    processed_value = run_agent(input_data.question)

    # Return the processed value
    return {"processed_value": processed_value}
