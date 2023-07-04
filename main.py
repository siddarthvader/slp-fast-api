from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
app = FastAPI()

# Configure CORS settings
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    # Add more allowed origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    "docs/SLPDirectory.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

# Define a Pydantic model for the request payload
class InputData(BaseModel):
    question: str


def run_agent(query):
    return agent.run(query)



@app.post("/chat-stream")
def process_chat_stream(input_data:InputData):
    print("-----------",input_data)
    # Perform processing on the input string
    processed_value = run_agent(input_data.question)  # Example: Convert to uppercase

    # Return the processed value
    return {"processed_value": processed_value}
