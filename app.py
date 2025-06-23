from flask import Flask, request, jsonify
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
import uuid
from dotenv import load_dotenv

load_dotenv() 


# Initialize Flask app
app = Flask(__name__)

# Initialize memory and tools
memory = MemorySaver()

llm = ChatGroq(model="qwen-qwq-32b", temperature=0.5)
# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

tool = TavilySearch(max_results=7)
tools = [tool]
llm_with_tool = llm.bind_tools(tools)

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define LangGraph node
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tool.invoke(state["messages"])]}

# Build LangGraph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile(checkpointer=memory)

# In-memory store for threads (not suitable for production)
sessions = {}

@app.route("/")
def index():
    return "<h1> 🤖 Sales Assistant is ready. Go to the chat endpoint <h1>"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    thread_id = data.get("thread_id")

    if not user_input:
        return jsonify({"error": "No user input provided"}), 400

    # Assign thread ID if not provided
    if not thread_id:
        thread_id = str(uuid.uuid4())

    # Initialize session if not present
    if thread_id not in sessions:
        sessions[thread_id] = [{
            "role": "user",
            "content": """ You are a friendly, helpful junior software consultant named Sales Assistant.
                           Greet users warmly in short sentence, when they start a conversation.

                            Your job is to:
                            - Understand the user's business goal or software idea
                            - Ask questions to clarify the project scope
                            - Suggest potential software solutions, tools, and technologies
                            - Provide rough timelines and budget estimates"""
        }]

    # Append current user input
    sessions[thread_id].append({"role": "user", "content": user_input})

    # Run the LangGraph
    config = {"configurable": {"thread_id": thread_id}}
    response = graph.invoke({"messages": sessions[thread_id]}, config=config)

    # Update messages and get last AI reply
    sessions[thread_id] = response["messages"]
    ai_reply = response["messages"][-1].content

    return jsonify({
        "thread_id": thread_id,
        "response": ai_reply
    })


if __name__ == "__main__":
    app.run(debug=True)
