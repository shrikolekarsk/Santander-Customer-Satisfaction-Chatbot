# health_insurance_chatbot.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import SystemMessage
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY
)

# MySQL DB Connection Details
host = 'localhost'
port = '3306'
username = 'root'
password = ''  # Add if needed
database_schema = "scs_project"

mysql_uri = f"mysql+pymysql://{username}@{host}:{port}/{database_schema}"
db = SQLDatabase.from_uri(mysql_uri, include_tables=["customer_2"], sample_rows_in_table_info=2)

# Create SQL Database Chain
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

# Retrieve DB context
def retrieve_from_db(query: str) -> str:
    db_context = db_chain.run(query)
    return db_context.strip()

# Generate LLM response
def generate(query: str) -> str:
    db_context = retrieve_from_db(query)

    system_message = """You are a data expert specializing in customer satisfaction analytics.

    Your task is to analyze and answer user questions based on the Santander Customer Satisfaction dataset.
    This dataset includes anonymized features like `var_0`, `var_1`, ..., along with the target column `TARGET`,
    where `TARGET = 1` indicates a dissatisfied customer.

    Use the sample dataset context and your knowledge to give a concise and helpful answer.
    """

    human_prompt = HumanMessagePromptTemplate.from_template(
        """Input:
{human_input}

Context:`+
{db_context}

Output:"""
    )

    messages = [
        SystemMessage(content=system_message),
        human_prompt.format(human_input=query, db_context=db_context)
    ]

    return llm(messages).content



st.set_page_config(page_title="📊 Customer Satisfaction Chatbot", page_icon="💬", layout="centered")

# Custom UI Styles
st.markdown("""
<style>
body {
    background-color: #111827;
    color: #f1f5f9;
    font-family: 'Segoe UI', sans-serif;
}
h1, h3 {
    color: #e2e8f0;
    text-align: center;
}
.chat-container {
    background-color: #1f2937;
    padding: 2rem;
    border-radius: 18px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.3);
    margin-top: 20px;
}
.response-box {
    background-color: #1e293b;
    padding: 1.5rem;
    border-radius: 12px;
    margin-top: 20px;
    color: #f8fafc;
    font-size: 16px;
    line-height: 1.6;
    box-shadow: 0 2px 10px rgba(0,0,0,0.15);
}
input[type="text"] {
    flex: 1;
    padding: 14px;
    font-size: 16px;
    color: #ffffff;
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    outline: none;
    box-shadow: inset 0 0 0 1px #475569;
}
input[type="text"]::placeholder {
    color: #94a3b8;
}
input[type="text"]:focus {
    border-color: #4f46e5;
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.4);
}
.stButton > button {
    background: linear-gradient(to right, #6366f1, #06b6d4);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 20px;
    font-size: 16px;
    border: none;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.markdown("## 📊 AI-Powered Customer Satisfaction Chatbot")
st.markdown("#### Ask your questions about the Santander Customer Satisfaction dataset.")

# Input Field
with st.form("chat_form", clear_on_submit=False):
    user_input = st.text_input(
        "🧠 Your Question:",
        placeholder="e.g. Which variables are most linked with dissatisfaction?",
        label_visibility="collapsed"
    )
    submitted = st.form_submit_button("🚀 Submit")

# Handle Response
if submitted and user_input:
    with st.spinner("🤖 Thinking..."):
        try:
            answer = generate(user_input)
            st.markdown("### 🤖 Answer")
            st.markdown(f"<div class='response-box'>{answer}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)
