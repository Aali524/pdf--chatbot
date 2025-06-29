from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from typing import List
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import tempfile
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import fitz

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def chatbot_query(message: str):
    # Replace this with your actual chatbot logic (e.g., RAG or LangChain call)
    return f"Thanks for filling the form! How can I assist you with your ad campaign, {message}?"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Pre-load your PDF (example: "media_kit.pdf")
def load_pdf_text(file_path="media_kit.pdf"):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

pdf_text = load_pdf_text()

@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit/", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    address: str = Form(...)
):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "submitted": True,
        "name": name,
        "email": email,
        "phone": phone,
        "address": address
    })
    
def get_pdf_text(pdf_paths):
    text = ""
    for path in pdf_paths:
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not available in the context, say "Answer is not available in the context."

    Context: {context}
    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@app.post("/process_pdfs")
async def process_pdfs(files: List[UploadFile] = File(...)):
    temp_paths = []
    for file in files:
        suffix = os.path.splitext(file.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(await file.read())
        temp_file.close()
        temp_paths.append(temp_file.name)

    raw_text = get_pdf_text(temp_paths)
    chunks = get_text_chunks(raw_text)
    get_vector_store(chunks)

    for path in temp_paths:
        os.remove(path)

    return JSONResponse({"message": "PDFs processed and indexed successfully."})

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)
    chain = get_conversational_chain()
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return JSONResponse({"answer": result["output_text"]})
