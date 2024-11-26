import streamlit as st
from typing import TypedDict, Annotated, List, Optional, Any
from dotenv import load_dotenv
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.llms import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langgraph.graph import add_messages, END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import PyPDF2
import weave
weave.init("weave-koyeb")

# Load environment variables
load_dotenv()


class CustomHostedLLM(LLM):
    api_url: str = "https://hilarious-allina-triple-m-legal-hackaton-296f44c4.koyeb.app/v1/completions"
    
    @property
    def _llm_type(self) -> str:
        return "custom_hosted_llm"
    @weave.op()
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prompt": prompt,
            "max_tokens": 5000,
            "temperature": 0.1
        }
        try:
            # Add debugging for payload
            print("Sending request with payload:", payload)
            
            response = requests.post(self.api_url, json=payload)
            
            # Add debugging for response
            #st.write("Response status code:", response.status_code)
            #st.write("Response content:", response.text)
            
            response.raise_for_status()
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["text"]
            else:
                return f"API request failed with status code {response.status_code}: {response.text}"
        except Exception as e:
            return f"Error calling API: {str(e)}"

# Initialize the custom LLM
custom_llm = CustomHostedLLM()

@weave.op()
def process_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_embeddings_and_store(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# Initialize session state for storing document content
if 'doc_content' not in st.session_state:
    st.session_state.doc_content = None

# Streamlit Interface
st.title("📚 Document Chat Assistant")
st.write("Upload a PDF document and chat with its contents using RAG")

# Sidebar for PDF upload and processing
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Extract text from PDF
                    text = process_pdf(uploaded_file)
                    # Store the text content
                    st.session_state.doc_content = text
                    st.success("Document processed successfully!")
                    
                    # Create embeddings and store
                    st.session_state.vectorstore = create_embeddings_and_store(text)
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

# Main chat interface
if st.session_state.doc_content:
    # Display document content in expander
    with st.expander("View Document Content"):
        st.write(st.session_state.doc_content)
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    st.header("Chat with your Document")
    user_question = st.text_input("Ask a question about the document:")
    
    if user_question:
        try:
            # Create retrieval chain
            retrieval_chain = ConversationalRetrievalChain.from_llm(
                llm=custom_llm,
                retriever=st.session_state.vectorstore.as_retriever(),
                return_source_documents=True
            )
            
            # Get response
            response = retrieval_chain.invoke({
                "question": user_question,
                "chat_history": st.session_state.chat_history
            })
            
            # Update chat history
            st.session_state.chat_history.append((user_question, response["answer"]))
            
            # Display chat history
            st.subheader("Chat History")
            for q, a in st.session_state.chat_history:
                st.write("🙋‍♂️ **Question:**", q)
                st.write("🤖 **Answer:**", a)
                st.write("---")
            
            # Display source documents
            with st.expander("View Source Documents"):
                for i, doc in enumerate(response["source_documents"], 1):
                    st.write(f"📄 Source {i}:")
                    st.write(doc.page_content)
                    st.write("---")
                    
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
else:
    st.info("👈 Please upload and process a document using the sidebar to start chatting!")

# Clear chat history button
if st.session_state.get('chat_history'):
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Koyeb")