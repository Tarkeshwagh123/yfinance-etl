import streamlit as st
import pdfplumber
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockLLM 
from langchain.chains import RetrievalQA
import boto3
from langchain.prompts import PromptTemplate

def extract_text_from_pdf(uploaded_file, password=None):
    """Extracts text from an uploaded PDF file."""
    import pdfplumber
    try:
        with pdfplumber.open(uploaded_file, password=password) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")
    
def generate_pdf_summary(text: str, qa_chain=None) -> str:
    """Generate a comprehensive financial document summary using RAG and LLM."""
    # If no QA chain provided and the text is short enough, do simple summarization
    if qa_chain is None and len(text) < 5000:
        return text[:1000] + "…" if len(text) > 1000 else text
    
    # For longer documents or when we have a QA chain, use RAG for detailed summary
    try:
        # Define specific financial document summary questions
        summary_questions = [
            "What is the type of this financial document?",
            "What are the key financial figures or metrics mentioned?",
            "What time period does this document cover?",
            "What are the main financial insights from this document?",
            "Are there any notable trends or concerns mentioned?"
        ]
        
        # If we have an existing QA chain, use it directly
        if qa_chain is not None:
            # Get answers to summary questions using the existing QA chain
            summary_parts = []
            for question in summary_questions:
                try:
                    response = qa_chain.invoke({"query": question})
                    answer = response.get("result", "Information not available")
                    if answer and len(answer) > 5:  # Only include non-empty answers
                        summary_parts.append(f"**{question}**\n{answer}\n")
                except Exception as e:
                    summary_parts.append(f"**{question}**\nUnable to determine: {str(e)}\n")
        
        # If no QA chain provided, create a one-time chain for summarization
        else:
            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            docs = splitter.create_documents([text])
            
            # Create embeddings and vector store
            embedder = FastEmbedEmbeddings()
            db = FAISS.from_documents(docs, embedder)
            retriever = db.as_retriever(search_kwargs={"k": 5})
            
            # Set up LLM with Bedrock (with fallback to OpenAI if needed)
            try:
                llm = BedrockLLM(
                    model_id="amazon.titan-text-premier-v1:0",
                    client=boto3.client("bedrock-runtime", region_name="us-east-1"),
                    model_kwargs={"temperature": 0.3, "top_p": 0.9}
                )
            except Exception:
                from langchain_community.llms import OpenAI
                llm = OpenAI(temperature=0.3)
            
            # Get answers to summary questions by querying with retriever
            summary_parts = []
            for question in summary_questions:
                try:
                    # Get relevant context from the retriever
                    context_docs = retriever.get_relevant_documents(question)
                    context = "\n\n".join([doc.page_content for doc in context_docs])
                    
                    # Construct the prompt
                    summary_prompt = f"""
                    Based on the following excerpt from a financial document, please answer this question:
                    
                    Question: {question}
                    
                    Financial Document Content:
                    {context}
                    
                    Answer the question concisely based only on the provided content. If the information isn't available, say so.
                    """
                    
                    answer = llm.invoke(summary_prompt)
                    if answer and len(answer) > 5:  # Only include non-empty answers
                        summary_parts.append(f"**{question}**\n{answer}\n")
                except Exception as e:
                    summary_parts.append(f"**{question}**\nUnable to determine: {str(e)}\n")
        
        # Combine the answers into a complete summary
        if summary_parts:
            return "# Financial Document Summary\n\n" + "\n".join(summary_parts)
        else:
            # Fallback if RAG summary failed
            return "Could not generate a summary for this document. " + text[:500] + "…"
            
    except Exception as e:
        # Ultimate fallback on any error
        return f"Error generating summary: {str(e)}\n\nPartial content preview:\n{text[:500]}…"

def render_chat_button():
    """Shows only the chat button in the sidebar that sets state."""
    if st.sidebar.button("💬 PDF Assistant", use_container_width=True):
        st.session_state.chat_dialog_open = True
        st.rerun()

def run_pdf_rag_chatbot(mode='full'):
    """
    Manages the PDF RAG chatbot UI.
    
    Args:
        mode (str): Either 'button' (just show button in sidebar), 
                'popover' (just show popover in main area if state is set),
                or 'full' (default, show both)
    """
    # --- Initialize Session State ---
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = None
    if "pdf_qa_chain" not in st.session_state:
        st.session_state.pdf_qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_dialog_open" not in st.session_state:
        st.session_state.chat_dialog_open = False
        st.session_state.chat_dialog_open_text = False
    if "show_summary" not in st.session_state:
        st.session_state.show_summary = False
        st.session_state.show_summary_text = False
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
        

    # First, add these CSS rules to the style block at the beginning:
    st.markdown("""
        <style>
        /* Set fixed width for popover and handle text overflow */
        .stPopover {
            max-width: 350px !important;
            width: 350px !important;
        }
        
        /* Handle text overflow for long filenames */
        .uploadedFileName {
            max-width: 300px !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        
        /* Make file uploader container respect width */
        .stFileUploader > div {
            max-width: 320px !important;
        }
        
        /* Make text inputs respect width */
        .stTextInput > div {
            max-width: 320px !important;
        }
        
        /* Control subheaders inside popover */
        .stPopover h3 {
            font-size: 1rem !important;
            margin-bottom: 8px !important;
        }
        
        /* Make buttons more compact in popovers */
        .stPopover .stButton > button {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            min-height: 0 !important;
            height: 2rem !important;
            font-size: 0.8rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    
    
    # Button in sidebar mode
    if mode == 'button':
        render_chat_button()
        return
    
    # Popover in main area mode
    if mode == 'popover' and st.session_state.chat_dialog_open:
        with st.popover("PDF Financial Assistant"):
            # --- Step 1: PDF Upload ---
            if st.session_state.pdf_text is None and st.session_state.show_summary == False:
                st.subheader("Upload Your Document")
                uploaded_file = st.file_uploader(
                    "Upload a financial PDF to begin.", type=["pdf"], key="pdf_upload_dialog"
                )
                password = st.text_input(
                    "PDF Password (if protected)", type="password", key="pdf_password_dialog"
                )

                if uploaded_file:
                    try:
                        with st.spinner("Processing PDF... This may take a moment."):
                            # 1. Extract text
                            text = extract_text_from_pdf(uploaded_file, password if password else None)
                            st.session_state.pdf_text = text
                            st.session_state.show_summary = True
                            st.session_state.pdf_uploaded = True

                            # 2. Split text into chunks
                            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                            docs = splitter.create_documents([text])

                            # 3. Embed and create vector store
                            embedder = FastEmbedEmbeddings()
                            db = FAISS.from_documents(docs, embedder)

                            # 4. Set up LLM
                            try:
                                llm = BedrockLLM(
                                    model_id="amazon.titan-text-premier-v1:0",
                                    client=boto3.client("bedrock-runtime", region_name="us-east-1"),
                                    model_kwargs={"temperature": 0.5, "top_p": 0.9}
                                )
                            except Exception:
                                from langchain_community.llms import OpenAI
                                st.warning("Bedrock LLM not found. Falling back to OpenAI.")
                                llm = OpenAI(temperature=0.7)

                            # 5. Create prompt template
                            prompt_template = """
                            Use the financial context below to answer the question accurately.
                            If the answer is not in the context, say so.

                            Context: {context}
                            Question: {question}
                            Answer:
                            """
                            prompt = PromptTemplate(
                                template=prompt_template, input_variables=["context", "question"]
                            )

                            # 6. Create QA chain
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                retriever=db.as_retriever(search_kwargs={"k": 3}),
                                chain_type="stuff",
                                chain_type_kwargs={"prompt": prompt}
                            )

                            # 7. Store chain and initial message in session state
                            st.session_state.pdf_qa_chain = qa_chain
                            st.session_state.chat_history.append(
                                ("bot", "PDF processed! How can I help you with this document?")
                            )
                            st.rerun()

                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
                        st.session_state.pdf_text = None # Reset on failure
                        st.session_state.show_summary = False
                        
            # --- After upload: Show two action buttons ---
            if st.session_state.pdf_uploaded and not (st.session_state.show_summary_text or st.session_state.chat_dialog_open_text):
                st.subheader("What would you like to do?")
                c1, c2 = st.columns(2, gap="small")
                with c1:
                    if st.button("Summary"):
                        st.session_state.show_summary_text = True
                        st.rerun()
                with c2:
                    if st.button("Chat Bot"):
                        st.session_state.chat_dialog_open_text = True
                        st.rerun()

            # --- Show PDF Summary ---
            if st.session_state.show_summary_text:
                st.subheader("📄 PDF Summary")
                if "pdf_summary" not in st.session_state or not st.session_state.pdf_summary:
                    with st.spinner("Generating detailed financial summary..."):
                        summary = generate_pdf_summary(
                            st.session_state.pdf_text, 
                            st.session_state.pdf_qa_chain
                        )
                        # Store the summary in session state
                        st.session_state.pdf_summary = summary
                else:
                    # Use the cached summary
                    summary = st.session_state.pdf_summary
                st.markdown(summary)
                col1, col2 = st.columns(2, gap="small")
                with col1:
                    if st.button("← Back"):
                        st.session_state.show_summary_text = False
                        st.session_state.chat_dialog_open_text = False
                        st.rerun()
                with col2:
                    if st.button("Close ×"):
                        st.session_state.chat_dialog_open = False
                        st.rerun()

            # --- Step 2: Chat Interface ---
            
            elif st.session_state.chat_dialog_open_text:
                st.subheader("Chat with Your Financial Document")

                # Display chat history in a scrollable container
                chat_container = st.container()
                with chat_container:
                    for sender, message in st.session_state.chat_history:
                        st.markdown(f"**{'You' if sender == 'user' else 'Assistant'}:** {message}")
                        st.markdown("---")

                # Chat input form
                with st.form(key="chat_form_dialog", clear_on_submit=True):
                    user_question = st.text_input(
                        "Ask a question:", key="user_question_dialog", placeholder="e.g., What is the total investment in mutual funds?"
                    )
                    submit_button = st.form_submit_button("Send")

                    if submit_button and user_question:
                        st.session_state.chat_history.append(("user", user_question))
                        try:
                            with st.spinner("Finding answer..."):
                                response = st.session_state.pdf_qa_chain.invoke({"query": user_question})
                                answer = response.get("result", "Sorry, I could not find an answer.")
                                st.session_state.chat_history.append(("bot", answer))
                                st.rerun()
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")

                # Option to upload a new document
                col1, col2, col3 = st.columns([1, 1, 1], gap="small")
                with col1:
                    if st.button("← Back"):
                        st.session_state.show_summary_text = False
                        st.session_state.chat_dialog_open_text = False
                        st.rerun()
                with col2:
                    if st.button("New"):
                        st.session_state.pdf_text = None
                        st.session_state.show_summary = False
                        st.session_state.pdf_qa_chain = None
                        st.session_state.chat_history = []
                        st.rerun()
                with col3:
                    if st.button("Close ×"):
                        st.session_state.chat_dialog_open = False
                        st.rerun()
    
    # Full mode (default, original behavior)
    if mode == 'full':
        # --- CSS for the fixed chat button ---
        st.markdown("""
            <style>
            .fixed-bottom-container {
                position: fixed;
                bottom: 20px; /* Adjust vertical position */
                left: 50%;
                transform: translateX(-50%); /* Center the button */
                z-index: 1000;
            }
            </style>
        """, unsafe_allow_html=True)

        # --- Create a container and apply the custom class ---
        st.markdown('<div class="fixed-bottom-container">', unsafe_allow_html=True)

        # --- Main Page Trigger to Open Chatbot ---
        # This button will appear fixed at the bottom-center of the page.
        if st.button("💬"):
            st.session_state.chat_dialog_open = True
            st.rerun()

        # Close the custom div
        st.markdown('</div>', unsafe_allow_html=True)

        # Run the popover part
        run_pdf_rag_chatbot(mode='popover')