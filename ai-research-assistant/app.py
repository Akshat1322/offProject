import streamlit as st
import ollama
import time
from duckduckgo_search import DDGS
from PyPDF2 import PdfReader

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="centered"
)

# -------------------------------
# AGENTS
# -------------------------------

# 🌐 Web Search Agent
def search_agent(query):
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(r['body'])
        return " ".join(results) if results else "No results found."
    except Exception:
        return "Web search unavailable. Answering from model knowledge."


# 📄 PDF Reader
def read_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


# 🧠 Research Agent
def research_agent(query, context=None):
    system = (
        "You are a research agent. Your job is to extract and present "
        "factual information clearly and concisely based on the provided context. "
        "Do not add information that is not in the context."
    )

    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        prompt = query

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt}
        ]
    )
    return response['message']['content']


# ✍️ Summarizer Agent
def summarizer_agent(text):
    system = (
        "You are a summarizer agent. Your ONLY job is to extract the 5 most "
        "important points from any text. Be concise. No fluff. Return bullet points only."
    )

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': f"Summarize this:\n{text}"}
        ]
    )
    return response['message']['content']


# 📊 Answer Agent
def answer_agent(summary):
    system = (
        "You are a presentation agent. Format information into clean, "
        "student-friendly structured answers. Always use this exact structure:\n\n"
        "**Definition:**\n(1-2 lines)\n\n"
        "**Key Points:**\n- point 1\n- point 2\n...\n\n"
        "**Conclusion:**\n(2-3 lines)"
    )

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': summary}
        ]
    )
    return response['message']['content']


# -------------------------------
# SESSION STATE INIT
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None


# -------------------------------
# UI
# -------------------------------

st.title("🤖 Adaptive AI Research Assistant")
st.caption("Multi-Agent System · Web Search · Optional Document Retrieval")

uploaded_file = st.file_uploader("📄 Upload PDF (Optional)", type=["pdf"])

# Cache PDF in session state — only re-read if file changes
if uploaded_file is not None:
    if st.session_state.last_uploaded_file != uploaded_file.name:
        with st.spinner("Reading PDF..."):
            st.session_state.pdf_text = read_pdf(uploaded_file)
            st.session_state.last_uploaded_file = uploaded_file.name
        st.success(f"✅ PDF loaded: {uploaded_file.name}")
else:
    # Clear PDF if file removed
    st.session_state.pdf_text = None
    st.session_state.last_uploaded_file = None

st.divider()

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
query = st.chat_input("Ask your question...")


# -------------------------------
# MAIN PIPELINE
# -------------------------------

if query:
    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            # --- Step display using st.status ---
            with st.status("Running agents...", expanded=True) as status:

                st.write("📥 Received query...")
                time.sleep(0.4)

                # Mode decision
                if st.session_state.pdf_text:
                    st.write("📄 PDF detected → Document Mode (RAG)")
                    context = st.session_state.pdf_text[:3000]
                    mode_label = "📄 Document Mode (RAG)"
                else:
                    st.write("🌐 No PDF → Web Search Mode")
                    st.write("🔍 Searching the web...")
                    context = search_agent(query)
                    mode_label = "🌐 Web Search Mode"

                time.sleep(0.4)

                st.write("🧠 Research agent processing...")
                data = research_agent(query, context)
                time.sleep(0.4)

                st.write("✍️ Summarizer agent condensing...")
                summary = summarizer_agent(data)
                time.sleep(0.4)

                st.write("📊 Answer agent structuring response...")
                final = answer_agent(summary)
                time.sleep(0.4)

                status.update(label="✅ Done!", state="complete", expanded=False)

            # Show mode badge
            st.info(mode_label)

            # Show final answer
            st.markdown(final)

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final
            })

        except Exception as e:
            st.error(f"⚠️ Something went wrong: {str(e)}")
            st.info("💡 Try rephrasing your question, re-uploading the PDF, or checking if Ollama is running.")