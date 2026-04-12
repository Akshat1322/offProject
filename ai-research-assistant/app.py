import streamlit as st
import ollama
from duckduckgo_search import DDGS
from PyPDF2 import PdfReader
from urllib.parse import quote_plus

st.set_page_config(
    page_title="Adaptive-AI Research Assistant",
    layout="wide"
)

st.markdown("""
<style>
.response-card {
    background-color: #111;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #333;
    margin-top: 10px;
}
.small-text {
    color: grey;
    font-size: 13px;
}
.sidebar-question {
    background-color: #1a1a1a;
    padding: 8px 12px;
    border-radius: 8px;
    border-left: 3px solid #555;
    margin-bottom: 6px;
    font-size: 13px;
    color: #ccc;
    cursor: pointer;
}
.sidebar-question:hover {
    border-left-color: #888;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = {}

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "rerun_query" not in st.session_state:
    st.session_state.rerun_query = None

with st.sidebar:
    st.markdown("Query History")
    st.divider()

    user_questions = [
        (i, msg["content"])
        for i, msg in enumerate(st.session_state.messages)
        if msg["role"] == "user"
    ]

    if not user_questions:
        st.caption("No queries yet")
    else:
        for idx, (msg_index, question) in enumerate(reversed(user_questions)):
            display_text = question if len(question) <= 60 else question[:57] + "..."
            label = f"{display_text}"

            if st.button(label, key=f"q_{msg_index}", use_container_width=True):
                st.session_state.rerun_query = question

    st.divider()

    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.session_state.rerun_query = None
        st.rerun()

st.title("Adaptive AI Research Assistant")
st.caption("Multi-Agent Intelligence System")

uploaded_file = st.file_uploader("Upload Document (Optional)", type=["pdf"])
mode = "Document Mode" if uploaded_file else "Web Mode"
st.markdown(f"Mode: {mode}")
st.divider()

if uploaded_file is not None:
    if st.session_state.last_uploaded_file != uploaded_file.name:
        with st.spinner("Reading document"):
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            st.session_state.pdf_text = text
            st.session_state.last_uploaded_file = uploaded_file.name
else:
    st.session_state.pdf_text = None
    st.session_state.last_uploaded_file = None

def search_agent(query):
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append({
                    "title": r.get("title", "No title"),
                    "body": r.get("body", ""),
                    "link": r.get("href", "")
                })

        if not results:
            encoded_query = quote_plus(query)
            results = [
                {"title": f"Google: {query}", "body": "", "link": f"https://www.google.com/search?q={encoded_query}"},
                {"title": f"Wikipedia: {query}", "body": "", "link": f"https://en.wikipedia.org/wiki/Special:Search?search={encoded_query}"},
                {"title": f"Bing: {query}", "body": "", "link": f"https://www.bing.com/search?q={encoded_query}"}
            ]

        return results

    except Exception:
        encoded_query = quote_plus(query)
        return [
            {"title": f"Google: {query}", "body": "", "link": f"https://www.google.com/search?q={encoded_query}"},
            {"title": f"Wikipedia: {query}", "body": "", "link": f"https://en.wikipedia.org/wiki/Special:Search?search={encoded_query}"},
            {"title": f"Bing: {query}", "body": "", "link": f"https://www.bing.com/search?q={encoded_query}"}
        ]

def research_agent(query, context):
    system = "You are a research agent. Extract and present factual information clearly and concisely based only on the provided context."
    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response['message']['content']

def summarizer_agent(text):
    system = "You are a summarizer agent. Extract the 5 most important points. Return bullet points only."
    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': f"Summarize this:\n{text}"}
        ]
    )
    return response['message']['content']

def answer_agent(summary):
    system = "Format response into Definition, Key Points, and Conclusion."
    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': summary}
        ]
    )
    return response['message']['content']

def render_assistant_message(msg, sources=None):
    st.markdown(
        f"<p class='small-text'>{msg.get('steps', '')}</p>",
        unsafe_allow_html=True
    )
    st.markdown(f"""
    <div class="response-card">
    {msg["content"]}
    </div>
    """, unsafe_allow_html=True)

    if sources:
        st.markdown("Sources")
        for src in sources:
            if src.get("link"):
                st.markdown(f"- [{src['title']}]({src['link']})")
            else:
                st.markdown(f"- {src['title']}")

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            saved_sources = st.session_state.sources.get(i)
            render_assistant_message(msg, saved_sources)
        else:
            st.markdown(msg["content"])

def run_pipeline(query):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            sources = []
            process_steps = []

            with st.status("Processing", expanded=True) as status:

                if st.session_state.pdf_text:
                    status.update(label="Reading document", state="running")
                    context = st.session_state.pdf_text[:3000]
                    sources = [{"title": st.session_state.last_uploaded_file, "link": ""}]
                    process_steps.append("Document loaded")
                else:
                    status.update(label="Searching web", state="running")
                    web_results = search_agent(query)
                    context = " ".join([r["body"] for r in web_results])
                    sources = web_results
                    process_steps.append("Web data retrieved")

                status.update(label="Processing with AI", state="running")
                data = research_agent(query, context)
                process_steps.append("AI processed data")

                status.update(label="Summarizing", state="running")
                summary = summarizer_agent(data)
                process_steps.append("Summarization done")

                status.update(label="Generating answer", state="running")
                final = answer_agent(summary)
                process_steps.append("Final response generated")

                status.update(label="Done", state="complete")

            steps_text = " → ".join(process_steps)
            render_assistant_message({"content": final, "steps": steps_text}, sources)

            msg_index = len(st.session_state.messages)
            st.session_state.messages.append({
                "role": "assistant",
                "content": final,
                "steps": steps_text
            })
            st.session_state.sources[msg_index] = sources

        except Exception as e:
            st.error(f"Error: {str(e)}")

query = st.chat_input("Enter your query")

if st.session_state.rerun_query:
    rerun_q = st.session_state.rerun_query
    st.session_state.rerun_query = None
    run_pipeline(rerun_q)

elif query:
    run_pipeline(query)