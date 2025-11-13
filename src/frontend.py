# # frontend.py (New file for Streamlit frontend)
# import streamlit as st
# import requests

# st.title("RAG Query Interface")
# st.markdown("Enter a query to retrieve and process user profile and posts for personality analysis.")

# query = st.text_input("Enter your query:", placeholder="e.g., Tell me about Al Amin")

# if st.button("Submit Query"):
#     if query:
#         with st.spinner("Processing..."):
#             try:
#                 response = requests.post("http://localhost:8000/query", json={"query": query})
#                 if response.status_code == 200:
#                     result = response.json().get("result", "No result returned.")
#                     st.success("Query processed successfully!")
#                     st.text_area("Result:", value=result, height=400)
#                 else:
#                     st.error(f"Error: {response.status_code} - {response.text}")
#             except requests.exceptions.RequestException as e:
#                 st.error(f"Failed to connect to API: {str(e)}. Make sure the FastAPI server is running.")
#     else:
#         st.warning("Please enter a query.")


# frontend.py
import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Personality RAG",
    page_icon="magnifying_glass_tilted_left",
    layout="centered"
)

# Title
st.title("Personality Profile Analyzer")
st.caption("Enter a user query to retrieve profile and posts from RAG system")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about a user (e.g., 'Tell me about Al Amin')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                response = requests.post(
                    "http://localhost:8000/query",
                    json={"query": prompt}
                )
                if response.status_code == 200:
                    result = response.json().get("result", "No result returned.")
                    st.markdown(result)
                elif response.status_code == 429:
                    st.error("Gemini API quota exceeded (10/min). Please wait 1 minute.")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except requests.ConnectionError:
                st.error("Cannot connect to backend. Is `uvicorn app:app` running?")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    # Add assistant response to history
    # (We don't save full response here to avoid duplication — already shown above)
    # But if you want full persistence, capture it:
    # st.session_state.messages.append({"role": "assistant", "content": result})y