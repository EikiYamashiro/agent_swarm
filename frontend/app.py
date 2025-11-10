import streamlit as st
import requests
import os

from dotenv import load_dotenv
load_dotenv()


def send_message_to_backend(message: str, backend_url: str, user_id: str) -> dict:
    """Send a message+user_id to the backend /swarm endpoint.

    Returns the parsed JSON response as a dict. On error returns a dict with 'error'.
    """
    try:
        payload = {"message": message, "user_id": user_id}
        resp = requests.post(f"{backend_url.rstrip('/')}/swarm", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    st.set_page_config(page_title="CloudWalk Chat", page_icon="💬")
    st.title("CloudWalk Chat")

    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    default_user = os.environ.get("DEFAULT_USER_ID", "user123")

    # Allow user to set a user_id in the sidebar (persisted in session)
    if "user_id" not in st.session_state:
        st.session_state.user_id = default_user

    st.sidebar.header("Settings")
    st.session_state.user_id = st.sidebar.text_input("User ID", value=st.session_state.user_id)
    st.sidebar.markdown(f"**Backend:** {backend_url}")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me anything."}]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_input = st.chat_input(placeholder="Type a message and press Enter")
    if user_input:
        # show user's message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # call backend
        result = send_message_to_backend(user_input, backend_url=backend_url, user_id=st.session_state.user_id)

        if result.get("error"):
            assistant_text = f"[error] {result['error']}"
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            with st.chat_message("assistant"):
                st.write(assistant_text)
        else:
            # backend returns fields like: answer, sources, used_retrieval, tools_used
            answer = result.get("answer") or result.get("reply") or ""
            sources = result.get("sources") or []
            tools_used = result.get("tools_used")
            used_retrieval = result.get("used_retrieval")

            # display assistant answer
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

            # display metadata below the assistant message
            meta_lines = []
            if used_retrieval is not None:
                meta_lines.append(f"**used_retrieval:** {used_retrieval}")
            if tools_used:
                meta_lines.append(f"**tools_used:** {tools_used}")
            if sources:
                meta_lines.append("**sources:**")
                for s in sources:
                    # each source may be a dict with url/title or a string
                    if isinstance(s, dict):
                        url = s.get("url") or s.get("source") or ""
                        title = s.get("title") or s.get("name") or url
                        meta_lines.append(f"- [{title}]({url})")
                    else:
                        meta_lines.append(f"- {s}")

            if meta_lines:
                with st.expander("Response details"):
                    for line in meta_lines:
                        st.markdown(line)


if __name__ == "__main__":
    main()