# Required imports
import streamlit as st
from anjibot_logging import append_to_sheet
from rag import handle_query


def main():
    st.title("Ask Anjibot 2.0")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.write_stream(handle_query(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})

        append_to_sheet(prompt, response)

if __name__ == "__main__":
    main()
