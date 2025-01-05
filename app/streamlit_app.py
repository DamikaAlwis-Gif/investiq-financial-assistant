from graph.workflow import create_workflow
import streamlit as st
import uuid
from dotenv import load_dotenv
from graph.errors.finance_exceptions import FinanceError
from utils.process_json_files import get_json_files_list, load_file_content_to_vector_store, load_processed_files, save_processed_file
from config.constants import JSON_FILES_DIRECTORY, PROCESSED_FILES_PATH
from config.constants import BOT_NAME, HEADER_TEXT, SUB_HEADER_TEXT
import logging
import os 
def main():

    # Load environment variables from .env file
    load_dotenv()

    json_files = get_json_files_list(JSON_FILES_DIRECTORY)
    if not json_files:
        logging.info(f"No JSON files found in {JSON_FILES_DIRECTORY}.")

    # get the list of processed files
    processed_files = load_processed_files(PROCESSED_FILES_PATH)

    for json_file in json_files:
        if json_file not in processed_files:
            load_file_content_to_vector_store(json_file)
            save_processed_file(PROCESSED_FILES_PATH, json_file)


    app = create_workflow()
    # Set the page configuration
     
    st.set_page_config(
        page_title=BOT_NAME,
        page_icon="📈",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.header(HEADER_TEXT)
    st.subheader(SUB_HEADER_TEXT)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Sidebar with New Chat button and session ID
    with st.sidebar:
        session_id_container = st.empty()
        # Display the session ID
        session_id_container.write(
            f"**Session ID:** {st.session_state['session_id']}")

        # Add a button to start a new chat
        if st.button("New Chat"):
            # Reset chat history and generate a new session ID
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            session_id_container.write(
                f"**Session ID:** {st.session_state['session_id']}")
            st.success("New chat started!")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Message {BOT_NAME}..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            typing_indicator = st.empty()
            typing_indicator.write(f"{BOT_NAME} is processing your request...")

            # Configuration for app invocation
            config = {
                "configurable": {
                                    "thread_id": st.session_state.session_id
                                }
                    }
            
            response = app.invoke({"input": prompt}, config=config)
            
            answer = response["messages"][-1].content
           
            typing_indicator.empty()  # Remove the typing indicator when done

            with st.chat_message("ai"):
                # Display the AI's answer
                st.markdown(answer)

            # Add AI response to chat history
            st.session_state.messages.append({"role": "ai", "content": answer})
            
        except FinanceError as e:
            typing_indicator.empty()
            with st.chat_message("ai"):
                answer = e.chat_message()
                st.markdown(answer)
            st.session_state.messages.append({"role": "ai", "content": answer})
               
        except Exception as e:
            print(e)
            typing_indicator.empty()  # Ensure the typing indicator is cleared
            answer = "An error occurred while processing your request."
            st.error(answer)
            st.session_state.messages.append({"role": "ai", "content": answer})

            st.sidebar.write("Error details:", str(e))


if __name__ == "__main__":
    main()
