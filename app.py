import logging
import os

import streamlit as st
from dotenv import load_dotenv
from groq import APIError, Groq, RateLimitError

load_dotenv()

MODEL = "openai/gpt-oss-20b"
TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
SYSTEM_MESSAGE = "You are a helpful assistant. Answer clearly and concisely."

st.set_page_config(page_title="Groq Assistant", page_icon="💬")
st.title("Groq Assistant")
st.caption("Ask a question and get an answer powered by Groq.")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError):
        api_key = None

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("About")
    st.write(f"Model: `{MODEL}`")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.warning(
        "Set `GROQ_API_KEY` in your `.env` file or Streamlit secrets to start chatting."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

submission = st.chat_input(
    "Ask anything",
    accept_audio=True,
    audio_sample_rate=16000,
)

if isinstance(submission, str):
    prompt = submission.strip()
    audio_file = None
elif submission is not None:
    prompt = submission.text.strip()
    audio_file = submission.audio
else:
    prompt = None
    audio_file = None

if audio_file is not None:
    typed_prompt = prompt
    prompt = None
    if not api_key:
        st.error("GROQ_API_KEY is required to transcribe a recording.")
        st.stop()

    try:
        with st.spinner("Transcribing your question..."):
            transcription = Groq(
                api_key=api_key,
                max_retries=0,
            ).audio.transcriptions.create(
                file=audio_file,
                model=TRANSCRIPTION_MODEL,
                response_format="json",
            )
        transcribed_prompt = transcription.text.strip()
        if transcribed_prompt:
            prompt = " ".join(
                part for part in (typed_prompt, transcribed_prompt) if part
            )
        else:
            prompt = None
            st.warning("No speech was detected. Please record your question again.")
    except RateLimitError as error:
        retry_after = error.response.headers.get("retry-after")
        message = "Groq usage limit reached. Please try again later."
        if retry_after:
            message = f"Groq usage limit reached. Try again after {retry_after} seconds."
        st.warning(message)
    except APIError:
        logging.exception("Groq audio transcription failed")
        st.error("Could not transcribe the recording. Check your API key and try again.")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        st.error("GROQ_API_KEY is required to send a message.")
        st.stop()

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = Groq(api_key=api_key, max_retries=0).chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        *st.session_state.messages,
                    ],
                    temperature=0.2,
                    max_completion_tokens=512,
                )
            answer = response.choices[0].message.content
            if answer:
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            else:
                st.info("Groq did not return an answer. Please try another question.")
        except RateLimitError as error:
            retry_after = error.response.headers.get("retry-after")
            message = "Groq usage limit reached. Please try again later."
            if retry_after:
                message = f"Groq usage limit reached. Try again after {retry_after} seconds."
            st.warning(message)
        except APIError:
            logging.exception("Groq chat completion failed")
            st.error("The request to Groq failed. Check your API key and try again.")
