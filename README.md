# GenAI Groq Assistant

A chat UI built with Streamlit and Groq. Ask questions by typing or recording them with your microphone; voice recordings are transcribed with Groq Whisper.

## Run locally

1. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root and add your Groq API key:

   ```text
   GROQ_API_KEY=your-groq-api-key
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy with Streamlit Community Cloud

1. Push this project to a GitHub repository. Do not commit `.env`; it is ignored by Git.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from that repository and select `app.py` as the entry point.
3. In the app's **Settings → Secrets**, add:

   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   ```

4. Save the secret and deploy the app.