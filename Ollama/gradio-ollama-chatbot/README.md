# Gradio Ollama Chatbot

This project integrates a Gradio interface with an Ollama chatbot running in a Docker container. It allows users to interact with the chatbot through a web interface.

## Project Structure

```
gradio-ollama-chatbot
├── src
│   ├── app.py          # Main entry point for the Gradio application
│   └── ollama_client.py # Client for connecting to the Ollama Docker instance
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd gradio-ollama-chatbot
   ```

2. **Install dependencies:**
   Make sure you have Python and pip installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Ollama Docker container:**
   Ensure that the Ollama Docker container is running. You can start it with:
   ```bash
   docker run -d --name ollama-container <ollama-image>
   ```

4. **Start the Gradio application:**
   Run the following command to start the Gradio interface:
   ```bash
   python src/app.py
   ```

5. **Access the application:**
   Open your web browser and go to `http://localhost:7860` to interact with the chatbot.

## Usage

Once the application is running, you can type your queries into the input box and receive responses from the Ollama chatbot. The Gradio interface provides a user-friendly way to communicate with the chatbot.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements for the project.