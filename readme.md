
# Project Title

A Gradio based chat interface with ability to connect to alternative LLM providers like Openrouter and Mistral.

### Prerequisites

Before you begin, ensure you have met the following requirements:

- A suitable code editor (e.g., VSCode, PyCharm)
- Python installed on your machine (preferably Python 3.8 or above)
- Git (for cloning the repository)

### Installation

#### Copying the Repository

To copy the repository, run the following command in your terminal:

```bash
git clone https://github.com/emil2099/chat-gradio
```

#### Setting Up the Python Virtual Environment

Navigate to the project directory and set up a virtual environment:

```bash
cd your-project-directory
python -m venv venv
```

Activate the virtual environment:

- On Windows:
    ```bash
    .\venv\Scripts\activate
    ```

- On Unix or MacOS:
    ```bash
    source venv/bin/activate
    ```

Install the required dependencies using the `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Configuration

#### Updating API Providers List

To update the list of API providers, modify the `chat.py` script:

- Original line:
    ```python
    default_providers = ['openrouter', 'openai', 'mistral']
    ```

- To modify, simply add or remove the providers as per your requirement.

#### Creating the .env File

Create a `.env` file in the root directory of your project to securely store your API keys. Structure the file as follows:

```plaintext
OPENROUTER_API_KEY='your_openrouter_api_key_here'
OPENAI_API_KEY='your_openai_api_key_here'
MISTRAL_API_KEY='your_mistral_api_key_here'
```

Replace the placeholder API keys with your actual keys.

### Running the Application

With all configurations done, run the script with the following command:

```bash
python chat.py
```

### Accessing and using the chat

Access the chat on local URL: http://0.0.0.0:7860