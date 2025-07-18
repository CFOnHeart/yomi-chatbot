## Setup environment
1. Run with uv
    ```powershell
    uv venv
    .venv\Scripts\activate
    uv sync # install packages
    ```
2. rename `.env.template` to `.env` and set up correct api key 

## Configuration
You can configure the settings in the settings.py file to specify which the model, embedding, db you use. 
By default, I use the `azure openai` model, `azure openai embedding`, `database under the database folder named chat_history.db, documents.db`

## Simple Start 
```powershell
uv run quick_start.py
```

## start with your RAG
```powershell
uv run quick_start.py # select 2 to upload your knowledge file, I upload ReAct.pdf by default, of course you could upload your own file
```
![](images/upload_pdf.png)
```
uv run quick_start.py # select 1 to chat with the bot
input: what is the chain-of-thought
```
![](images/Rag_chat_output.png)
