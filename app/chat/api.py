import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from app.chat.chat import ChatService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Persist every prompt/response pair to JSONL for later analysis/reports.
HISTORY_PATH = Path(os.getenv("CHAT_HISTORY_PATH", ".output/chat_history.jsonl"))


def _log_interaction(request: "ChatRequest", response: str) -> None:
    """Appends one prompt/response record to the JSONL history file."""
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": request.prompt,
            "response": response,
            "params": {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_k": request.top_k,
            },
        }
        with HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # Persistence is non-critical; never break the response over a log failure.
        logger.warning(f"Failed to persist chat history: {e}")

chat_service = ChatService(
    checkpoint_path=Path(os.getenv("CHECKPOINT_PATH", ".output/checkpoint.pt"))
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and tokenizer on startup
    logger.info("Initializing model and tokenizer for API service...")
    if not chat_service._ensure_checkpoint_local():
        logger.error("Failed to ensure local checkpoint. Make sure checkpoint.pt exists or S3 credentials are valid.")
    else:
        try:
            chat_service._load_tokenizer()
            chat_service._load_model()
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    yield
    # Clean up on shutdown
    logger.info("Shutting down API service.")

app = FastAPI(
    title="tfm-slm API Server",
    description="Inference API for the Hybrid Transformer-GRU Small Language Model",
    version="0.1.0",
    lifespan=lifespan
)

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def generate_chat_response(request: ChatRequest):
    if chat_service.model is None or chat_service.tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs for initialization errors."
        )
    
    try:
        # Formulate standard assistant prompt format
        full_prompt = f"User: {request.prompt}\nAssistant:"
        response_text = chat_service._generate_response(
            prompt=full_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )
        _log_interaction(request, response_text)
        return ChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history(limit: int = 100):
    """Returns the most recent persisted prompt/response interactions."""
    if not HISTORY_PATH.exists():
        return {"count": 0, "interactions": []}
    with HISTORY_PATH.open(encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    records = [json.loads(line) for line in lines[-limit:]]
    return {"count": len(records), "interactions": records}


@app.get("/health")
async def health_check():
    status = "healthy" if chat_service.model is not None else "loading_or_error"
    return {"status": status}

@app.get("/", response_class=HTMLResponse)
async def root_web_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>tfm-slm Chatbot</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-color: #0b0f19;
                --panel-bg: rgba(17, 24, 39, 0.7);
                --accent-color: #4f46e5;
                --accent-hover: #4338ca;
                --text-main: #f3f4f6;
                --text-muted: #9ca3af;
                --chat-user-bg: #4f46e5;
                --chat-assistant-bg: #1f2937;
                --glass-border: rgba(255, 255, 255, 0.08);
            }
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg-color);
                color: var(--text-main);
                display: flex;
                flex-direction: column;
                height: 100vh;
                background-image: 
                    radial-gradient(at 0% 0%, rgba(79, 70, 229, 0.15) 0px, transparent 50%),
                    radial-gradient(at 100% 100%, rgba(99, 102, 241, 0.1) 0px, transparent 50%);
            }
            header {
                padding: 1.5rem;
                background-color: var(--panel-bg);
                backdrop-filter: blur(12px);
                border-bottom: 1px solid var(--glass-border);
                display: flex;
                justify-content: space-between;
                align-items: center;
                z-index: 10;
            }
            .header-title h1 {
                font-size: 1.25rem;
                font-weight: 600;
                letter-spacing: -0.025em;
                background: linear-gradient(to right, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .header-title p {
                font-size: 0.75rem;
                color: var(--text-muted);
            }
            .status-badge {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.8rem;
                background: rgba(16, 185, 129, 0.1);
                color: #10b981;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                border: 1px solid rgba(16, 185, 129, 0.2);
            }
            .status-dot {
                width: 8px;
                height: 8px;
                background-color: #10b981;
                border-radius: 50%;
                box-shadow: 0 0 8px #10b981;
            }
            main {
                flex: 1;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                max-width: 900px;
                width: 100%;
                margin: 0 auto;
                padding: 1.5rem;
                overflow: hidden;
            }
            .chat-container {
                flex: 1;
                overflow-y: auto;
                padding-right: 0.5rem;
                margin-bottom: 1.5rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            .chat-container::-webkit-scrollbar {
                width: 6px;
            }
            .chat-container::-webkit-scrollbar-track {
                background: transparent;
            }
            .chat-container::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
            }
            .message-row {
                display: flex;
                width: 100%;
                animation: fadeIn 0.3s ease-out forwards;
            }
            .message-row.user {
                justify-content: flex-end;
            }
            .message-row.assistant {
                justify-content: flex-start;
            }
            .message-bubble {
                max-width: 75%;
                padding: 0.85rem 1.25rem;
                border-radius: 16px;
                font-size: 0.95rem;
                line-height: 1.5;
            }
            .message-row.user .message-bubble {
                background-color: var(--chat-user-bg);
                color: #ffffff;
                border-bottom-right-radius: 4px;
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
            }
            .message-row.assistant .message-bubble {
                background-color: var(--chat-assistant-bg);
                color: var(--text-main);
                border-bottom-left-radius: 4px;
                border: 1px solid var(--glass-border);
            }
            .message-time {
                font-size: 0.7rem;
                color: var(--text-muted);
                margin-top: 0.4rem;
                text-align: right;
            }
            .message-row.assistant .message-time {
                text-align: left;
            }
            .input-panel {
                background-color: var(--panel-bg);
                backdrop-filter: blur(12px);
                border: 1px solid var(--glass-border);
                border-radius: 24px;
                padding: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            }
            .input-panel input {
                flex: 1;
                background: transparent;
                border: none;
                outline: none;
                color: var(--text-main);
                font-family: inherit;
                font-size: 0.95rem;
                padding: 0.75rem 1rem;
            }
            .input-panel input::placeholder {
                color: var(--text-muted);
            }
            .send-btn {
                background-color: var(--accent-color);
                color: white;
                border: none;
                outline: none;
                border-radius: 18px;
                padding: 0.75rem 1.5rem;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .send-btn:hover {
                background-color: var(--accent-hover);
                transform: translateY(-1px);
            }
            .send-btn:active {
                transform: translateY(0);
            }
            .send-btn:disabled {
                background-color: var(--chat-assistant-bg);
                color: var(--text-muted);
                cursor: not-allowed;
                transform: none;
            }
            .typing-indicator {
                display: flex;
                align-items: center;
                gap: 4px;
                padding: 0.5rem 0.75rem;
            }
            .typing-dot {
                width: 6px;
                height: 6px;
                background-color: var(--text-muted);
                border-radius: 50%;
                animation: typing 1.4s infinite ease-in-out;
            }
            .typing-dot:nth-child(2) { animation-delay: 0.2s; }
            .typing-dot:nth-child(3) { animation-delay: 0.4s; }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes typing {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-4px); }
            }
        </style>
    </head>
    <body>
        <header>
            <div class="header-title">
                <h1>tfm-slm</h1>
                <p>Modelo Híbrido Transformer-GRU (124M Params)</p>
            </div>
            <div class="status-badge" id="status-container">
                <div class="status-dot"></div>
                <span id="status-text">Conectado</span>
            </div>
        </header>
        <main>
            <div class="chat-container" id="chat-container">
                <div class="message-row assistant">
                    <div>
                        <div class="message-bubble">
                            ¡Hola! Soy el asistente virtual basado en tfm-slm. ¿En qué puedo ayudarte hoy?
                        </div>
                        <div class="message-time">Recién</div>
                    </div>
                </div>
            </div>
            <div class="input-panel">
                <input type="text" id="user-input" placeholder="Pregunta algo al chatbot..." autocomplete="off">
                <button class="send-btn" id="send-button">Enviar</button>
            </div>
        </main>
        <script>
            const chatContainer = document.getElementById('chat-container');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');

            function getFormattedTime() {
                const now = new Date();
                return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            }

            function appendMessage(text, isUser) {
                const row = document.createElement('div');
                row.className = `message-row ${isUser ? 'user' : 'assistant'}`;
                
                const wrapper = document.createElement('div');
                
                const bubble = document.createElement('div');
                bubble.className = 'message-bubble';
                bubble.innerText = text;
                
                const time = document.createElement('div');
                time.className = 'message-time';
                time.innerText = getFormattedTime();
                
                wrapper.appendChild(bubble);
                wrapper.appendChild(time);
                row.appendChild(wrapper);
                chatContainer.appendChild(row);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                
                return row;
            }

            function appendTypingIndicator() {
                const row = document.createElement('div');
                row.className = 'message-row assistant';
                row.id = 'typing-indicator-row';
                
                const wrapper = document.createElement('div');
                
                const bubble = document.createElement('div');
                bubble.className = 'message-bubble typing-indicator';
                
                for (let i = 0; i < 3; i++) {
                    const dot = document.createElement('div');
                    dot.className = 'typing-dot';
                    bubble.appendChild(dot);
                }
                
                wrapper.appendChild(bubble);
                row.appendChild(wrapper);
                chatContainer.appendChild(row);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                return row;
            }

            function removeTypingIndicator() {
                const indicator = document.getElementById('typing-indicator-row');
                if (indicator) {
                    indicator.remove();
                }
            }

            async function sendMessage() {
                const text = userInput.value.trim();
                if (!text) return;
                
                userInput.value = '';
                userInput.disabled = true;
                sendButton.disabled = true;
                
                appendMessage(text, true);
                appendTypingIndicator();
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ prompt: text }),
                    });
                    
                    removeTypingIndicator();
                    
                    if (response.ok) {
                        const data = await response.json();
                        appendMessage(data.response, false);
                    } else {
                        appendMessage('Error: No se pudo obtener respuesta del servidor.', false);
                    }
                } catch (error) {
                    removeTypingIndicator();
                    appendMessage('Error de conexión con el servidor.', false);
                } finally {
                    userInput.disabled = false;
                    sendButton.disabled = false;
                    userInput.focus();
                }
            }

            sendButton.addEventListener('click', sendMessage);
            userInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });

            // Check health on load
            fetch('/health')
                .then(r => r.json())
                .then(data => {
                    const statusText = document.getElementById('status-text');
                    const statusDot = document.querySelector('.status-dot');
                    if (data.status === 'healthy') {
                        statusText.innerText = 'Listo';
                    } else {
                        statusText.innerText = 'Cargando Modelo...';
                        statusDot.style.backgroundColor = '#fbbf24';
                        statusDot.style.boxShadow = '0 0 8px #fbbf24';
                    }
                })
                .catch(() => {
                    const statusText = document.getElementById('status-text');
                    const statusDot = document.querySelector('.status-dot');
                    statusText.innerText = 'Desconectado';
                    statusDot.style.backgroundColor = '#ef4444';
                    statusDot.style.boxShadow = '0 0 8px #ef4444';
                });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.chat.api:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
