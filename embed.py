# embed.py - Sentence transformer embedding service
# Uses jumpboot's JSONQueue for communication with Go

import jumpboot
from sentence_transformers import SentenceTransformer

# Global model instance
model = None
model_name = None

def load_model(name: str = "all-MiniLM-L6-v2"):
    """Load or switch embedding model."""
    global model, model_name
    if model_name != name:
        model = SentenceTransformer(name)
        model_name = name
    return {"status": "ok", "model": name, "dimension": model.get_sentence_embedding_dimension()}

def embed(texts: list) -> dict:
    """Generate embeddings for a list of texts."""
    if model is None:
        load_model()
    
    embeddings = model.encode(texts, convert_to_numpy=True)
    return {
        "embeddings": embeddings.tolist(),
        "model": model_name,
        "dimension": len(embeddings[0]) if len(embeddings) > 0 else 0
    }

def handle_command(cmd: dict) -> dict:
    """Route commands to appropriate handlers."""
    action = cmd.get("command", cmd.get("action", ""))
    data = cmd.get("data", cmd)
    
    if action == "load":
        return load_model(data.get("model", "all-MiniLM-L6-v2"))
    elif action == "embed":
        texts = data.get("texts", [])
        if isinstance(texts, str):
            texts = [texts]
        return embed(texts)
    elif action == "info":
        return {
            "model": model_name,
            "dimension": model.get_sentence_embedding_dimension() if model else None,
            "ready": model is not None
        }
    elif action == "exit":
        return {"status": "exiting"}
    else:
        return {"error": f"Unknown action: {action}"}

def main():
    """Main loop using jumpboot's JSONQueue."""
    # Pre-load default model
    load_model()
    
    # Create queue for communication with Go
    queue = jumpboot.JSONQueue(jumpboot.Pipe_in, jumpboot.Pipe_out)
    
    # Signal ready
    queue.put({"status": "ready", "model": model_name})
    
    while True:
        try:
            cmd = queue.get(block=True, timeout=1)
        except TimeoutError:
            continue
        except EOFError:
            break
        except Exception as e:
            continue
        
        if cmd is None:
            continue
            
        try:
            response = handle_command(cmd)
            queue.put(response)
            
            if cmd.get("command") == "exit" or cmd.get("action") == "exit":
                break
        except Exception as e:
            queue.put({"error": str(e)})

if __name__ == "__main__":
    main()
