MODEL_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
MODEL_ID = "Llama-3.2-3B-Instruct-Q4_K_L.gguf"

DEVICE = "mps"  # "cpu" or "cuda" or "mps"

# Default database credentials
DB_USER = "admin"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "tiny-graphrag"
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
