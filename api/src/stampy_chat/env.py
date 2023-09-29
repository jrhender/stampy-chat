import os
import openai
import pinecone

if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()
else:
    print("'api/.env' not found. Rename the 'api/.env.example' file and fill in values.")

FLASK_PORT = int(os.environ.get('FLASK_PORT', '3001'))

LOG_LEVEL           = os.environ.get("LOG_LEVEL", "WARNING").upper()
DISCORD_LOG_LEVEL   = os.environ.get("DISCORD_LOG_LEVEL", "WARNING").upper()
DISCORD_LOGGING_URL = os.environ.get('LOGGING_URL')

### OpenAI ###
OPENAI_API_KEY   = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY # non-optional

### Pinecone ###
PINECONE_API_KEY     = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT', "us-east1-gcp")
PINECONE_INDEX_NAME  = os.environ.get("PINECONE_INDEX_NAME", "alignment-search")
PINECONE_INDEX       = None
PINECONE_NAMESPACE   = os.environ.get("PINECONE_NAMESPACE", "alignment-search")  # "normal" or "finetuned" for the new index, "alignment-search" for the old one

# Only init pinecone if we have an env value for it.
if PINECONE_API_KEY:
    pinecone.init(
        api_key = PINECONE_API_KEY,
        environment = PINECONE_ENVIRONMENT,
    )

    PINECONE_INDEX = pinecone.Index(index_name=PINECONE_INDEX_NAME)

### MySQL ###
user = os.environ.get("CHAT_DB_USER", "user")
password = os.environ.get("CHAT_DB_PASSWORD", "we all live in a yellow submarine")
host = os.environ.get("CHAT_DB_HOST", "127.0.0.1")
port = os.environ.get("CHAT_DB_PORT", "3306")
db_name = os.environ.get("CHAT_DB_NAME", "stampy_chat")
DB_CONNECTION_URI = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}"