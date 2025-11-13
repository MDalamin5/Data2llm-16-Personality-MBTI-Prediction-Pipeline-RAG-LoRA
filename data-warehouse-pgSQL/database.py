# database.py (updated)
from sqlmodel import create_engine, Session, SQLModel
from sqlalchemy.engine import Engine
from sqlalchemy import event
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus  # Import for URL encoding

load_dotenv()  # Load .env file

# Load separate env vars
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")  # Default fallback
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "jogajog_sales_connections_db")

# Construct DATABASE_URL with proper encoding for special chars in password
encoded_password = quote_plus(DB_PASSWORD)
DATABASE_URL = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


engine: Engine = create_engine(DATABASE_URL, echo=True)  # echo=True for debug logs

# Optional: Enforce foreign key constraints (useful for SQLite, but good practice)
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if "sqlite" in DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

def get_session():
    with Session(engine) as session:
        yield session

