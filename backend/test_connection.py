from core.config import get_settings
from core.database import init_database
from core.database import Prompt
from sqlalchemy import inspect

def test_connection():
    print("=== Testing config.py ↔ database.py ===")
    
    # Load settings
    settings = get_settings()
    print(f"[1/3] Config loaded. Database URL: {settings.database_url}")
    
    # Initialize database
    db_manager = init_database(settings)
    print("[2/3] Database initialized successfully!")
    
    # Verify tables exist
    with db_manager.get_session() as session:
        inspector = inspect(db_manager._engine)
        table_names = inspector.get_table_names()
        print(f"[3/3] Tables detected: {table_names}")

def test_model_operations():
    settings = get_settings()
    db_manager = init_database(settings)
    
    # Force clean slate
    db_manager.drop_tables()
    db_manager.create_tables()
    
    # Test with minimal required fields only
    with db_manager.get_session() as session:
        prompt = Prompt(
            content="What is AI?",  # Only absolutely required fields
            name="Test Prompt"      # Add other fields one-by-one after success
        )
        session.add(prompt)
        session.commit()
        print("✅ Basic prompt creation succeeded!")

if __name__ == "__main__":
    test_connection()
    test_model_operations()  # Add this call