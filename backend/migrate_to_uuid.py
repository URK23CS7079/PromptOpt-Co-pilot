from sqlalchemy import inspect
from core.config import get_settings
from core.database import DatabaseManager

def migrate_database():
    settings = get_settings()
    db_manager = DatabaseManager(settings.database_path)
    
    # Get existing table info (if needed for data migration)
    inspector = inspect(db_manager._engine)
    existing_tables = inspector.get_table_names()
    
    # Drop and recreate with new schema
    db_manager.drop_tables()
    db_manager.create_tables()
    print("Database migrated successfully")

if __name__ == "__main__":
    migrate_database()