import pytest
from core.config import Settings
from core.database import DatabaseManager, Prompt

@pytest.fixture
def db_manager(tmp_path):
    """Isolated database for testing"""
    db_path = tmp_path / "test.db"
    settings = Settings(database_url=f"sqlite:///{db_path}")
    manager = DatabaseManager(settings.database_url)
    manager.create_tables()
    yield manager
    manager.close()

def test_table_creation(db_manager):
    """Verify tables exist with correct columns"""
    with db_manager.get_session() as session:
        inspector = inspect(db_manager._engine)
        assert "prompts" in inspector.get_table_names()
        columns = {c["name"] for c in inspector.get_columns("prompts")}
        expected = {"id", "content", "created_at", "updated_at", "tags", "user_id"}
        assert columns.issuperset(expected)

def test_prompt_insert(db_manager):
    """Test basic CRUD operation"""
    with db_manager.get_session() as session:
        prompt = Prompt(content="Test prompt", user_id="user123")
        session.add(prompt)
        session.commit()
        
        saved_prompt = session.query(Prompt).first()
        assert saved_prompt.content == "Test prompt"
        assert saved_prompt.user_id == "user123"