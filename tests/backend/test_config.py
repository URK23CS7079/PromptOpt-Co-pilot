from backend.core.config import get_settings, validate_settings

def test_config_loading():
    settings = get_settings()
    assert settings.environment in ["development", "production", "testing"]
    
def test_validation():
    report = validate_settings()
    assert report["valid"] is True