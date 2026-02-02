"""
Configuration settings for Mealawe Kitchen Automation System
"""
from pydantic import BaseSettings, Field
from typing import Dict, Any, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Info
    app_name: str = "Mealawe Kitchen Automation"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database Configuration
    mongodb_url: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    database_name: str = Field(default="mealawe_automation", env="DATABASE_NAME")
    
    # Performance Thresholds (from requirements)
    max_processing_time_seconds: float = Field(default=3.0, env="MAX_PROCESSING_TIME_SECONDS")
    compartment_detection_map_threshold: float = Field(default=0.94, env="COMPARTMENT_DETECTION_MAP_THRESHOLD")  # 94-98%
    quantity_verification_accuracy_threshold: float = Field(default=0.88, env="QUANTITY_VERIFICATION_ACCURACY_THRESHOLD")  # 88-92%
    quality_assessment_accuracy_threshold: float = Field(default=0.75, env="QUALITY_ASSESSMENT_ACCURACY_THRESHOLD")  # 75-85%
    
    # AI Model Paths
    yolo_model_path: str = Field(default="models/yolov8_seg.pt", env="YOLO_MODEL_PATH")
    foodlmm_model_path: str = Field(default="models/foodlmm.pt", env="FOODLMM_MODEL_PATH")
    
    # Image Processing
    target_image_size: tuple = (640, 640)
    max_image_size_mb: float = Field(default=10.0, env="MAX_IMAGE_SIZE_MB")
    supported_image_formats: list = [".jpg", ".jpeg", ".png", ".bmp"]
    
    # Storage Paths
    base_storage_path: str = Field(default="data", env="BASE_STORAGE_PATH")
    raw_images_path: str = Field(default="data/images/raw", env="RAW_IMAGES_PATH")
    processed_images_path: str = Field(default="data/images/processed", env="PROCESSED_IMAGES_PATH")
    model_storage_path: str = Field(default="data/models", env="MODEL_STORAGE_PATH")
    
    # ETL Pipeline Configuration
    etl_batch_size: int = Field(default=100, env="ETL_BATCH_SIZE")
    etl_max_errors: int = Field(default=50, env="ETL_MAX_ERRORS")
    
    # Business Rules (from ETL pipeline)
    max_item_quantities: Dict[str, int] = {
        'roti': 10,
        'rice': 5,
        'dal': 3,
        'sabzi': 3,
        'default': 15
    }
    max_addons_per_order: int = 4
    
    # Outlier Detection Thresholds
    outlier_quantity_multiplier: float = Field(default=2.0, env="OUTLIER_QUANTITY_MULTIPLIER")
    outlier_detection_enabled: bool = Field(default=True, env="OUTLIER_DETECTION_ENABLED")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    cors_origins: list = ["http://localhost:3000", "http://localhost:8080"]
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Monitoring
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")
    
    # Real-time Features
    websocket_enabled: bool = Field(default=True, env="WEBSOCKET_ENABLED")
    sse_enabled: bool = Field(default=True, env="SSE_ENABLED")
    
    # Training Configuration
    training_data_split: Dict[str, float] = {
        'train': 0.7,
        'validation': 0.2,
        'test': 0.1
    }
    seed_annotation_count: int = Field(default=100, env="SEED_ANNOTATION_COUNT")
    auto_annotation_batch_size: int = Field(default=50, env="AUTO_ANNOTATION_BATCH_SIZE")
    
    # CVAT Integration
    cvat_enabled: bool = Field(default=False, env="CVAT_ENABLED")
    cvat_url: Optional[str] = Field(default=None, env="CVAT_URL")
    cvat_username: Optional[str] = Field(default=None, env="CVAT_USERNAME")
    cvat_password: Optional[str] = Field(default=None, env="CVAT_PASSWORD")
    
    # LangGraph Configuration
    langgraph_enabled: bool = Field(default=True, env="LANGGRAPH_ENABLED")
    max_correction_cycles: int = Field(default=3, env="MAX_CORRECTION_CYCLES")
    workflow_timeout_seconds: int = Field(default=30, env="WORKFLOW_TIMEOUT_SECONDS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_database_url(self) -> str:
        """Get complete database URL"""
        return f"{self.mongodb_url}/{self.database_name}"
    
    def get_storage_paths(self) -> Dict[str, Path]:
        """Get all storage paths as Path objects"""
        base_path = Path(self.base_storage_path)
        return {
            'base': base_path,
            'raw_images': Path(self.raw_images_path),
            'processed_images': Path(self.processed_images_path),
            'models': Path(self.model_storage_path),
            'logs': base_path / 'logs',
            'temp': base_path / 'temp'
        }
    
    def create_storage_directories(self):
        """Create all required storage directories"""
        paths = self.get_storage_paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get AI model configuration"""
        return {
            'yolo': {
                'model_path': self.yolo_model_path,
                'target_map': self.compartment_detection_map_threshold,
                'input_size': self.target_image_size
            },
            'foodlmm': {
                'model_path': self.foodlmm_model_path,
                'target_accuracy': self.quality_assessment_accuracy_threshold
            },
            'processing': {
                'max_time_seconds': self.max_processing_time_seconds,
                'target_image_size': self.target_image_size
            }
        }
    
    def get_business_rules(self) -> Dict[str, Any]:
        """Get business rules configuration"""
        return {
            'max_quantities': self.max_item_quantities,
            'max_addons': self.max_addons_per_order,
            'outlier_detection': {
                'enabled': self.outlier_detection_enabled,
                'multiplier': self.outlier_quantity_multiplier
            },
            'correction_cycles': {
                'max_cycles': self.max_correction_cycles,
                'timeout_seconds': self.workflow_timeout_seconds
            }
        }


# Global settings instance
settings = Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    mongodb_url: str = "mongodb://localhost:27017"
    cors_origins: list = ["*"]  # Allow all origins in development


class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    log_level: str = "INFO"
    api_workers: int = 4
    cors_origins: list = []  # Restrict origins in production


class TestingSettings(Settings):
    """Testing environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    database_name: str = "mealawe_automation_test"
    mongodb_url: str = "mongodb://localhost:27017"
    etl_batch_size: int = 10  # Smaller batches for testing


def get_settings() -> Settings:
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Export the appropriate settings
settings = get_settings()

# Create storage directories on import
settings.create_storage_directories()