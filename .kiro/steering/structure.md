# Project Structure

## Directory Organization

```
mealawe_automation/                 # Main Python package
├── __init__.py                    # Package initialization
├── config.py                     # Application configuration and settings
├── database/                     # Database layer
│   ├── __init__.py
│   ├── connection.py            # MongoDB connection management
│   └── operations.py            # Database operations and repositories
├── models/                      # Data models and schemas
│   ├── __init__.py
│   └── core.py                 # Pydantic models for orders, kitchens, verification results
└── preprocessing/               # ETL and data processing
    ├── __init__.py
    └── etl_pipeline.py         # Main ETL pipeline for 4,405 order dataset

tests/                           # Test suite
├── __init__.py
└── test_properties.py          # Property-based tests (currently empty)

.kiro/                          # Kiro IDE configuration
├── specs/                      # Feature specifications
│   └── mealawe-kitchen-automation/
│       ├── requirements.md     # Requirements document
│       ├── design.md          # Design document with architecture
│       └── tasks.md           # Implementation task list
└── steering/                   # AI assistant guidance documents
    ├── product.md             # Product overview
    ├── tech.md               # Technology stack and commands
    └── structure.md          # This file - project organization

# Data files
mealaweProdDatabase.foodorders.xlsx  # Source dataset (4,405 orders)
dataset_analysis_report.txt          # Dataset structure analysis
analyze_dataset.py                   # Dataset analysis script
```

## Code Organization Patterns

### Models (`mealawe_automation/models/`)
- **Pydantic models** for type safety and validation
- **MongoDB integration** with PyObjectId for document IDs
- **Hierarchical structure**: Order → OrderItem → VerificationAttempt
- **Embedded documents** for SOP ground truth and workflow state

### Database Layer (`mealawe_automation/database/`)
- **Async operations** using Motor (AsyncIOMotorCollection)
- **Repository pattern** with BaseRepository for common operations
- **Connection management** with proper error handling
- **Collections**: orders, kitchens, verification_states, annotations, analytics

### Configuration (`mealawe_automation/config.py`)
- **Pydantic BaseSettings** for environment variable management
- **Performance targets** as configuration constants
- **Model paths** and thresholds centrally managed
- **Environment-specific** settings with .env file support

### ETL Pipeline (`mealawe_automation/preprocessing/`)
- **Modular design**: TextNormalizer, OutlierDetector, SOPGroundTruthGenerator
- **Statistics tracking** throughout the pipeline
- **Business rule validation** for data quality
- **Pandas-based** transformation with error handling

## Naming Conventions

### Files and Directories
- **Snake_case** for Python files and directories
- **Kebab-case** for spec directories (e.g., `mealawe-kitchen-automation`)
- **Descriptive names** that indicate purpose (e.g., `etl_pipeline.py`, `core.py`)

### Code Conventions
- **Class names**: PascalCase (e.g., `TrayVerificationResult`, `MealaweETLPipeline`)
- **Function/method names**: snake_case (e.g., `verify_tray_structure`, `extract_order_basics`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_PROCESSING_TIME_SECONDS`, `ITEM_MAPPINGS`)
- **Private methods**: Leading underscore (e.g., `_extract_order_basics`, `_detect_order_outliers`)

### Database Conventions
- **Collection names**: Lowercase plural (e.g., `orders`, `kitchens`, `analytics`)
- **Field names**: snake_case matching Python model fields
- **Document IDs**: Use PyObjectId with `_id` field alias
- **Embedded documents**: Nested dictionaries for complex data (e.g., `workflow_state`, `sop_ground_truth`)

## Architecture Layers

### 1. Data Layer
- **MongoDB collections** with proper indexing
- **Async operations** for performance
- **Change Streams** for real-time updates

### 2. Business Logic Layer
- **Pydantic models** for data validation
- **Repository pattern** for data access
- **Service classes** for business operations

### 3. AI/ML Layer
- **YOLOv8-seg** for computer vision
- **FoodLMM** for reasoning and quality assessment
- **LangGraph** for workflow orchestration

### 4. API Layer
- **FastAPI** for REST endpoints
- **WebSocket/SSE** for real-time features
- **Prometheus metrics** for monitoring

### 5. Frontend Layer
- **React components** for dashboard
- **Real-time updates** via WebSocket/SSE
- **Analytics visualization** for kitchen performance

## Import Patterns

```python
# Relative imports within package
from ..models.core import Order, Kitchen
from ..database.operations import DatabaseOperations
from .etl_pipeline import MealaweETLPipeline

# External dependencies
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic import BaseModel, Field
```

## Testing Structure

- **Property-based tests** in `tests/test_properties.py` using Hypothesis
- **Unit tests** for individual components
- **Integration tests** for database operations
- **End-to-end tests** for complete verification workflows