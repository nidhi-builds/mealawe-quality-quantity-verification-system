# Technology Stack

## Core Technologies

### Backend
- **Python**: Primary backend language for AI/ML components
- **FastAPI**: API framework for model serving and backend services
- **Motor**: Async MongoDB driver for Python
- **Pydantic**: Data validation and settings management

### Frontend
- **React**: Dashboard and real-time UI components
- **Node.js**: Real-time features and WebSocket handling

### Database
- **MongoDB**: Primary database with Change Streams for real-time updates
- **Collections**: orders, kitchens, verification_states, annotations, analytics

### AI/ML Stack
- **YOLOv8-seg**: Compartment detection and segmentation (Ultralytics)
- **FoodLMM**: Food-specific reasoning and quality assessment
- **CVAT + Nuclio**: Annotation pipeline for training data
- **LangGraph**: Agentic workflow orchestration and state management

### Data Processing
- **Pandas**: ETL pipeline and data transformation
- **NumPy**: Numerical operations and image processing
- **OpenCV**: Image preprocessing and augmentation

### Monitoring & Infrastructure
- **Prometheus**: Metrics collection and alerting
- **Docker**: Containerization (implied from architecture)
- **Server-Sent Events (SSE)**: Real-time dashboard updates

## Development Workflow

### Data Pipeline
```bash
# ETL Pipeline for 4,405 order dataset
python -m mealawe_automation.preprocessing.etl_pipeline

# Training data annotation (CVAT)
# 1. Manual annotation of 100 seed images
# 2. Auto-annotation of remaining 4,300+ images with HITL verification
```

### Model Training
```bash
# YOLOv8 training progression
# Seed phase: YOLOv8-nano-seg on 100 images
# Production phase: YOLOv8-small-seg on full dataset
```

### Testing
```bash
# Property-based testing with Hypothesis
python -m pytest tests/test_properties.py

# Unit testing
python -m pytest tests/

# Model accuracy validation
python -m mealawe_automation.models.validation
```

### Development Server
```bash
# Backend API server
uvicorn mealawe_automation.api.main:app --reload --port 8000

# Frontend development (if applicable)
npm run dev  # React dashboard
```

## Configuration

### Environment Variables
- `MONGODB_URL`: Database connection string
- `YOLO_MODEL_PATH`: Path to YOLOv8 model weights
- `FOODLMM_MODEL_PATH`: Path to FoodLMM model
- `MAX_PROCESSING_TIME_SECONDS`: Performance threshold (default: 3)
- `LOG_LEVEL`: Logging configuration

### Performance Thresholds
- Compartment detection mAP: 94-98%
- Quantity verification accuracy: 88-92%
- Quality assessment accuracy: 75-85%
- End-to-end processing time: <3 seconds

## Build Commands

### Setup
```bash
# Install dependencies (requirements.txt not present, use pip install as needed)
pip install fastapi motor pydantic pandas numpy opencv-python ultralytics

# Database setup
python -m mealawe_automation.database.setup

# Load dataset
python -m mealawe_automation.preprocessing.etl_pipeline --input mealaweProdDatabase.foodorders.xlsx
```

### Testing
```bash
# Run all tests
python -m pytest

# Run property-based tests specifically
python -m pytest tests/test_properties.py -v

# Run with coverage
python -m pytest --cov=mealawe_automation
```