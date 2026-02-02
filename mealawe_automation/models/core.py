"""
Core data models for Mealawe Kitchen Automation System
"""
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic models"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class CompartmentMask(BaseModel):
    """Represents a segmented compartment in a food tray"""
    compartment_id: str
    polygon_coordinates: List[Tuple[float, float]]
    confidence_score: float = Field(ge=0.0, le=1.0)
    item_type: Optional[str] = None
    bounding_box: Dict[str, float]  # {"x_min", "y_min", "x_max", "y_max"}


class QualityIssue(BaseModel):
    """Represents a quality issue detected in food items"""
    issue_type: str  # "burnt", "watery", "portion_size", "texture"
    severity: float = Field(ge=0.0, le=1.0)
    affected_area: CompartmentMask
    description: str
    correction_suggestion: str


class QuantityDiscrepancy(BaseModel):
    """Represents a quantity discrepancy in food portions"""
    item_name: str
    expected_quantity: float
    actual_quantity: float
    discrepancy_percentage: float
    correction_needed: str


class TrayVerificationResult(BaseModel):
    """Result of tray structure verification"""
    passed: bool
    compartments_detected: int
    compartments_expected: int
    compartment_masks: List[CompartmentMask]
    issues: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


class QuantityResult(BaseModel):
    """Result of quantity verification"""
    passed: bool
    item_quantities: Dict[str, float]  # {"Roti": 3, "Dal": 0.8}
    expected_quantities: Dict[str, float]
    discrepancies: List[QuantityDiscrepancy]
    overall_accuracy: float = Field(ge=0.0, le=1.0)


class QualityResult(BaseModel):
    """Result of quality assessment"""
    passed: bool
    quality_scores: Dict[str, float]  # Per compartment quality assessment
    issues: List[QualityIssue]
    reasoning: str  # FoodLMM explanation
    overall_quality_score: float = Field(ge=0.0, le=1.0)


class FinalResult(BaseModel):
    """Final verification result combining all stages"""
    overall_passed: bool
    tray_passed: bool
    quantity_passed: bool
    quality_passed: bool
    correction_actions: List[str]
    processing_time_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VerificationAttempt(BaseModel):
    """Single verification attempt for an order"""
    attempt_id: str
    order_id: str
    image_url: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tray_result: Optional[TrayVerificationResult] = None
    quantity_result: Optional[QuantityResult] = None
    quality_result: Optional[QualityResult] = None
    final_result: Optional[FinalResult] = None
    workflow_state: Dict[str, Any] = Field(default_factory=dict)


class OrderItem(BaseModel):
    """Individual item in an order"""
    item_name: str
    item_type: Optional[str] = None
    item_description: Optional[str] = None
    quantity: int = Field(ge=1)
    add_ons: List[str] = Field(default_factory=list, max_items=4)
    quality_standards: Dict[str, Any] = Field(default_factory=dict)


class Order(BaseModel):
    """Core order model with SOP ground truth"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    order_id: str = Field(unique=True)
    kitchen_id: str
    package_name: Optional[str] = None
    items: List[OrderItem]
    sop_ground_truth: Dict[str, float]  # e.g., {"Roti": 3, "Dal": 1}
    verification_history: List[VerificationAttempt] = Field(default_factory=list)
    current_state: str = "pending"  # "pending", "in_progress", "verified", "rejected"
    workflow_state: Dict[str, Any] = Field(default_factory=dict)
    before_packing_image_url: Optional[str] = None
    after_packing_image_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class KitchenMetrics(BaseModel):
    """Performance metrics for a kitchen"""
    sop_adherence_rate: float = Field(ge=0.0, le=1.0)
    average_correction_time: float  # in seconds
    total_verifications: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class Kitchen(BaseModel):
    """Kitchen/vendor model"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    kitchen_id: str = Field(unique=True)
    name: str
    location: Optional[str] = None
    contact_info: Dict[str, str] = Field(default_factory=dict)
    performance_metrics: KitchenMetrics = Field(default_factory=KitchenMetrics)
    sop_configurations: Dict[str, Any] = Field(default_factory=dict)
    active_orders: List[str] = Field(default_factory=list)
    status: str = "active"  # "active", "inactive", "suspended"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class AnalyticsRecord(BaseModel):
    """Analytics record for time-series analysis"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    record_id: str
    kitchen_id: str
    order_id: str
    verification_result: bool
    processing_time: float
    correction_cycles: int = Field(ge=0)
    tray_accuracy: Optional[float] = None
    quantity_accuracy: Optional[float] = None
    quality_accuracy: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}