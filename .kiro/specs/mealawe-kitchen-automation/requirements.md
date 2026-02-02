# Requirements Document

## Introduction

The Mealawe Kitchen Vendor Automation system is a real-time agentic verification system that automates quality and quantity audits in kitchen operations. The system combines computer vision with semantic reasoning to provide automated food quality control, reducing manual inspection overhead while maintaining high standards of food quality and SOP compliance.

## Glossary

- **System**: The Mealawe Kitchen Vendor Automation system
- **Chef**: Kitchen staff member who uploads food tray images
- **Vendor**: Kitchen operator managing food preparation and quality
- **Order**: A food preparation request with specific items and add-ons
- **Tray**: Physical container with compartments holding prepared food items
- **Compartment**: Individual section of a tray containing specific food items
- **SOP**: Standard Operating Procedure for food quality and preparation
- **Quality_Verifier**: AI component that assesses food quality using computer vision
- **Segmentation_Engine**: YOLOv8-based component for tray compartment detection
- **Food_Reasoner**: FoodLMM component for semantic quality analysis
- **Dashboard**: Real-time visualization interface for kitchen performance
- **Correction_Action**: Specific instruction generated when quality issues are detected

## Requirements

### Requirement 1: Sequential Verification Pipeline

**User Story:** As a chef, I want a systematic verification process that checks tray structure, then quantity, then quality, so that issues are caught in logical order with clear feedback.

#### Acceptance Criteria

1. WHEN a chef uploads a food tray image with an order ID, THE System SHALL first perform tray/compartment structure verification
2. WHEN tray structure verification fails, THE System SHALL reject the image and provide corrective instructions without proceeding to quantity checks
3. WHEN tray structure verification passes, THE System SHALL proceed to quantity verification against SOP ground truth
4. WHEN quantity verification fails, THE System SHALL reject the image and provide corrective instructions without proceeding to quality checks
5. WHEN quantity verification passes, THE System SHALL proceed to quality assessment of food items

### Requirement 2: Tray Structure and Compartment Verification

**User Story:** As a quality manager, I want automated tray structure verification as the first checkpoint, so that only properly structured trays proceed to detailed analysis.

#### Acceptance Criteria

1. WHEN analyzing tray structure, THE Segmentation_Engine SHALL detect compartments with 94-98% mAP accuracy using YOLOv8-seg
2. WHEN compartment detection is complete, THE System SHALL validate compartment count and arrangement against order requirements
3. WHEN tray structure is invalid, THE System SHALL generate specific corrective actions for tray arrangement
4. WHEN compartment segmentation fails, THE System SHALL reject the verification and request image re-upload
5. THE System SHALL complete tray structure verification before proceeding to quantity checks

### Requirement 3: Quantity Verification and SOP Compliance

**User Story:** As a kitchen operator, I want automated quantity verification against SOP ground truth as the second checkpoint, so that I can ensure correct portions after tray structure is validated.

#### Acceptance Criteria

1. WHEN tray structure verification passes, THE System SHALL retrieve SOP Ground Truth from MongoDB using the order ID
2. WHEN performing quantity verification, THE System SHALL achieve 88-92% accuracy using tray rib-line geometric markers
3. WHEN counting liquid items, THE System SHALL use vertical rib markers for depth estimation and portion measurement
4. WHEN quantity discrepancies are detected, THE System SHALL reject the verification and specify exact portion adjustments needed
5. WHEN quantity verification passes, THE System SHALL proceed to quality assessment phase

### Requirement 4: Quality Assessment and Final Verification

**User Story:** As a quality manager, I want automated quality assessment as the final checkpoint after quantity verification, so that only properly portioned food undergoes quality analysis.

#### Acceptance Criteria

1. WHEN quantity verification passes, THE Food_Reasoner SHALL perform quality assessment achieving 75-85% accuracy
2. WHEN detecting quality issues, THE System SHALL identify burnt, watery, or improperly textured food items using reasoning segmentation
3. WHEN quality assessment is complete, THE System SHALL apply final verification logic gate combining tray, quantity, and quality results
4. WHEN all three verification stages pass, THE System SHALL mark the order as verified and complete
5. WHEN any verification stage fails, THE System SHALL provide stage-specific corrective actions and reject the overall verification

### Requirement 5: Kitchen Management and Vendor Portal

**User Story:** As a vendor, I want to manage my kitchen operations and track performance, so that I can maintain quality standards and optimize operations.

#### Acceptance Criteria

1. WHEN a vendor registers, THE System SHALL create a kitchen profile with unique identification
2. WHEN managing orders, THE System SHALL track order status from preparation to completion
3. WHEN managing menu items, THE System SHALL support up to 4 add-ons per order with standardized descriptions
4. WHEN viewing performance analytics, THE System SHALL display SOP adherence rates and average correction times
5. THE System SHALL maintain order history with associated quality and quantity verification results

### Requirement 6: Unified MongoDB Data Strategy

**User Story:** As a system administrator, I want a unified MongoDB-based data strategy, so that the system maintains alignment with existing kitchen infrastructure and simplifies data management.

#### Acceptance Criteria

1. WHEN storing order metadata and SOP ground truth, THE System SHALL use MongoDB to maintain alignment with existing kitchen infrastructure
2. WHEN storing SOP Ground Truth, THE System SHALL maintain nested documents within Menu or Kitchen collections for simplified retrieval
3. WHEN storing AI inference results, THE System SHALL store JSON blobs from YOLO/FoodLMM directly associated with Order ID in MongoDB
4. WHEN implementing analytics, THE System SHALL use MongoDB's time-series-like patterns for real-time analytics to reduce reliance on external OLAP engines
5. THE System SHALL maintain data consistency and support historical comparisons within the unified MongoDB schema

### Requirement 7: AI Model Integration and Performance

**User Story:** As a system architect, I want reliable AI model integration with specialized food domain capabilities, so that the system can provide accurate and consistent quality assessments.

#### Acceptance Criteria

1. WHEN performing compartment segmentation, THE Segmentation_Engine SHALL use YOLOv8-seg model achieving 94-98% mAP accuracy
2. WHEN conducting quality reasoning, THE Food_Reasoner SHALL integrate FoodLMM for reasoning segmentation capabilities
3. WHEN processing images end-to-end, THE System SHALL complete analysis within 3 seconds from upload to result
4. WHEN models require updates, THE System SHALL support automated annotation pipeline using CVAT for polygon masks
5. THE System SHALL maintain model performance metrics and alert on accuracy degradation below thresholds

### Requirement 8: Self-Corrective Feedback Loop with State Persistence

**User Story:** As a chef, I want specific corrective instructions when quality or quantity issues are detected with persistent state tracking, so that I can make multiple correction attempts with system memory of previous failures.

#### Acceptance Criteria

1. WHEN quality or quantity deviations are detected, THE System SHALL generate specific actionable Correction_Actions
2. WHEN corrections are needed, THE System SHALL provide precise instructions (e.g., "Rice portion is low; fill to the top zig-zag line")
3. WHEN a corrected image is uploaded, THE System SHALL compare against the previous failure log stored in MongoDB to verify fixes
4. WHEN multiple correction attempts are made, THE System SHALL persist the conversation/verification state in MongoDB to allow for multi-turn corrective loops
5. THE System SHALL close the corrective loop only when verification passes after re-upload, maintaining complete audit trail of correction attempts

### Requirement 9: Real-Time Dashboard and Analytics

**User Story:** As a kitchen manager, I want real-time visibility into kitchen performance across 4,405+ orders, so that I can monitor operations and identify improvement opportunities.

#### Acceptance Criteria

1. WHEN accessing the Dashboard, THE System SHALL display real-time kitchen performance metrics using React interface
2. WHEN monitoring SOP adherence, THE System SHALL show current compliance rates by individual kitchen
3. WHEN tracking corrections, THE System SHALL display average correction time metrics and trends
4. WHEN viewing analytics, THE System SHALL provide historical trend analysis over the complete dataset
5. THE System SHALL update Dashboard metrics in real-time as new verification results are processed

### Requirement 10: System Health and Performance Monitoring

**User Story:** As a system administrator, I want comprehensive system monitoring with sub-3-second latency guarantees, so that I can ensure reliable operation and quickly identify issues.

#### Acceptance Criteria

1. WHEN monitoring system health, THE System SHALL use Prometheus for metrics collection and alerting
2. WHEN system performance degrades, THE System SHALL generate alerts for critical thresholds including latency spikes
3. WHEN tracking processing latency, THE System SHALL maintain sub-3-second response times for end-to-end verification
4. WHEN monitoring AI models, THE System SHALL track accuracy metrics and performance degradation patterns
5. THE System SHALL provide comprehensive logging for debugging, audit purposes, and failure analysis

### Requirement 11: Order Processing and Workflow Orchestration

**User Story:** As a kitchen operator, I want streamlined order processing with automated quality and quantity checks, so that I can efficiently manage food preparation workflows.

#### Acceptance Criteria

1. WHEN processing orders, THE System SHALL support the complete dataset of 4,405+ food orders with 17 data fields
2. WHEN managing order data, THE System SHALL handle kitchen names, order numbers, item descriptions, and up to 4 add-ons
3. WHEN orchestrating workflows, THE System SHALL use LangGraph for agent state machine management
4. WHEN tracking orders, THE System SHALL maintain before/after packing image associations and verification states
5. THE System SHALL maintain order state consistency throughout the agentic verification pipeline

### Requirement 12: Data Preprocessing and Assisted Labeling Pipeline

**User Story:** As a data scientist, I want automated data preprocessing with human-in-the-loop verification, so that I can scale from 100 seed images to 4,300+ labeled images efficiently.

#### Acceptance Criteria

1. WHEN preprocessing data, THE System SHALL standardize itemDescription text to resolve naming inconsistencies
2. WHEN performing initial annotation, THE System SHALL support manual polygon mask creation using CVAT for 100+ seed images
3. WHEN scaling annotation, THE System SHALL use auto-annotation with human verification for remaining 4,300+ images
4. WHEN maintaining image-data associations, THE System SHALL store mapping in MongoDB between raw images, YOLO-generated polygon masks, and final Human-in-the-loop (HITL) verified labels
5. THE System SHALL progress from YOLOv8-nano-seg for seed training to YOLOv8-small-seg for production with complete annotation audit trail