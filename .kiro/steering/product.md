# Product Overview

## Mealawe Kitchen Vendor Automation System

The Mealawe Kitchen Vendor Automation system is a real-time agentic verification platform that automates quality and quantity audits in kitchen operations. The system combines computer vision with semantic reasoning to provide automated food quality control, reducing manual inspection overhead while maintaining high standards of food quality and SOP compliance.

## Core Functionality

- **Sequential Verification Pipeline**: Three-stage verification process (tray structure → quantity → quality → final verification)
- **Computer Vision Integration**: YOLOv8-seg for compartment detection and FoodLMM for quality reasoning
- **Self-Corrective Feedback Loop**: Generates specific corrective instructions when issues are detected
- **Real-Time Dashboard**: Kitchen performance monitoring and analytics
- **Agentic Workflow**: LangGraph-orchestrated state machine for verification processes

## Key Performance Targets

- **Processing Time**: Sub-3-second end-to-end verification
- **Compartment Detection**: 94-98% mAP accuracy using YOLOv8-seg
- **Quantity Verification**: 88-92% accuracy using geometric markers
- **Quality Assessment**: 75-85% accuracy using FoodLMM reasoning

## Dataset Scale

- **4,405+ food orders** with 17 data fields
- **Kitchen operations** across multiple vendor locations
- **Before/after packing images** for verification workflows
- **SOP ground truth** for automated compliance checking