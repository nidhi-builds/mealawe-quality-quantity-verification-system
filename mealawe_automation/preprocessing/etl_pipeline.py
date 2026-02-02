"""
ETL Pipeline for Mealawe Kitchen Automation System
Handles extraction, transformation, and loading of the 4,405 order dataset
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from pathlib import Path

from ..models.core import Order, OrderItem, Kitchen
from ..database.operations import DatabaseOperations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextNormalizer:
    """Handles text normalization and standardization for food items"""
    
    # Mapping for common food item variations
    ITEM_MAPPINGS = {
        # Roti variations
        'chapati': 'roti',
        'phulka': 'roti',
        'rotli': 'roti',
        'fulka': 'roti',
        
        # Rice variations
        'chawal': 'rice',
        'bhat': 'rice',
        'steamed_rice': 'rice',
        
        # Dal variations
        'daal': 'dal',
        'lentil': 'dal',
        'pulse': 'dal',
        
        # Sabzi variations
        'vegetable': 'sabzi',
        'curry': 'sabzi',
        'subji': 'sabzi',
        
        # Common add-ons
        'extra_ghee': 'ghee',
        'additional_ghee': 'ghee',
        'butter': 'ghee',
        'pickle': 'achar',
        'achaar': 'achar',
        'papad': 'papadum',
        'poppadom': 'papadum'
    }
    
    @classmethod
    def normalize_item_name(cls, item_name: str) -> str:
        """Normalize item names to standard format"""
        if not item_name or pd.isna(item_name):
            return ""
            
        # Convert to lowercase and remove extra spaces
        normalized = str(item_name).lower().strip()
        
        # Remove special characters and numbers
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\d+', '', normalized)
        normalized = re.sub(r'\s+', '_', normalized)
        
        # Apply mappings
        if normalized in cls.ITEM_MAPPINGS:
            normalized = cls.ITEM_MAPPINGS[normalized]
            
        return normalized
    
    @classmethod
    def extract_quantity_from_description(cls, description: str) -> int:
        """Extract quantity from item description"""
        if not description or pd.isna(description):
            return 1
            
        # Look for numbers in the description
        numbers = re.findall(r'\d+', str(description))
        if numbers:
            # Take the first number found
            quantity = int(numbers[0])
            # Sanity check for reasonable quantities
            if 1 <= quantity <= 20:
                return quantity
        
        return 1


class OutlierDetector:
    """Detects and handles outliers in order data"""
    
    # Business rules for reasonable quantities
    MAX_QUANTITIES = {
        'roti': 10,
        'rice': 5,  # portions
        'dal': 3,   # portions
        'sabzi': 3, # portions
        'default': 15
    }
    
    MAX_ADDONS = 4
    
    @classmethod
    def detect_quantity_outliers(cls, item_name: str, quantity: int) -> Tuple[bool, str]:
        """Detect if quantity is an outlier"""
        max_allowed = cls.MAX_QUANTITIES.get(item_name, cls.MAX_QUANTITIES['default'])
        
        if quantity > max_allowed:
            return True, f"Quantity {quantity} exceeds maximum {max_allowed} for {item_name}"
        
        if quantity <= 0:
            return True, f"Invalid quantity {quantity} for {item_name}"
            
        return False, ""
    
    @classmethod
    def detect_addon_outliers(cls, addons: List[str]) -> Tuple[bool, str]:
        """Detect if add-ons configuration is an outlier"""
        if len(addons) > cls.MAX_ADDONS:
            return True, f"Too many add-ons: {len(addons)} (max: {cls.MAX_ADDONS})"
            
        return False, ""


class SOPGroundTruthGenerator:
    """Generates SOP Ground Truth from order data"""
    
    @classmethod
    def generate_sop_from_items(cls, items: List[OrderItem]) -> Dict[str, float]:
        """Generate SOP ground truth from order items"""
        sop_truth = {}
        
        for item in items:
            item_key = item.item_name.lower()
            
            # For liquid items, use portion sizes (0.5, 1.0, 1.5)
            if item_key in ['dal', 'curry', 'sabzi']:
                sop_truth[item_key] = float(item.quantity)
            else:
                # For discrete items, use actual count
                sop_truth[item_key] = float(item.quantity)
                
        return sop_truth


class MealaweETLPipeline:
    """Main ETL pipeline for processing Mealawe dataset"""
    
    def __init__(self, db_operations: DatabaseOperations):
        self.db_ops = db_operations
        self.text_normalizer = TextNormalizer()
        self.outlier_detector = OutlierDetector()
        self.sop_generator = SOPGroundTruthGenerator()
        
        # Statistics tracking
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'outliers_detected': 0,
            'normalization_changes': 0,
            'errors': 0
        }
    
    def extract_data(self, excel_path: str) -> pd.DataFrame:
        """Extract data from Excel file"""
        logger.info(f"Extracting data from {excel_path}")
        
        try:
            df = pd.read_excel(excel_path)
            self.stats['total_records'] = len(df)
            logger.info(f"Extracted {len(df)} records with {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            raise
    
    def transform_order_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Transform raw data into structured order format"""
        logger.info("Starting data transformation...")
        
        transformed_orders = []
        
        for idx, row in df.iterrows():
            try:
                # Extract basic order info
                order_data = self._extract_order_basics(row)
                
                # Process items
                items = self._extract_items(row)
                if not items:
                    logger.warning(f"No items found for order {order_data.get('order_id', idx)}")
                    continue
                
                # Process add-ons
                addons = self._extract_addons(row)
                
                # Apply add-ons to items (distribute across items)
                items = self._apply_addons_to_items(items, addons)
                
                # Generate SOP ground truth
                sop_ground_truth = self.sop_generator.generate_sop_from_items(items)
                
                # Create complete order
                order_data.update({
                    'items': [item.dict() for item in items],
                    'sop_ground_truth': sop_ground_truth,
                    'before_packing_image_url': self._clean_url(row.get('beforePackingImageUrl')),
                    'after_packing_image_url': self._clean_url(row.get('afterPackingImageUrl'))
                })
                
                # Validate and detect outliers
                outlier_issues = self._detect_order_outliers(order_data)
                if outlier_issues:
                    logger.warning(f"Outliers detected in order {order_data['order_id']}: {outlier_issues}")
                    self.stats['outliers_detected'] += 1
                
                transformed_orders.append(order_data)
                self.stats['processed_records'] += 1
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                self.stats['errors'] += 1
                continue
        
        logger.info(f"Transformation complete. Processed {len(transformed_orders)} orders")
        return transformed_orders
    
    def _extract_order_basics(self, row: pd.Series) -> Dict[str, Any]:
        """Extract basic order information"""
        return {
            'order_id': str(row.get('orderNo', f"order_{row.name}")),
            'kitchen_id': self._extract_kitchen_id(row.get('kitchenName', 'unknown')),
            'package_name': str(row.get('packageName', '')).strip() if pd.notna(row.get('packageName')) else None,
            'current_state': 'pending',
            'workflow_state': {},
            'verification_history': []
        }
    
    def _extract_kitchen_id(self, kitchen_name: str) -> str:
        """Generate kitchen ID from kitchen name"""
        if not kitchen_name or pd.isna(kitchen_name):
            return 'kitchen_unknown'
        
        # Normalize kitchen name to create ID
        kitchen_id = re.sub(r'[^\w\s]', '', str(kitchen_name).lower())
        kitchen_id = re.sub(r'\s+', '_', kitchen_id)
        return f"kitchen_{kitchen_id}"
    
    def _extract_items(self, row: pd.Series) -> List[OrderItem]:
        """Extract items from row data"""
        items = []
        
        # Extract main item
        item_name = row.get('itemList[0].itemName')
        if item_name and pd.notna(item_name):
            normalized_name = self.text_normalizer.normalize_item_name(item_name)
            if normalized_name != item_name.lower().replace(' ', '_'):
                self.stats['normalization_changes'] += 1
            
            quantity = self.text_normalizer.extract_quantity_from_description(
                row.get('itemList[0].itemDescription', '')
            )
            
            item = OrderItem(
                item_name=normalized_name,
                item_type=str(row.get('itemList[0].itemType', '')).strip() if pd.notna(row.get('itemList[0].itemType')) else None,
                item_description=str(row.get('itemList[0].itemDescription', '')).strip() if pd.notna(row.get('itemList[0].itemDescription')) else None,
                quantity=quantity,
                add_ons=[],
                quality_standards={}
            )
            items.append(item)
        
        return items
    
    def _extract_addons(self, row: pd.Series) -> List[Tuple[str, int]]:
        """Extract add-ons with their counts"""
        addons = []
        
        for i in range(4):  # Up to 4 add-ons
            addon_name = row.get(f'addOns[{i}].addOnName')
            addon_count = row.get(f'addOns[{i}].count')
            
            if addon_name and pd.notna(addon_name):
                normalized_addon = self.text_normalizer.normalize_item_name(addon_name)
                count = int(addon_count) if pd.notna(addon_count) else 1
                addons.append((normalized_addon, count))
        
        return addons
    
    def _apply_addons_to_items(self, items: List[OrderItem], addons: List[Tuple[str, int]]) -> List[OrderItem]:
        """Apply add-ons to items"""
        if not addons or not items:
            return items
        
        # For simplicity, add all add-ons to the first item
        # In a more complex system, we might have rules for which add-ons go with which items
        if items:
            addon_names = [addon[0] for addon in addons]
            items[0].add_ons = addon_names[:4]  # Limit to 4 add-ons
        
        return items
    
    def _clean_url(self, url: Any) -> Optional[str]:
        """Clean and validate image URLs"""
        if not url or pd.isna(url):
            return None
        
        url_str = str(url).strip()
        if url_str.startswith('http'):
            return url_str
        
        return None
    
    def _detect_order_outliers(self, order_data: Dict[str, Any]) -> List[str]:
        """Detect outliers in order data"""
        issues = []
        
        for item_data in order_data.get('items', []):
            # Check quantity outliers
            is_outlier, message = self.outlier_detector.detect_quantity_outliers(
                item_data['item_name'], item_data['quantity']
            )
            if is_outlier:
                issues.append(message)
            
            # Check add-on outliers
            is_outlier, message = self.outlier_detector.detect_addon_outliers(
                item_data.get('add_ons', [])
            )
            if is_outlier:
                issues.append(message)
        
        return issues
    
    def load_data(self, transformed_orders: List[Dict[str, Any]]) -> bool:
        """Load transformed data into MongoDB"""
        logger.info("Loading data into MongoDB...")
        
        try:
            # Extract unique kitchens and create them first
            kitchens = self._extract_unique_kitchens(transformed_orders)
            for kitchen_data in kitchens:
                kitchen = Kitchen(**kitchen_data)
                self.db_ops.create_kitchen(kitchen)
            
            # Load orders
            loaded_count = 0
            for order_data in transformed_orders:
                try:
                    order = Order(**order_data)
                    self.db_ops.create_order(order)
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Error loading order {order_data.get('order_id')}: {str(e)}")
                    self.stats['errors'] += 1
            
            logger.info(f"Successfully loaded {loaded_count} orders and {len(kitchens)} kitchens")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def _extract_unique_kitchens(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique kitchen data from orders"""
        kitchens = {}
        
        for order in orders:
            kitchen_id = order['kitchen_id']
            if kitchen_id not in kitchens:
                # Extract kitchen name from the original data
                kitchen_name = kitchen_id.replace('kitchen_', '').replace('_', ' ').title()
                
                kitchens[kitchen_id] = {
                    'kitchen_id': kitchen_id,
                    'name': kitchen_name,
                    'status': 'active',
                    'contact_info': {},
                    'sop_configurations': {},
                    'active_orders': []
                }
        
        return list(kitchens.values())
    
    def run_pipeline(self, excel_path: str) -> Dict[str, Any]:
        """Run the complete ETL pipeline"""
        logger.info("Starting Mealawe ETL Pipeline...")
        start_time = datetime.now()
        
        try:
            # Extract
            df = self.extract_data(excel_path)
            
            # Transform
            transformed_orders = self.transform_order_data(df)
            
            # Load
            success = self.load_data(transformed_orders)
            
            # Calculate final statistics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            self.stats.update({
                'success': success,
                'processing_time_seconds': processing_time,
                'completion_time': end_time.isoformat()
            })
            
            logger.info(f"ETL Pipeline completed in {processing_time:.2f} seconds")
            logger.info(f"Final statistics: {self.stats}")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {str(e)}")
            self.stats.update({
                'success': False,
                'error': str(e)
            })
            return self.stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.stats.copy()


# Utility function for easy pipeline execution
def run_etl_pipeline(excel_path: str, db_operations: DatabaseOperations) -> Dict[str, Any]:
    """Convenience function to run the ETL pipeline"""
    pipeline = MealaweETLPipeline(db_operations)
    return pipeline.run_pipeline(excel_path)