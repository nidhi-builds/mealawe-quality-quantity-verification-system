"""
Image preprocessing pipeline for Mealawe Kitchen Automation System
Handles geometric augmentation, lighting normalization, and image storage
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import logging
from datetime import datetime
import hashlib
import json
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAugmentationEngine:
    """Handles geometric augmentation for various shooting angles and conditions"""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
        
        # Define augmentation pipeline for training data
        self.train_transform = A.Compose([
            # Geometric transformations
            A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=10, 
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.6
            ),
            
            # Perspective and distortion (common in phone cameras)
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
            
            # Resize to target size
            A.Resize(height=self.target_size[0], width=self.target_size[1], p=1.0),
            
            # Normalize for model input
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
        ])
        
        # Validation/inference pipeline (minimal augmentation)
        self.val_transform = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1], p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
        ])
        
        # Light augmentation for production (maintains food appearance)
        self.production_transform = A.Compose([
            A.Rotate(limit=5, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomScale(scale_limit=0.1, p=0.3),
            A.Resize(height=self.target_size[0], width=self.target_size[1], p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
        ])
    
    def augment_for_training(self, image: np.ndarray, masks: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """Apply training augmentations to image and masks"""
        try:
            if masks is not None:
                # Apply same transformation to image and masks
                transformed = self.train_transform(image=image, masks=masks)
                return {
                    'image': transformed['image'],
                    'masks': transformed['masks'],
                    'applied_transforms': self._get_applied_transforms(transformed)
                }
            else:
                transformed = self.train_transform(image=image)
                return {
                    'image': transformed['image'],
                    'applied_transforms': self._get_applied_transforms(transformed)
                }
        except Exception as e:
            logger.error(f"Error in training augmentation: {str(e)}")
            # Fallback to validation transform
            return self.augment_for_validation(image, masks)
    
    def augment_for_validation(self, image: np.ndarray, masks: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """Apply validation augmentations (minimal)"""
        try:
            if masks is not None:
                transformed = self.val_transform(image=image, masks=masks)
                return {
                    'image': transformed['image'],
                    'masks': transformed['masks'],
                    'applied_transforms': ['resize', 'normalize']
                }
            else:
                transformed = self.val_transform(image=image)
                return {
                    'image': transformed['image'],
                    'applied_transforms': ['resize', 'normalize']
                }
        except Exception as e:
            logger.error(f"Error in validation augmentation: {str(e)}")
            raise
    
    def augment_for_production(self, image: np.ndarray) -> Dict[str, Any]:
        """Apply light augmentations for production inference"""
        try:
            transformed = self.production_transform(image=image)
            return {
                'image': transformed['image'],
                'applied_transforms': self._get_applied_transforms(transformed)
            }
        except Exception as e:
            logger.error(f"Error in production augmentation: {str(e)}")
            # Fallback to validation transform
            return self.augment_for_validation(image)
    
    def _get_applied_transforms(self, transformed_result: Dict) -> List[str]:
        """Extract list of applied transformations"""
        # This is a simplified version - in practice, you'd track which transforms were applied
        return ['geometric_augmentation', 'resize', 'normalize']


class LightingNormalizer:
    """Handles lighting normalization and color perturbation"""
    
    def __init__(self):
        # Define lighting correction parameters
        self.brightness_range = (0.8, 1.2)
        self.contrast_range = (0.8, 1.2)
        self.saturation_range = (0.8, 1.2)
        self.hue_shift_range = (-10, 10)
    
    def normalize_lighting(self, image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """Normalize lighting conditions in the image"""
        try:
            if method == 'adaptive':
                return self._adaptive_histogram_equalization(image)
            elif method == 'gamma':
                return self._gamma_correction(image)
            elif method == 'white_balance':
                return self._white_balance_correction(image)
            else:
                logger.warning(f"Unknown lighting normalization method: {method}")
                return image
        except Exception as e:
            logger.error(f"Error in lighting normalization: {str(e)}")
            return image
    
    def apply_color_perturbation(self, image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Apply color perturbation for data augmentation"""
        try:
            # Convert to PIL for easier color manipulation
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Apply random brightness adjustment
            brightness_factor = np.random.uniform(
                1 - intensity * 0.2, 1 + intensity * 0.2
            )
            pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness_factor)
            
            # Apply random contrast adjustment
            contrast_factor = np.random.uniform(
                1 - intensity * 0.2, 1 + intensity * 0.2
            )
            pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast_factor)
            
            # Apply random saturation adjustment
            saturation_factor = np.random.uniform(
                1 - intensity * 0.1, 1 + intensity * 0.1
            )
            pil_image = ImageEnhance.Color(pil_image).enhance(saturation_factor)
            
            # Convert back to OpenCV format
            result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return result
            
        except Exception as e:
            logger.error(f"Error in color perturbation: {str(e)}")
            return image
    
    def _adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return result
    
    def _gamma_correction(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """Apply gamma correction for lighting adjustment"""
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction
        return cv2.LUT(image, table)
    
    def _white_balance_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply simple white balance correction"""
        # Calculate average values for each channel
        avg_b = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_r = np.mean(image[:, :, 2])
        
        # Calculate scaling factors
        avg_gray = (avg_b + avg_g + avg_r) / 3
        scale_b = avg_gray / avg_b if avg_b > 0 else 1.0
        scale_g = avg_gray / avg_g if avg_g > 0 else 1.0
        scale_r = avg_gray / avg_r if avg_r > 0 else 1.0
        
        # Apply scaling (with clipping to prevent overflow)
        result = image.copy().astype(np.float32)
        result[:, :, 0] = np.clip(result[:, :, 0] * scale_b, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * scale_g, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * scale_r, 0, 255)
        
        return result.astype(np.uint8)


class ImageStorageManager:
    """Manages image storage and retrieval with proper organization"""
    
    def __init__(self, base_storage_path: str = "data/images"):
        self.base_path = Path(base_storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.raw_images_path = self.base_path / "raw"
        self.processed_images_path = self.base_path / "processed"
        self.augmented_images_path = self.base_path / "augmented"
        self.training_images_path = self.base_path / "training"
        
        for path in [self.raw_images_path, self.processed_images_path, 
                     self.augmented_images_path, self.training_images_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def store_raw_image(self, image: np.ndarray, order_id: str, image_type: str = "before") -> str:
        """Store raw image with proper naming convention"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{order_id}_{image_type}_{timestamp}.jpg"
            filepath = self.raw_images_path / filename
            
            # Save image
            success = cv2.imwrite(str(filepath), image)
            if not success:
                raise Exception(f"Failed to save image to {filepath}")
            
            logger.info(f"Stored raw image: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error storing raw image: {str(e)}")
            raise
    
    def store_processed_image(self, image: np.ndarray, order_id: str, 
                            processing_info: Dict[str, Any]) -> str:
        """Store processed image with metadata"""
        try:
            # Generate filename with processing hash
            processing_hash = self._generate_processing_hash(processing_info)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{order_id}_processed_{processing_hash}_{timestamp}.jpg"
            filepath = self.processed_images_path / filename
            
            # Save image
            success = cv2.imwrite(str(filepath), image)
            if not success:
                raise Exception(f"Failed to save processed image to {filepath}")
            
            # Save metadata
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'order_id': order_id,
                    'processing_info': processing_info,
                    'timestamp': timestamp,
                    'image_path': str(filepath)
                }, f, indent=2)
            
            logger.info(f"Stored processed image: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error storing processed image: {str(e)}")
            raise
    
    def store_augmented_batch(self, images: List[np.ndarray], order_id: str, 
                            augmentation_info: Dict[str, Any]) -> List[str]:
        """Store batch of augmented images for training"""
        try:
            stored_paths = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for i, image in enumerate(images):
                filename = f"{order_id}_aug_{i:03d}_{timestamp}.jpg"
                filepath = self.augmented_images_path / filename
                
                # Save image
                success = cv2.imwrite(str(filepath), image)
                if not success:
                    logger.warning(f"Failed to save augmented image: {filepath}")
                    continue
                
                stored_paths.append(str(filepath))
            
            # Save batch metadata
            metadata_path = self.augmented_images_path / f"{order_id}_batch_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'order_id': order_id,
                    'batch_size': len(images),
                    'augmentation_info': augmentation_info,
                    'timestamp': timestamp,
                    'image_paths': stored_paths
                }, f, indent=2)
            
            logger.info(f"Stored {len(stored_paths)} augmented images for order {order_id}")
            return stored_paths
            
        except Exception as e:
            logger.error(f"Error storing augmented batch: {str(e)}")
            return []
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from storage"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image from {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
    
    def get_image_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Load metadata for an image"""
        try:
            metadata_path = Path(image_path).with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            return None
    
    def _generate_processing_hash(self, processing_info: Dict[str, Any]) -> str:
        """Generate hash for processing parameters"""
        # Create deterministic hash from processing parameters
        processing_str = json.dumps(processing_info, sort_keys=True)
        return hashlib.md5(processing_str.encode()).hexdigest()[:8]


class MealaweImagePipeline:
    """Main image preprocessing pipeline for Mealawe system"""
    
    def __init__(self, storage_path: str = "data/images", target_size: Tuple[int, int] = (640, 640)):
        self.augmentation_engine = ImageAugmentationEngine(target_size)
        self.lighting_normalizer = LightingNormalizer()
        self.storage_manager = ImageStorageManager(storage_path)
        
        # Pipeline statistics
        self.stats = {
            'images_processed': 0,
            'augmentations_generated': 0,
            'lighting_corrections': 0,
            'storage_operations': 0,
            'errors': 0
        }
    
    def process_raw_image(self, image_path: str, order_id: str, 
                         processing_mode: str = 'production') -> Dict[str, Any]:
        """Process a single raw image through the complete pipeline"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"Failed to load image from {image_path}")
            
            # Store original
            original_path = self.storage_manager.store_raw_image(image, order_id, "original")
            
            # Apply lighting normalization
            normalized_image = self.lighting_normalizer.normalize_lighting(image, method='adaptive')
            self.stats['lighting_corrections'] += 1
            
            # Apply appropriate augmentation based on mode
            if processing_mode == 'training':
                augmented_result = self.augmentation_engine.augment_for_training(normalized_image)
            elif processing_mode == 'validation':
                augmented_result = self.augmentation_engine.augment_for_validation(normalized_image)
            else:  # production
                augmented_result = self.augmentation_engine.augment_for_production(normalized_image)
            
            # Store processed image
            processing_info = {
                'mode': processing_mode,
                'lighting_normalization': 'adaptive',
                'applied_transforms': augmented_result.get('applied_transforms', []),
                'original_path': original_path
            }
            
            processed_path = self.storage_manager.store_processed_image(
                augmented_result['image'], order_id, processing_info
            )
            
            self.stats['images_processed'] += 1
            self.stats['storage_operations'] += 2  # original + processed
            
            return {
                'success': True,
                'original_path': original_path,
                'processed_path': processed_path,
                'processed_image': augmented_result['image'],
                'processing_info': processing_info
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            self.stats['errors'] += 1
            return {
                'success': False,
                'error': str(e),
                'original_path': image_path
            }
    
    def generate_training_augmentations(self, image_path: str, order_id: str, 
                                      num_augmentations: int = 5) -> Dict[str, Any]:
        """Generate multiple augmented versions for training"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"Failed to load image from {image_path}")
            
            # Apply lighting normalization
            normalized_image = self.lighting_normalizer.normalize_lighting(image)
            
            # Generate multiple augmentations
            augmented_images = []
            for i in range(num_augmentations):
                # Apply color perturbation
                color_perturbed = self.lighting_normalizer.apply_color_perturbation(
                    normalized_image, intensity=0.3
                )
                
                # Apply geometric augmentation
                augmented_result = self.augmentation_engine.augment_for_training(color_perturbed)
                augmented_images.append(augmented_result['image'])
            
            # Store augmented batch
            augmentation_info = {
                'source_image': image_path,
                'num_augmentations': num_augmentations,
                'lighting_normalization': True,
                'color_perturbation': True,
                'geometric_augmentation': True
            }
            
            stored_paths = self.storage_manager.store_augmented_batch(
                augmented_images, order_id, augmentation_info
            )
            
            self.stats['augmentations_generated'] += len(augmented_images)
            self.stats['storage_operations'] += len(stored_paths)
            
            return {
                'success': True,
                'num_generated': len(augmented_images),
                'stored_paths': stored_paths,
                'augmentation_info': augmentation_info
            }
            
        except Exception as e:
            logger.error(f"Error generating augmentations for {image_path}: {str(e)}")
            self.stats['errors'] += 1
            return {
                'success': False,
                'error': str(e)
            }
    
    def batch_process_images(self, image_paths: List[str], order_ids: List[str], 
                           processing_mode: str = 'production') -> Dict[str, Any]:
        """Process multiple images in batch"""
        if len(image_paths) != len(order_ids):
            raise ValueError("Number of image paths must match number of order IDs")
        
        results = []
        successful = 0
        failed = 0
        
        for image_path, order_id in zip(image_paths, order_ids):
            result = self.process_raw_image(image_path, order_id, processing_mode)
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        return {
            'total_processed': len(image_paths),
            'successful': successful,
            'failed': failed,
            'results': results,
            'pipeline_stats': self.get_statistics()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset pipeline statistics"""
        self.stats = {
            'images_processed': 0,
            'augmentations_generated': 0,
            'lighting_corrections': 0,
            'storage_operations': 0,
            'errors': 0
        }


# Utility functions for easy pipeline usage
def process_single_image(image_path: str, order_id: str, 
                        processing_mode: str = 'production') -> Dict[str, Any]:
    """Convenience function to process a single image"""
    pipeline = MealaweImagePipeline()
    return pipeline.process_raw_image(image_path, order_id, processing_mode)


def generate_training_data(image_path: str, order_id: str, 
                         num_augmentations: int = 5) -> Dict[str, Any]:
    """Convenience function to generate training augmentations"""
    pipeline = MealaweImagePipeline()
    return pipeline.generate_training_augmentations(image_path, order_id, num_augmentations)