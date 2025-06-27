"""
Image Preprocessing Module for Fashion-MNIST CNN
Handles preprocessing of external images for prediction
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class ImagePreprocessor:
    """Handles preprocessing of external images for Fashion-MNIST CNN prediction"""
    
    def __init__(self):
        self.target_size = (28, 28)
        self.target_channels = 1  # Grayscale
    
    def detect_and_fix_background(self, image):
        """
        Detect background color and invert if necessary for Fashion-MNIST compatibility
        
        Args:
            image: PIL Image (grayscale)
            
        Returns:
            PIL Image: Image with corrected background
        """
        # Convert to numpy array for analysis
        img_array = np.array(image)
        height, width = img_array.shape
        
        # Sample background pixels (corners and edge midpoints)
        background_pixels = [
            img_array[0, 0],           # Top-left corner
            img_array[0, width-1],     # Top-right corner
            img_array[height-1, 0],    # Bottom-left corner
            img_array[height-1, width-1], # Bottom-right corner
            img_array[0, width//2],    # Top-middle
            img_array[height-1, width//2], # Bottom-middle
            img_array[height//2, 0],   # Left-middle
            img_array[height//2, width-1]  # Right-middle
        ]
        
        # Calculate average background brightness
        avg_background = np.mean(background_pixels)
        
        # If background is light (>127), invert the image
        if avg_background > 127:
            # Invert colors: white becomes black, black becomes white
            inverted_array = 255 - img_array
            processed_image = Image.fromarray(inverted_array.astype('uint8'))
            print(f"  Background detected as LIGHT ({avg_background:.1f}) - INVERTED colors")
            return processed_image
        else:
            print(f"  Background detected as DARK ({avg_background:.1f}) - kept original")
            return image
        
    def preprocess_image(self, image_path, show_steps=False):
        """
        Preprocess external image for Fashion-MNIST prediction
        
        Args:
            image_path (str): Path to input image
            show_steps (bool): Whether to show preprocessing steps
            
        Returns:
            tuple: (preprocessed_array, original_image, processed_image)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Processing: {os.path.basename(image_path)}")
        
        # Step 1: Load original image
        original_image = Image.open(image_path)
        print(f"  Original size: {original_image.size}, Mode: {original_image.mode}")
        
        # Step 2: Convert to grayscale if needed
        if original_image.mode != 'L':
            processed_image = original_image.convert('L')
            print(f"  Converted to grayscale")
        else:
            processed_image = original_image.copy()
            print(f"  Already grayscale")

        # Step 2.5: Auto-detect and fix background
        processed_image = self.detect_and_fix_background(processed_image)
        
        # Step 3: Resize to 28x28
        processed_image = processed_image.resize(self.target_size, Image.Resampling.LANCZOS)
        print(f"  Resized to {self.target_size}")
        
        # Step 4: Convert to numpy array and normalize
        image_array = np.array(processed_image).astype('float32') / 255.0
        print(f"  Normalized: [{image_array.min():.3f}, {image_array.max():.3f}]")
        
        # Step 5: Reshape for model input (add batch and channel dimensions)
        model_input = image_array.reshape(1, 28, 28, 1)
        print(f"  Model input shape: {model_input.shape}")
        
        if show_steps:
            self.visualize_preprocessing_steps(original_image, processed_image, image_array, image_path)
        
        return model_input, original_image, processed_image
    
    def preprocess_batch(self, image_paths, show_progress=True):
        """
        Preprocess multiple images
        
        Args:
            image_paths (list): List of image paths
            show_progress (bool): Whether to show progress
            
        Returns:
            tuple: (batch_array, original_images, processed_images)
        """
        batch_array = []
        original_images = []
        processed_images = []
        
        for i, image_path in enumerate(image_paths):
            if show_progress:
                print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                model_input, original, processed = self.preprocess_image(image_path)
                batch_array.append(model_input[0])  # Remove batch dimension
                original_images.append(original)
                processed_images.append(processed)
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                continue
        
        if batch_array:
            batch_array = np.array(batch_array)
            print(f"\nBatch processed: {batch_array.shape}")
        
        return batch_array, original_images, processed_images
    
    def visualize_preprocessing_steps(self, original_image, processed_image, normalized_array, image_path):
        """
        Visualize the preprocessing pipeline
        
        Args:
            original_image: Original PIL image
            processed_image: Processed PIL image (28x28 grayscale)
            normalized_array: Normalized numpy array
            image_path: Path to original image
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Processing steps
        # Original image
        axes[0, 0].imshow(original_image, cmap='gray' if original_image.mode == 'L' else None)
        axes[0, 0].set_title(f'Original\n{original_image.size} - {original_image.mode}')
        axes[0, 0].axis('off')
        
        # Processed image
        axes[0, 1].imshow(processed_image, cmap='gray')
        axes[0, 1].set_title(f'Processed\n28×28 Grayscale')
        axes[0, 1].axis('off')
        
        # Normalized array
        axes[0, 2].imshow(normalized_array, cmap='gray')
        axes[0, 2].set_title('Normalized [0,1]')
        axes[0, 2].axis('off')
        
        # Row 2: Analysis
        # Pixel-level view with grid
        axes[1, 0].imshow(normalized_array, cmap='gray', interpolation='nearest')
        axes[1, 0].set_title('Pixel Grid View')
        # Add grid lines
        for i in range(29):
            axes[1, 0].axhline(i-0.5, color='red', linewidth=0.2, alpha=0.5)
            axes[1, 0].axvline(i-0.5, color='red', linewidth=0.2, alpha=0.5)
        axes[1, 0].axis('off')
        
        # Intensity heatmap
        im = axes[1, 1].imshow(normalized_array, cmap='viridis')
        axes[1, 1].set_title('Intensity Heatmap')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        axes[1, 1].axis('off')
        
        # Pixel distribution histogram
        axes[1, 2].hist(normalized_array.flatten(), bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 2].set_title('Pixel Distribution')
        axes[1, 2].set_xlabel('Intensity (0-1)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Image Preprocessing Pipeline: {os.path.basename(image_path)}', fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        output_name = f'preprocessing_{os.path.splitext(os.path.basename(image_path))[0]}.png'
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"  Visualization saved: {output_name}")
        
        plt.show()
    
    def validate_image_format(self, image_path):
        """
        Validate if image can be processed
        
        Args:
            image_path (str): Path to image
            
        Returns:
            dict: Validation results
        """
        results = {
            'valid': False,
            'issues': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                results['issues'].append("File does not exist")
                return results
            
            # Try to open image
            image = Image.open(image_path)
            results['info']['original_size'] = image.size
            results['info']['original_mode'] = image.mode
            results['info']['format'] = image.format
            
            # Check image properties
            width, height = image.size
            
            # Warnings for potential issues
            if width < 28 or height < 28:
                results['warnings'].append(f"Image is very small ({width}×{height}). May lose quality when resized.")
            
            if width > 2000 or height > 2000:
                results['warnings'].append(f"Image is very large ({width}×{height}). Consider resizing before processing.")
            
            aspect_ratio = width / height
            if aspect_ratio > 3 or aspect_ratio < 0.33:
                results['warnings'].append(f"Unusual aspect ratio ({aspect_ratio:.2f}). Fashion items work best with roughly square images.")
            
            if image.mode not in ['L', 'RGB', 'RGBA']:
                results['warnings'].append(f"Unusual color mode ({image.mode}). May not convert properly.")
            
            # If we get here, image is valid
            results['valid'] = True
            
        except Exception as e:
            results['issues'].append(f"Cannot open image: {str(e)}")
        
        return results
    
        
    def create_comparison_grid(self, image_paths, save_path=None):
        """
        Create a comparison grid showing original vs processed images
        
        Args:
            image_paths (list): List of image paths
            save_path (str): Path to save comparison grid
        """
        if not image_paths:
            print("No images provided for comparison")
            return
        
        num_images = len(image_paths)
        fig, axes = plt.subplots(2, num_images, figsize=(4*num_images, 8))
        
        if num_images == 1:
            axes = axes.reshape(-1, 1)
        
        for i, image_path in enumerate(image_paths):
            try:
                # Process image
                model_input, original, processed = self.preprocess_image(image_path)
                
                # Show original
                axes[0, i].imshow(original, cmap='gray' if original.mode == 'L' else None)
                axes[0, i].set_title(f'Original\n{os.path.basename(image_path)}\n{original.size}')
                axes[0, i].axis('off')
                
                # Show processed
                axes[1, i].imshow(processed, cmap='gray')
                axes[1, i].set_title('28×28 Processed')
                axes[1, i].axis('off')
                
            except Exception as e:
                # Show error
                axes[0, i].text(0.5, 0.5, f'Error:\n{str(e)}', ha='center', va='center', 
                               transform=axes[0, i].transAxes)
                axes[0, i].axis('off')
                axes[1, i].axis('off')
        
        plt.suptitle('Original vs Processed Images', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison grid saved: {save_path}")
        
        plt.show()


def main():
    """Test the image preprocessor"""
    print("Testing Image Preprocessor")
    print("=" * 50)
    
    preprocessor = ImagePreprocessor()
    
    # Test with user input
    image_path = input("Enter path to test image (or press Enter to skip): ").strip()
    
    if image_path and os.path.exists(image_path):
        print(f"\n1. Validating image: {image_path}")
        validation = preprocessor.validate_image_format(image_path)
        
        print(f"Valid: {validation['valid']}")
        if validation['issues']:
            print("Issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        if validation['valid']:
            print(f"\n2. Processing image...")
            try:
                model_input, original, processed = preprocessor.preprocess_image(
                    image_path, show_steps=True
                )
                print(f"✓ Successfully preprocessed!")
                print(f"✓ Ready for model prediction with shape: {model_input.shape}")
                
            except Exception as e:
                print(f"✗ Error during preprocessing: {e}")
    else:
        print("No valid image path provided. Skipping test.")
    
    print(f"\n✓ Image preprocessor test completed!")


if __name__ == "__main__":
    main()