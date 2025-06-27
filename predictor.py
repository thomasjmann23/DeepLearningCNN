"""
Prediction Module for Fashion-MNIST CNN
Handles model loading and image prediction with visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

from image_preprocessor import ImagePreprocessor
from data_loader import FashionMNISTDataLoader


class FashionMNISTPredictor:
    """Handles loading trained models and making predictions on new images"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self.preprocessor = ImagePreprocessor()
        self.data_loader = FashionMNISTDataLoader()
        self.class_names = self.data_loader.class_names
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load trained model from file
        
        Args:
            model_path (str): Path to saved model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            print(f"✓ Model loaded successfully from: {os.path.basename(model_path)}")
            
            # Display model info
            print(f"  Model name: {self.model.name}")
            print(f"  Input shape: {self.model.input_shape}")
            print(f"  Output shape: {self.model.output_shape}")
            print(f"  Parameters: {self.model.count_params():,}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
            return False
    
    def predict_single_image(self, image_path, show_visualization=True, confidence_threshold=0.1):
        """
        Predict class of a single image
        
        Args:
            image_path (str): Path to image file
            show_visualization (bool): Whether to show prediction visualization
            confidence_threshold (float): Minimum confidence to highlight in visualization
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Use load_model() first.")
        
        print(f"Predicting: {os.path.basename(image_path)}")
        
        # Preprocess image
        model_input, original_image, processed_image = self.preprocessor.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(model_input, verbose=0)
        prediction_probs = predictions[0]
        
        # Get results
        predicted_class_idx = np.argmax(prediction_probs)
        predicted_class_name = self.class_names[predicted_class_idx]
        confidence = prediction_probs[predicted_class_idx]
        
        # Create results dictionary
        results = {
            'image_path': image_path,
            'predicted_class_index': predicted_class_idx,
            'predicted_class_name': predicted_class_name,
            'confidence': confidence,
            'all_probabilities': prediction_probs,
            'original_image': original_image,
            'processed_image': processed_image
        }
        
        # Print results
        print(f"  Prediction: {predicted_class_name}")
        print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Show top 3 predictions
        top_3_indices = np.argsort(prediction_probs)[::-1][:3]
        print(f"  Top 3 predictions:")
        for i, idx in enumerate(top_3_indices):
            print(f"    {i+1}. {self.class_names[idx]}: {prediction_probs[idx]:.4f} ({prediction_probs[idx]*100:.2f}%)")
        
        # Show warnings for low confidence
        if confidence < confidence_threshold:
            print(f"  ⚠️  Warning: Low confidence prediction ({confidence:.2%})")
        
        # Visualization
        if show_visualization:
            self.visualize_prediction(results, confidence_threshold)
        
        return results
    
    def predict_batch(self, image_paths, show_summary=True):
        """
        Predict classes for multiple images
        
        Args:
            image_paths (list): List of image paths
            show_summary (bool): Whether to show prediction summary
            
        Returns:
            list: List of prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Use load_model() first.")
        
        print(f"Predicting batch of {len(image_paths)} images...")
        
        all_results = []
        successful_predictions = 0
        
        for i, image_path in enumerate(image_paths):
            print(f"\n{i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                results = self.predict_single_image(image_path, show_visualization=False)
                all_results.append(results)
                successful_predictions += 1
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                all_results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        if show_summary:
            self.show_batch_summary(all_results, successful_predictions)
        
        return all_results
    
    def visualize_prediction(self, results, confidence_threshold=0.1):
        """
        Create visualization of prediction results
        
        Args:
            results (dict): Prediction results
            confidence_threshold (float): Threshold for highlighting low confidence
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        
        # Original image
        original = results['original_image']
        ax1.imshow(original, cmap='gray' if original.mode == 'L' else None)
        ax1.set_title(f'Original Image\n{os.path.basename(results["image_path"])}\nSize: {original.size}')
        ax1.axis('off')
        
        # Processed image (what model sees)
        processed = results['processed_image']
        ax2.imshow(processed, cmap='gray')
        ax2.set_title('Processed (28×28)\nWhat Model Sees')
        ax2.axis('off')
        
        # Add pixel grid to processed image
        for i in range(29):
            ax2.axhline(i-0.5, color='white', linewidth=0.1, alpha=0.3)
            ax2.axvline(i-0.5, color='white', linewidth=0.1, alpha=0.3)
        
        # Prediction probabilities
        probabilities = results['all_probabilities']
        predicted_idx = results['predicted_class_index']
        confidence = results['confidence']
        
        # Create bar chart
        bars = ax3.bar(range(len(self.class_names)), probabilities, color='lightblue')
        
        # Highlight predicted class
        bars[predicted_idx].set_color('orange' if confidence >= confidence_threshold else 'red')
        
        # Set labels and title
        ax3.set_xlabel('Fashion Classes')
        ax3.set_ylabel('Probability')
        
        title_color = 'black' if confidence >= confidence_threshold else 'red'
        title = f'Prediction: {results["predicted_class_name"]}\nConfidence: {confidence:.2%}'
        if confidence < confidence_threshold:
            title += '\n⚠️ Low Confidence'
        
        ax3.set_title(title, color=title_color)
        ax3.set_xticks(range(len(self.class_names)))
        ax3.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Add probability text on top of bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            if prob > 0.05:  # Only show text for significant probabilities
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save visualization
        output_name = f'prediction_{os.path.splitext(os.path.basename(results["image_path"]))[0]}.png'
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"  Visualization saved: {output_name}")
        
        plt.show()
    
    def show_batch_summary(self, all_results, successful_predictions):
        """
        Show summary of batch predictions
        
        Args:
            all_results (list): List of all prediction results
            successful_predictions (int): Number of successful predictions
        """
        print(f"\n" + "="*60)
        print("BATCH PREDICTION SUMMARY")
        print("="*60)
        
        print(f"Total images: {len(all_results)}")
        print(f"Successful predictions: {successful_predictions}")
        print(f"Failed predictions: {len(all_results) - successful_predictions}")
        
        if successful_predictions > 0:
            # Count predictions by class
            class_counts = {}
            confidence_scores = []
            
            for result in all_results:
                if 'predicted_class_name' in result:
                    class_name = result['predicted_class_name']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    confidence_scores.append(result['confidence'])
            
            print(f"\nPrediction Distribution:")
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count} images")
            
            print(f"\nConfidence Statistics:")
            confidence_scores = np.array(confidence_scores)
            print(f"  Average confidence: {confidence_scores.mean():.3f}")
            print(f"  Min confidence: {confidence_scores.min():.3f}")
            print(f"  Max confidence: {confidence_scores.max():.3f}")
            print(f"  Std deviation: {confidence_scores.std():.3f}")
            
            # Count low confidence predictions
            low_confidence = np.sum(confidence_scores < 0.5)
            if low_confidence > 0:
                print(f"  ⚠️  {low_confidence} predictions with confidence < 50%")
    
    def create_prediction_grid(self, image_paths, save_path=None):
        """
        Create a grid showing multiple predictions
        
        Args:
            image_paths (list): List of image paths
            save_path (str): Path to save the grid
        """
        if not image_paths:
            print("No images provided for prediction grid")
            return
        
        # Predict all images
        results = self.predict_batch(image_paths, show_summary=False)
        
        # Filter successful predictions
        successful_results = [r for r in results if 'predicted_class_name' in r]
        
        if not successful_results:
            print("No successful predictions to display")
            return
        
        # Create grid
        num_images = len(successful_results)
        cols = min(5, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, result in enumerate(successful_results):
            row = i // cols
            col = i % cols
            
            # Show processed image
            axes[row, col].imshow(result['processed_image'], cmap='gray')
            
            # Create title
            confidence = result['confidence']
            title = f'{result["predicted_class_name"]}\n{confidence:.2%}'
            
            color = 'green' if confidence >= 0.7 else 'orange' if confidence >= 0.4 else 'red'
            axes[row, col].set_title(title, color=color, fontsize=10)
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Batch Predictions ({num_images} images)', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Prediction grid saved: {save_path}")
        
        plt.show()
    
    def save_predictions_to_file(self, results, output_path):
        """
        Save prediction results to text file
        
        Args:
            results (list): List of prediction results
            output_path (str): Path to save results
        """
        with open(output_path, 'w') as f:
            f.write("FASHION-MNIST PREDICTION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {os.path.basename(self.model_path) if self.model_path else 'Unknown'}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {os.path.basename(result['image_path'])}\n")
                
                if 'predicted_class_name' in result:
                    f.write(f"   Predicted: {result['predicted_class_name']}\n")
                    f.write(f"   Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)\n")
                    
                    # Top 3 predictions
                    probs = result['all_probabilities']
                    top_3 = np.argsort(probs)[::-1][:3]
                    f.write(f"   Top 3:\n")
                    for j, idx in enumerate(top_3):
                        f.write(f"     {j+1}. {self.class_names[idx]}: {probs[idx]:.4f}\n")
                else:
                    f.write(f"   Error: {result.get('error', 'Unknown error')}\n")
                
                f.write("\n")
        
        print(f"✓ Results saved to: {output_path}")


def main():
    """Example usage of the predictor"""
    print("Fashion-MNIST CNN Predictor")
    print("=" * 50)
    
    # Initialize predictor
    model_path = input("Enter path to trained model (or press Enter to skip): ").strip()
    
    if not model_path or not os.path.exists(model_path):
        print("No valid model path provided. Please train a model first.")
        return
    
    predictor = FashionMNISTPredictor(model_path)
    
    if predictor.model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Test single image prediction
    image_path = input("Enter path to test image (or press Enter to skip): ").strip()
    
    if image_path and os.path.exists(image_path):
        print(f"\nMaking prediction...")
        try:
            results = predictor.predict_single_image(image_path, show_visualization=True)
            print(f"✓ Prediction completed!")
            
            # Ask if user wants to save results
            save_results = input("Save results to file? (y/n): ").strip().lower()
            if save_results == 'y':
                output_path = f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                predictor.save_predictions_to_file([results], output_path)
                
        except Exception as e:
            print(f"✗ Error during prediction: {e}")
    else:
        print("No valid image path provided. Skipping prediction test.")
    
    print(f"\n✓ Predictor test completed!")


if __name__ == "__main__":
    main()