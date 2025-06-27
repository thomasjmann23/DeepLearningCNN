"""
Main Controller for Fashion-MNIST CNN Classifier
Orchestrates all modules and provides user interface
"""

import os
import sys
from datetime import datetime

from data_loader import FashionMNISTDataLoader
from model_builder import FashionMNISTModelBuilder
from trainer import FashionMNISTTrainer
from image_preprocessor import ImagePreprocessor
from predictor import FashionMNISTPredictor


class FashionMNISTController:
    """Main controller that orchestrates all modules"""
    
    def __init__(self):
        self.data_loader = FashionMNISTDataLoader()
        self.model_builder = FashionMNISTModelBuilder()
        self.trainer = FashionMNISTTrainer()
        self.preprocessor = ImagePreprocessor()
        self.predictor = FashionMNISTPredictor()
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
    
    def show_main_menu(self):
        """Display main menu options"""
        print("\n" + "="*60)
        print("FASHION-MNIST CNN CLASSIFIER")
        print("="*60)
        print("1. Train new model")
        print("2. Load and test existing model")
        print("3. Predict single image")
        print("4. Predict batch of images")
        print("5. Compare different models")
        print("6. Preprocess and visualize image")
        print("7. View dataset information")
        print("8. Exit")
        print("="*60)
    
    def train_new_model(self):
        """Train a new model with user configuration"""
        print("\n" + "="*50)
        print("TRAINING NEW MODEL")
        print("="*50)
        
        # Choose model type
        print("\nAvailable model types:")
        print("1. Simple CNN (fast training, good for testing)")
        print("2. Deep CNN (better accuracy, slower training)")
        print("3. Lightweight CNN (fastest, lower accuracy)")
        
        model_choice = input("Choose model type (1-3): ").strip()
        model_types = {'1': 'simple', '2': 'deep', '3': 'lightweight'}
        model_type = model_types.get(model_choice, 'simple')
        
        # Configure training parameters
        try:
            epochs = int(input(f"Number of epochs (default 15): ").strip() or "15")
            batch_size = int(input(f"Batch size (default 128): ").strip() or "128")
        except ValueError:
            print("Invalid input, using defaults")
            epochs = 15
            batch_size = 128
        
        # Train model
        print(f"\nTraining {model_type} model for {epochs} epochs...")
        try:
            model, history = self.trainer.train_model(
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Evaluate model
            x_train, y_train, x_test, y_test = self.data_loader.load_data()
            results = self.trainer.evaluate_model(x_test, y_test)
            
            # Generate visualizations
            print("\nGenerating training visualizations...")
            self.trainer.plot_training_history()
            
            if 'y_true' in results and 'y_pred' in results:
                self.trainer.plot_confusion_matrix(results['y_true'], results['y_pred'])
                self.trainer.plot_sample_predictions(
                    x_test, y_test, results['y_pred'], results['y_pred_proba']
                )
            
            # Save training report
            self.trainer.save_training_report(results, model_type)
            
            print(f"\n✅ Training completed successfully!")
            print(f"Final accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
    
    def load_and_test_model(self):
        """Load existing model and test it"""
        print("\n" + "="*50)
        print("LOAD AND TEST MODEL")
        print("="*50)
        
        # List available models
        model_dir = 'models'
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
            
            if model_files:
                print("\nAvailable models:")
                for i, model_file in enumerate(model_files, 1):
                    print(f"{i}. {model_file}")
                
                try:
                    choice = int(input(f"Choose model (1-{len(model_files)}): ")) - 1
                    if 0 <= choice < len(model_files):
                        model_path = os.path.join(model_dir, model_files[choice])
                    else:
                        print("Invalid choice")
                        return
                except ValueError:
                    print("Invalid input")
                    return
            else:
                print("No trained models found in 'models' directory")
                return
        else:
            model_path = input("Enter path to model file: ").strip()
            if not os.path.exists(model_path):
                print("Model file not found")
                return
        
        # Load model
        if self.predictor.load_model(model_path):
            # Test on some dataset images
            print("\nTesting model on dataset images...")
            x_train, y_train, x_test, y_test = self.data_loader.load_data()
            
            # Get random test samples
            sample_images, sample_labels = self.data_loader.get_sample_images(x_test, y_test, 10)
            
            # Save test images temporarily and predict
            import tempfile
            import matplotlib.pyplot as plt
            
            temp_paths = []
            for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
                temp_path = f"temp_test_{i}.png"
                plt.imsave(temp_path, img.reshape(28, 28), cmap='gray')
                temp_paths.append(temp_path)
            
            # Make predictions
            results = self.predictor.predict_batch(temp_paths)
            
            # Calculate accuracy
            correct = 0
            for result, true_label in zip(results, sample_labels):
                if 'predicted_class_index' in result:
                    if result['predicted_class_index'] == true_label:
                        correct += 1
            
            accuracy = correct / len(results)
            print(f"\nTest accuracy on {len(results)} samples: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Cleanup temp files
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            print("✅ Model testing completed!")
    
    def predict_single_image(self):
        """Predict a single image"""
        print("\n" + "="*50)
        print("PREDICT SINGLE IMAGE")
        print("="*50)
        
        # Check if model is loaded
        if self.predictor.model is None:
            model_path = input("Enter path to trained model: ").strip()
            if not os.path.exists(model_path):
                print("Model file not found")
                return
            
            if not self.predictor.load_model(model_path):
                return
        
        # Get image path
        image_path = input("Enter path to image: ").strip()
        if not os.path.exists(image_path):
            print("Image file not found")
            return
        
        # Validate image first
        validation = self.preprocessor.validate_image_format(image_path)
        if not validation['valid']:
            print("❌ Image validation failed:")
            for issue in validation['issues']:
                print(f"  - {issue}")
            return
        
        if validation['warnings']:
            print("⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
            
            proceed = input("Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                return
        
        # Make prediction
        try:
            results = self.predictor.predict_single_image(
                image_path, 
                show_visualization=True,
                confidence_threshold=0.3
            )
            
            # Ask to save results
            save = input("\nSave prediction results? (y/n): ").strip().lower()
            if save == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"results/prediction_{timestamp}.txt"
                self.predictor.save_predictions_to_file([results], output_path)
            
            print("✅ Prediction completed!")
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
    
    def predict_batch_images(self):
        """Predict multiple images"""
        print("\n" + "="*50)
        print("PREDICT BATCH OF IMAGES")
        print("="*50)
        
        # Check if model is loaded
        if self.predictor.model is None:
            model_path = input("Enter path to trained model: ").strip()
            if not os.path.exists(model_path):
                print("Model file not found")
                return
            
            if not self.predictor.load_model(model_path):
                return
        
        # Get image paths
        print("\nEnter image paths (one per line, empty line to finish):")
        image_paths = []
        while True:
            path = input().strip()
            if not path:
                break
            if os.path.exists(path):
                image_paths.append(path)
            else:
                print(f"  Warning: {path} not found, skipping")
        
        if not image_paths:
            print("No valid image paths provided")
            return
        
        # Make predictions
        try:
            results = self.predictor.predict_batch(image_paths, show_summary=True)
            
            # Create prediction grid
            create_grid = input("\nCreate prediction grid visualization? (y/n): ").strip().lower()
            if create_grid == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                grid_path = f"results/prediction_grid_{timestamp}.png"
                self.predictor.create_prediction_grid(image_paths, grid_path)
            
            # Save results
            save = input("Save batch results? (y/n): ").strip().lower()
            if save == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"results/batch_predictions_{timestamp}.txt"
                self.predictor.save_predictions_to_file(results, output_path)
            
            print("✅ Batch prediction completed!")
            
        except Exception as e:
            print(f"❌ Batch prediction failed: {e}")
    
    def compare_models(self):
        """Compare different trained models"""
        print("\n" + "="*50)
        print("COMPARE MODELS")
        print("="*50)
        
        # List available models
        model_dir = 'models'
        if not os.path.exists(model_dir):
            print("No models directory found")
            return
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        if len(model_files) < 2:
            print("Need at least 2 models to compare")
            return
        
        print("Available models:")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {model_file}")
        
        # Select models to compare
        try:
            choices = input("Select models to compare (e.g., 1,3,4): ").strip()
            indices = [int(x.strip()) - 1 for x in choices.split(',')]
            selected_models = [model_files[i] for i in indices if 0 <= i < len(model_files)]
        except:
            print("Invalid selection")
            return
        
        if len(selected_models) < 2:
            print("Need at least 2 valid models")
            return
        
        # Compare models on test data
        x_train, y_train, x_test, y_test = self.data_loader.load_data()
        
        comparison_results = []
        for model_file in selected_models:
            model_path = os.path.join(model_dir, model_file)
            print(f"\nEvaluating {model_file}...")
            
            if self.predictor.load_model(model_path):
                # Evaluate on test set
                test_loss, test_accuracy = self.predictor.model.evaluate(x_test, y_test, verbose=0)
                
                comparison_results.append({
                    'model_name': model_file,
                    'test_accuracy': test_accuracy,
                    'test_loss': test_loss,
                    'parameters': self.predictor.model.count_params()
                })
        
        # Display comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(f"{'Model':<30} {'Accuracy':<12} {'Loss':<12} {'Parameters':<12}")
        print("-" * 60)
        
        for result in sorted(comparison_results, key=lambda x: x['test_accuracy'], reverse=True):
            print(f"{result['model_name']:<30} "
                  f"{result['test_accuracy']:.4f}     "
                  f"{result['test_loss']:.4f}     "
                  f"{result['parameters']:,}")
        
        print("✅ Model comparison completed!")
    
    def preprocess_and_visualize(self):
        """Preprocess and visualize an image"""
        print("\n" + "="*50)
        print("PREPROCESS AND VISUALIZE IMAGE")
        print("="*50)
        
        image_path = input("Enter path to image: ").strip()
        if not os.path.exists(image_path):
            print("Image file not found")
            return
        
        try:
            # Validate image
            validation = self.preprocessor.validate_image_format(image_path)
            print(f"\nImage validation:")
            print(f"Valid: {validation['valid']}")
            
            if validation['issues']:
                print("Issues:")
                for issue in validation['issues']:
                    print(f"  - {issue}")
                return
            
            if validation['warnings']:
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")
            
            # Preprocess with visualization
            model_input, original, processed = self.preprocessor.preprocess_image(
                image_path, show_steps=True
            )
            
            print("✅ Image preprocessing completed!")
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
    
    def view_dataset_info(self):
        """Display dataset information"""
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        
        try:
            # Load data
            x_train, y_train, x_test, y_test = self.data_loader.load_data()
            
            print(f"Training set: {x_train.shape[0]:,} images")
            print(f"Test set: {x_test.shape[0]:,} images")
            print(f"Image size: {x_train.shape[1]}×{x_train.shape[2]} pixels")
            print(f"Channels: {x_train.shape[3]} (grayscale)")
            print(f"Classes: {len(self.data_loader.class_names)}")
            
            print(f"\nClass names:")
            for i, class_name in enumerate(self.data_loader.class_names):
                print(f"  {i}: {class_name}")
            
            # Show class distribution
            distribution = self.data_loader.get_class_distribution(y_train)
            print(f"\nClass distribution in training set:")
            for class_name, count in distribution.items():
                print(f"  {class_name}: {count:,} images")
            
            # Show sample images
            show_samples = input("\nShow sample images? (y/n): ").strip().lower()
            if show_samples == 'y':
                import matplotlib.pyplot as plt
                
                sample_images, sample_labels = self.data_loader.get_sample_images(x_train, y_train, 15)
                
                fig, axes = plt.subplots(3, 5, figsize=(12, 8))
                for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
                    row, col = i // 5, i % 5
                    axes[row, col].imshow(img.reshape(28, 28), cmap='gray')
                    axes[row, col].set_title(self.data_loader.get_class_name(label))
                    axes[row, col].axis('off')
                
                plt.suptitle('Fashion-MNIST Sample Images', fontsize=16)
                plt.tight_layout()
                plt.savefig('results/dataset_samples.png', dpi=300, bbox_inches='tight')
                plt.show()
            
            print("✅ Dataset information displayed!")
            
        except Exception as e:
            print(f"❌ Failed to load dataset info: {e}")
    
    def run(self):
        """Main application loop"""
        print("Welcome to Fashion-MNIST CNN Classifier!")
        
        while True:
            try:
                self.show_main_menu()
                choice = input("\nEnter your choice (1-8): ").strip()
                
                if choice == '1':
                    self.train_new_model()
                elif choice == '2':
                    self.load_and_test_model()
                elif choice == '3':
                    self.predict_single_image()
                elif choice == '4':
                    self.predict_batch_images()
                elif choice == '5':
                    self.compare_models()
                elif choice == '6':
                    self.preprocess_and_visualize()
                elif choice == '7':
                    self.view_dataset_info()
                elif choice == '8':
                    print("\nThank you for using Fashion-MNIST CNN Classifier!")
                    print("Goodbye! 👋")
                    break
                else:
                    print("Invalid choice. Please select 1-8.")
                
                # Pause before showing menu again
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\nExiting... Goodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Please try again.")


def main():
    """Entry point of the application"""
    # Set up environment
    import tensorflow as tf
    
    # Suppress TensorFlow warnings (optional)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Check TensorFlow installation
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} device(s)")
        # Enable memory growth to avoid allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("Running on CPU")
    
    print()
    
    # Run the application
    controller = FashionMNISTController()
    controller.run()


if __name__ == "__main__":
    main()