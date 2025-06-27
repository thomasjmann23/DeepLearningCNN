"""
Training Module for Fashion-MNIST CNN
Handles model training, validation, and evaluation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime

from data_loader import FashionMNISTDataLoader
from model_builder import FashionMNISTModelBuilder


class FashionMNISTTrainer:
    """Handles training and evaluation of Fashion-MNIST CNN models"""
    
    def __init__(self, model_save_dir='models'):
        self.model_save_dir = model_save_dir
        self.data_loader = FashionMNISTDataLoader()
        self.model_builder = FashionMNISTModelBuilder()
        self.model = None
        self.history = None
        
        # Create model save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
    def train_model(self, model_type='simple', epochs=15, batch_size=128, 
                   validation_split=None, save_best=True):
        """
        Train a Fashion-MNIST CNN model
        
        Args:
            model_type: Type of model ('simple', 'deep', 'lightweight')
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Validation split ratio (if None, uses test set)
            save_best: Whether to save the best model during training
            
        Returns:
            tuple: (trained_model, training_history)
        """
        print("=" * 60)
        print("FASHION-MNIST CNN TRAINING")
        print("=" * 60)
        
        # Load data
        print("\n1. Loading data...")
        x_train, y_train, x_test, y_test = self.data_loader.load_data()
        
        # Build model
        print(f"\n2. Building {model_type} model...")
        if model_type == 'simple':
            self.model = self.model_builder.build_simple_cnn()
        elif model_type == 'deep':
            self.model = self.model_builder.build_deeper_cnn()
        elif model_type == 'lightweight':
            self.model = self.model_builder.build_lightweight_cnn()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_builder.get_model_summary(self.model)
        
        # Setup validation data
        if validation_split:
            validation_data = None
            print(f"Using validation split: {validation_split}")
        else:
            validation_data = (x_test, y_test)
            validation_split = None
            print("Using test set for validation")
        
        # Create callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(self.model_save_dir, f'{model_type}_model_{timestamp}.keras')
        
        callbacks = self.model_builder.create_callbacks(
            model_save_path=model_save_path if save_best else None
        )
        
        # Train model
        print(f"\n3. Training for {epochs} epochs...")
        print("-" * 40)
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks if save_best else None,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(self.model_save_dir, f'{model_type}_final_{timestamp}.keras')
        self.model_builder.save_model(self.model, final_model_path)
        
        print(f"\n✓ Training completed!")
        print(f"✓ Model saved: {final_model_path}")
        
        return self.model, self.history
    
    def evaluate_model(self, x_test, y_test, show_detailed=True):
        """
        Evaluate trained model performance
        
        Args:
            x_test: Test images
            y_test: Test labels
            show_detailed: Whether to show detailed classification report
            
        Returns:
            dict: Evaluation results
        """
        if self.model is None:
            raise ValueError("No model trained yet. Call train_model() first.")
        
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        
        # Basic evaluation
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Detailed evaluation
        if show_detailed:
            # Get predictions
            y_pred_proba = self.model.predict(x_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_test, axis=1)
            
            # Classification report
            print("\nClassification Report:")
            print("-" * 30)
            print(classification_report(
                y_true, y_pred, 
                target_names=self.data_loader.class_names,
                digits=4
            ))
            
            # Per-class accuracy
            print("\nPer-Class Accuracy:")
            print("-" * 30)
            cm = confusion_matrix(y_true, y_pred)
            class_accuracies = cm.diagonal() / cm.sum(axis=1)
            
            for i, (class_name, accuracy) in enumerate(zip(self.data_loader.class_names, class_accuracies)):
                print(f"  {class_name:<15}: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            results = {
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'class_accuracies': class_accuracies
            }
        else:
            results = {
                'test_accuracy': test_accuracy,
                'test_loss': test_loss
            }
        
        return results
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history curves
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            print("No training history available. Train a model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training history saved: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.data_loader.class_names,
                    yticklabels=self.data_loader.class_names,
                    cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Fashion-MNIST CNN', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
                ha='center', va='top', transform=plt.gca().transAxes, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved: {save_path}")
        
        plt.show()
    
    def plot_sample_predictions(self, x_test, y_test, y_pred, y_pred_proba, 
                               num_samples=15, save_path=None):
        """
        Plot sample predictions with confidence scores
        
        Args:
            x_test: Test images
            y_test: True labels (one-hot)
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            num_samples: Number of samples to show
            save_path: Path to save the plot (optional)
        """
        # Convert one-hot to regular labels
        y_true = np.argmax(y_test, axis=1)
        
        # Select random samples
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        
        # Calculate grid size
        cols = 5
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            row = i // cols
            col = i % cols
            
            # Get image and predictions
            image = x_test[idx].reshape(28, 28)
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            confidence = y_pred_proba[idx][pred_label]
            
            # Plot image
            axes[row, col].imshow(image, cmap='gray')
            
            # Set title with prediction info
            true_class = self.data_loader.get_class_name(true_label)
            pred_class = self.data_loader.get_class_name(pred_label)
            
            if true_label == pred_label:
                color = 'green'
                title = f'✓ {pred_class}\n{confidence:.2%}'
            else:
                color = 'red'
                title = f'✗ {pred_class}\nTrue: {true_class}\n{confidence:.2%}'
            
            axes[row, col].set_title(title, color=color, fontsize=10)
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Sample Predictions (✓ = Correct, ✗ = Incorrect)', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Sample predictions saved: {save_path}")
        
        plt.show()
    
    def save_training_report(self, results, model_type, save_dir='reports'):
        """
        Save comprehensive training report
        
        Args:
            results: Evaluation results
            model_type: Type of model trained
            save_dir: Directory to save report
        """
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(save_dir, f'training_report_{model_type}_{timestamp}.txt')
        
        with open(report_path, 'w') as f:
            f.write("FASHION-MNIST CNN TRAINING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)\n")
            f.write(f"Test Loss: {results['test_loss']:.4f}\n\n")
            
            if 'class_accuracies' in results:
                f.write("PER-CLASS ACCURACY:\n")
                f.write("-" * 20 + "\n")
                for class_name, accuracy in zip(self.data_loader.class_names, results['class_accuracies']):
                    f.write(f"{class_name:<15}: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            
            f.write(f"\nReport saved: {report_path}\n")
        
        print(f"✓ Training report saved: {report_path}")


def main():
    """Example usage of the trainer"""
    print("Fashion-MNIST CNN Trainer")
    print("=" * 50)
    
    # Initialize trainer
    trainer = FashionMNISTTrainer()
    
    # Train a simple model
    print("\nTraining simple CNN model...")
    model, history = trainer.train_model(
        model_type='simple',
        epochs=5,  # Reduced for demo
        batch_size=128
    )
    
    # Load test data for evaluation
    x_train, y_train, x_test, y_test = trainer.data_loader.load_data()
    
    # Evaluate model
    results = trainer.evaluate_model(x_test, y_test)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Plot confusion matrix
    if 'y_true' in results and 'y_pred' in results:
        trainer.plot_confusion_matrix(results['y_true'], results['y_pred'])
        
        # Plot sample predictions
        trainer.plot_sample_predictions(
            x_test, y_test, results['y_pred'], results['y_pred_proba']
        )
    
    # Save training report
    trainer.save_training_report(results, 'simple')
    
    print("\n✓ Training and evaluation completed!")


if __name__ == "__main__":
    main()