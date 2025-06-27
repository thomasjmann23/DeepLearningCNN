"""
Data Loading Module for Fashion-MNIST
Handles all data loading and preprocessing operations
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class FashionMNISTDataLoader:
    """Handles Fashion-MNIST dataset loading and preprocessing"""
    
    def __init__(self):
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        self.num_classes = len(self.class_names)
        
    def load_data(self, normalize=True, one_hot=True):
        """
        Load Fashion-MNIST dataset
        
        Args:
            normalize (bool): Whether to normalize pixel values to [0,1]
            one_hot (bool): Whether to convert labels to one-hot encoding
            
        Returns:
            tuple: (x_train, y_train, x_test, y_test)
        """
        print("Loading Fashion-MNIST dataset...")
        
        # Load raw data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        
        # Store original labels for reference
        self.original_train_labels = y_train.copy()
        self.original_test_labels = y_test.copy()
        
        # Preprocess images
        x_train, x_test = self._preprocess_images(x_train, x_test, normalize)
        
        # Preprocess labels
        if one_hot:
            y_train = to_categorical(y_train, self.num_classes)
            y_test = to_categorical(y_test, self.num_classes)
        
        print(f"✓ Training data: {x_train.shape}, Labels: {y_train.shape}")
        print(f"✓ Test data: {x_test.shape}, Labels: {y_test.shape}")
        print(f"✓ Pixel range: [{x_train.min():.3f}, {x_train.max():.3f}]")
        
        return x_train, y_train, x_test, y_test
    
    def _preprocess_images(self, x_train, x_test, normalize=True):
        """
        Preprocess image data
        
        Args:
            x_train: Training images
            x_test: Test images  
            normalize: Whether to normalize pixel values
            
        Returns:
            tuple: Preprocessed (x_train, x_test)
        """
        # Convert to float32
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
        # Normalize to [0, 1] range
        if normalize:
            x_train /= 255.0
            x_test /= 255.0
        
        # Reshape to add channel dimension (28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        return x_train, x_test
    
    def get_class_name(self, label_index):
        """
        Get class name from label index
        
        Args:
            label_index (int): Class index (0-9)
            
        Returns:
            str: Class name
        """
        if 0 <= label_index < len(self.class_names):
            return self.class_names[label_index]
        return "Unknown"
    
    def get_sample_images(self, x_data, y_data, num_samples=10):
        """
        Get sample images for visualization
        
        Args:
            x_data: Image data
            y_data: Label data (one-hot or regular)
            num_samples: Number of samples to return
            
        Returns:
            tuple: (sample_images, sample_labels)
        """
        # Convert one-hot back to regular labels if needed
        if y_data.ndim > 1 and y_data.shape[1] > 1:
            labels = np.argmax(y_data, axis=1)
        else:
            labels = y_data
        
        # Get random sample
        indices = np.random.choice(len(x_data), num_samples, replace=False)
        
        sample_images = x_data[indices]
        sample_labels = labels[indices]
        
        return sample_images, sample_labels
    
    def get_class_distribution(self, y_data):
        """
        Get distribution of classes in dataset
        
        Args:
            y_data: Label data
            
        Returns:
            dict: Class distribution
        """
        # Convert one-hot back to regular labels if needed
        if y_data.ndim > 1 and y_data.shape[1] > 1:
            labels = np.argmax(y_data, axis=1)
        else:
            labels = y_data
        
        unique, counts = np.unique(labels, return_counts=True)
        
        distribution = {}
        for class_idx, count in zip(unique, counts):
            distribution[self.get_class_name(class_idx)] = count
        
        return distribution


def main():
    """Test the data loader"""
    print("Testing Fashion-MNIST Data Loader")
    print("=" * 50)
    
    # Initialize data loader
    loader = FashionMNISTDataLoader()
    
    # Load data
    x_train, y_train, x_test, y_test = loader.load_data()
    
    # Show class distribution
    print("\nClass Distribution:")
    distribution = loader.get_class_distribution(y_train)
    for class_name, count in distribution.items():
        print(f"  {class_name}: {count:,} images")
    
    # Get sample images
    sample_images, sample_labels = loader.get_sample_images(x_train, y_train, 5)
    print(f"\nSample images shape: {sample_images.shape}")
    
    for i, label in enumerate(sample_labels):
        class_name = loader.get_class_name(label)
        print(f"  Sample {i}: {class_name}")


if __name__ == "__main__":
    main()