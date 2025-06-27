"""
Model Builder Module for Fashion-MNIST CNN
Handles model architecture creation and configuration
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class FashionMNISTModelBuilder:
    """Builds and configures CNN models for Fashion-MNIST classification"""
    
    def __init__(self):
        self.input_shape = (28, 28, 1)
        self.num_classes = 10
        
    def build_simple_cnn(self):
        """
        Build a simple CNN model
        
        Returns:
            tensorflow.keras.Model: Compiled CNN model
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, name='conv1'),
            MaxPooling2D((2, 2), name='pool1'),
            
            Conv2D(64, (3, 3), activation='relu', name='conv2'),
            MaxPooling2D((2, 2), name='pool2'),
            
            Flatten(name='flatten'),
            Dense(128, activation='relu', name='dense1'),
            Dropout(0.5, name='dropout'),
            Dense(self.num_classes, activation='softmax', name='output')
        ], name='SimpleCNN')
        
        self._compile_model(model)
        return model
    
    def build_deeper_cnn(self):
        """
        Build a deeper CNN model with more layers
        
        Returns:
            tensorflow.keras.Model: Compiled deeper CNN model
        """
        model = Sequential([
            # First conv block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, name='conv1'),
            Conv2D(32, (3, 3), activation='relu', name='conv2'),
            MaxPooling2D((2, 2), name='pool1'),
            Dropout(0.25, name='dropout1'),
            
            # Second conv block
            Conv2D(64, (3, 3), activation='relu', name='conv3'),
            Conv2D(64, (3, 3), activation='relu', name='conv4'),
            MaxPooling2D((2, 2), name='pool2'),
            Dropout(0.25, name='dropout2'),
            
            # Dense layers
            Flatten(name='flatten'),
            Dense(256, activation='relu', name='dense1'),
            Dropout(0.5, name='dropout3'),
            Dense(128, activation='relu', name='dense2'),
            Dropout(0.5, name='dropout4'),
            Dense(self.num_classes, activation='softmax', name='output')
        ], name='DeepCNN')
        
        self._compile_model(model)
        return model
    
    def build_lightweight_cnn(self):
        """
        Build a lightweight CNN model for faster training/inference
        
        Returns:
            tensorflow.keras.Model: Compiled lightweight CNN model
        """
        model = Sequential([
            Conv2D(16, (5, 5), activation='relu', input_shape=self.input_shape, name='conv1'),
            MaxPooling2D((2, 2), name='pool1'),
            
            Conv2D(32, (3, 3), activation='relu', name='conv2'),
            MaxPooling2D((2, 2), name='pool2'),
            
            Flatten(name='flatten'),
            Dense(64, activation='relu', name='dense1'),
            Dropout(0.3, name='dropout'),
            Dense(self.num_classes, activation='softmax', name='output')
        ], name='LightweightCNN')
        
        self._compile_model(model, learning_rate=0.001)
        return model
    
    def _compile_model(self, model, learning_rate=0.001, optimizer='adam'):
        """
        Compile the model with standard settings
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
        """
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_model_summary(self, model):
        """
        Get detailed model summary
        
        Args:
            model: Keras model
            
        Returns:
            str: Model summary string
        """
        print(f"\nModel: {model.name}")
        print("=" * 50)
        model.summary()
        
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\nParameter Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {non_trainable_params:,}")
        
        return model.summary()
    
    def create_callbacks(self, model_save_path='best_model.keras', patience=5):
        """
        Create training callbacks
        
        Args:
            model_save_path: Path to save best model
            patience: Patience for early stopping
            
        Returns:
            list: List of callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=max(2, patience//2),
                min_lr=0.00001,
                verbose=1,
                mode='min'
            ),
            
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            )
        ]
        
        return callbacks
    
    def load_model(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path: Path to saved model
            
        Returns:
            tensorflow.keras.Model: Loaded model
        """
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"✓ Model loaded from: {model_path}")
            return model
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None
    
    def save_model(self, model, save_path):
        """
        Save model to file
        
        Args:
            model: Keras model to save
            save_path: Path to save model
        """
        try:
            model.save(save_path)
            print(f"✓ Model saved to: {save_path}")
        except Exception as e:
            print(f"✗ Error saving model: {e}")


def main():
    """Test the model builder"""
    print("Testing Fashion-MNIST Model Builder")
    print("=" * 50)
    
    builder = FashionMNISTModelBuilder()
    
    # Test different model architectures
    print("\n1. Simple CNN:")
    simple_model = builder.build_simple_cnn()
    builder.get_model_summary(simple_model)
    
    print("\n2. Deeper CNN:")
    deep_model = builder.build_deeper_cnn()
    builder.get_model_summary(deep_model)
    
    print("\n3. Lightweight CNN:")
    light_model = builder.build_lightweight_cnn()
    builder.get_model_summary(light_model)
    
    # Test callbacks
    print("\n4. Creating callbacks...")
    callbacks = builder.create_callbacks()
    print(f"✓ Created {len(callbacks)} callbacks")
    for callback in callbacks:
        print(f"  - {callback.__class__.__name__}")


if __name__ == "__main__":
    main()