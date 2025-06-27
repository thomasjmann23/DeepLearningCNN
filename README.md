# Fashion-MNIST CNN Classifier

A modular, production-ready Fashion-MNIST CNN classifier with complete training, evaluation, and prediction capabilities.

## 🏗️ Project Structure

```
fashion-mnist-cnn/
├── data_loader.py           # Data loading and preprocessing
├── model_builder.py         # CNN model architectures
├── trainer.py              # Training and evaluation
├── image_preprocessor.py    # External image preprocessing
├── predictor.py            # Model prediction interface
├── main.py                 # Main controller and UI
├── README.md               # This file
├── models/                 # Saved trained models
├── results/                # Prediction results and visualizations
└── reports/                # Training reports
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow pillow matplotlib scikit-learn seaborn numpy
```

### Basic Usage

1. **Run the application:**
   ```bash
   python main.py
   ```

2. **Choose from the menu:**
   - Train new model
   - Predict images
   - Compare models
   - View dataset info

### Module Usage Examples

#### Training a Model
```python
from trainer import FashionMNISTTrainer

trainer = FashionMNISTTrainer()
model, history = trainer.train_model(
    model_type='simple',  # 'simple', 'deep', 'lightweight'
    epochs=15,
    batch_size=128
)
```

#### Making Predictions
```python
from predictor import FashionMNISTPredictor

predictor = FashionMNISTPredictor('models/my_model.keras')
results = predictor.predict_single_image('my_image.jpg')
print(f"Prediction: {results['predicted_class_name']}")
```

#### Preprocessing Images
```python
from image_preprocessor import ImagePreprocessor

preprocessor = ImagePreprocessor()
model_input, original, processed = preprocessor.preprocess_image(
    'my_image.jpg', 
    show_steps=True
)
```

## 📁 Module Details

### 1. Data Loader (`data_loader.py`)
- **Purpose**: Loads and preprocesses Fashion-MNIST dataset
- **Key Features**:
  - Automatic dataset downloading
  - Data normalization and reshaping
  - Class name mapping
  - Sample extraction for visualization

### 2. Model Builder (`model_builder.py`)
- **Purpose**: Creates CNN architectures
- **Available Models**:
  - **Simple CNN**: Fast training, good for testing
  - **Deep CNN**: Higher accuracy, more parameters
  - **Lightweight CNN**: Fastest inference, lower accuracy
- **Features**: Model compilation, callbacks, saving/loading

### 3. Trainer (`trainer.py`)
- **Purpose**: Handles model training and evaluation
- **Features**:
  - Configurable training parameters
  - Comprehensive evaluation metrics
  - Training visualization (accuracy/loss curves)
  - Confusion matrices
  - Sample prediction visualizations
  - Training reports

### 4. Image Preprocessor (`image_preprocessor.py`)
- **Purpose**: Prepares external images for prediction
- **Key Steps**:
  1. Load any image format/size
  2. Convert to grayscale
  3. Resize to 28×28
  4. Normalize pixel values [0,1]
  5. Add batch dimension
- **Features**: Step-by-step visualization, batch processing, validation

### 5. Predictor (`predictor.py`)
- **Purpose**: Makes predictions on new images
- **Features**:
  - Single image prediction
  - Batch prediction
  - Confidence scoring
  - Prediction visualization
  - Results export
  - Model comparison

### 6. Main Controller (`main.py`)
- **Purpose**: Orchestrates all modules
- **Features**: Interactive menu, error handling, workflow management

## 🎯 Model Architectures

### Simple CNN (Default)
```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(128) → Dense(10)
Parameters: ~34K
Training time: ~2-3 minutes
```

### Deep CNN
```
Conv2D(32) → Conv2D(32) → MaxPool → Dropout →
Conv2D(64) → Conv2D(64) → MaxPool → Dropout →
Dense(256) → Dense(128) → Dense(10)
Parameters: ~200K
Training time: ~5-7 minutes
```

### Lightweight CNN
```
Conv2D(16) → MaxPool → Conv2D(32) → MaxPool → Dense(64) → Dense(10)
Parameters: ~15K
Training time: ~1-2 minutes
```

## 📊 Performance Expectations

| Model | Accuracy | Training Time | Parameters |
|-------|----------|---------------|------------|
| Simple | ~91-93% | 2-3 min | 34K |
| Deep | ~92-94% | 5-7 min | 200K |
| Lightweight | ~89-91% | 1-2 min | 15K |

## 🖼️ Image Preprocessing Pipeline

```
Input Image (any size/format)
        ↓
   Convert to Grayscale
        ↓
    Resize to 28×28
        ↓
   Normalize [0,1]
        ↓
  Add Batch Dimension
        ↓
   Ready for Prediction
```

## 🔧 Usage Scenarios

### 1. Quick Training and Testing
```bash
python main.py
# Choose: 1 (Train new model)
# Select: Simple CNN, 10 epochs
# Then: 3 (Predict single image)
```

### 2. Production Workflow
```bash
# Train best model
python -c "
from trainer import FashionMNISTTrainer
trainer = FashionMNISTTrainer()
trainer.train_model('deep', epochs=20)
"

# Use for predictions
python -c "
from predictor import FashionMNISTPredictor
predictor = FashionMNISTPredictor('models/deep_model.keras')
predictor.predict_single_image('my_image.jpg')
"
```

### 3. Batch Processing
```python
from predictor import FashionMNISTPredictor

predictor = FashionMNISTPredictor('models/my_model.keras')
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predictor.predict_batch(image_paths)
predictor.create_prediction_grid(image_paths, 'results/grid.png')
```

## 📈 Visualization Features

- **Training curves**: Accuracy and loss over epochs
- **Confusion matrices**: Per-class performance analysis
- **Sample predictions**: Visual prediction results
- **Preprocessing steps**: Step-by-step image transformation
- **Prediction grids**: Batch prediction visualization

## 🚨 Common Issues and Solutions

### 1. Low Prediction Accuracy
- **Cause**: Complex real-world images vs simple Fashion-MNIST training data
- **Solution**: Use simple, centered, well-lit clothing images

### 2. Out of Memory Errors
- **Cause**: Large batch size or deep model
- **Solution**: Reduce batch_size or use lightweight model

### 3. Image Preprocessing Warnings
- **Cause**: Unusual image sizes or formats
- **Solution**: Check image with `validate_image_format()` first

## 🔄 Extending the System

### Adding New Model Architecture
```python
# In model_builder.py
def build_custom_cnn(self):
    model = Sequential([
        # Your custom layers here
    ])
    self._compile_model(model)
    return model
```

### Adding New Preprocessing Steps
```python
# In image_preprocessor.py
def custom_preprocess(self, image_path):
    # Your custom preprocessing
    return processed_image
```

## 📝 Output Files

- **Models**: `models/model_name_timestamp.keras`
- **Visualizations**: `results/visualization_name.png`
- **Reports**: `reports/training_report_timestamp.txt`
- **Predictions**: `results/predictions_timestamp.txt`

## 🏆 Best Practices

1. **Start with Simple CNN** for quick testing
2. **Use Deep CNN** for production accuracy
3. **Validate images** before prediction
4. **Save training reports** for model comparison
5. **Use batch prediction** for multiple images
6. **Monitor confidence scores** for prediction quality

## 🤝 Contributing

This modular architecture makes it easy to:
- Add new model architectures
- Implement different preprocessing strategies  
- Extend visualization capabilities
- Add new evaluation metrics

Each module is independent and can be modified without affecting others.

## 📄 License

Open source - feel free to modify and extend!

---

**Happy Classifying! 👗👔👟**