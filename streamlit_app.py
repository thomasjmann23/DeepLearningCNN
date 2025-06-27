"""
Fashion-MNIST CNN Classifier - Streamlit Web Application
A complete web interface for training, evaluating, and using Fashion-MNIST CNN models
"""

import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import tempfile
import io
import base64
from PIL import Image

# Import the existing modules
from data_loader import FashionMNISTDataLoader
from model_builder import FashionMNISTModelBuilder
from trainer import FashionMNISTTrainer
from image_preprocessor import ImagePreprocessor
from predictor import FashionMNISTPredictor
from dimensionality_visualization import DimensionalityVisualizer

# Configure Streamlit page
st.set_page_config(
    page_title="Fashion-MNIST CNN Classifier",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = FashionMNISTDataLoader()
if 'model_builder' not in st.session_state:
    st.session_state.model_builder = FashionMNISTModelBuilder()
if 'trainer' not in st.session_state:
    st.session_state.trainer = FashionMNISTTrainer()
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = ImagePreprocessor()
if 'predictor' not in st.session_state:
    st.session_state.predictor = FashionMNISTPredictor()
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = DimensionalityVisualizer()

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Main title
st.markdown('<h1 class="main-header">Fashion-MNIST CNN Classifier</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Home", "Dataset Info", "Train Model", "Load & Test Model", "Predict Images", "Model Comparison", "Image Preprocessing", "Dimensionality Visualization"] 
)

def display_dataset_info():
    """Display dataset information and sample images"""
    st.markdown('<h2 class="section-header">Dataset Information</h2>', unsafe_allow_html=True)
    
    try:
        # Load data
        with st.spinner("Loading Fashion-MNIST dataset..."):
            x_train, y_train, x_test, y_test = st.session_state.data_loader.load_data()
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Images", f"{x_train.shape[0]:,}")
        with col2:
            st.metric("Test Images", f"{x_test.shape[0]:,}")
        with col3:
            st.metric("Image Size", f"{x_train.shape[1]}x{x_train.shape[2]}")
        with col4:
            st.metric("Classes", len(st.session_state.data_loader.class_names))
        
        # Class names
        st.subheader("Fashion Classes")
        class_df = pd.DataFrame({
            'Index': range(len(st.session_state.data_loader.class_names)),
            'Class Name': st.session_state.data_loader.class_names
        })
        st.dataframe(class_df, use_container_width=True)
        
        # Class distribution
        st.subheader("Class Distribution")
        distribution = st.session_state.data_loader.get_class_distribution(y_train)
        dist_df = pd.DataFrame(list(distribution.items()), columns=['Class', 'Count'])
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(dist_df, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(dist_df['Class'], dist_df['Count'], color='skyblue')
            ax.set_xlabel('Fashion Classes')
            ax.set_ylabel('Number of Images')
            ax.set_title('Training Set Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Sample images
        st.subheader("Sample Images")
        if st.button("Generate New Samples"):
            sample_images, sample_labels = st.session_state.data_loader.get_sample_images(x_train, y_train, 15)
            
            fig, axes = plt.subplots(3, 5, figsize=(15, 9))
            for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
                row, col = i // 5, i % 5
                axes[row, col].imshow(img.reshape(28, 28), cmap='gray')
                axes[row, col].set_title(st.session_state.data_loader.get_class_name(label))
                axes[row, col].axis('off')
            
            plt.suptitle('Fashion-MNIST Sample Images', fontsize=16)
            plt.tight_layout()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

def train_model_page():
    """Model training interface"""
    st.markdown('<h2 class="section-header">Train New Model</h2>', unsafe_allow_html=True)
    
    # Model configuration
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Model Architecture",
            options=['simple', 'deep', 'lightweight'],
            help="Simple: Fast training, good for testing. Deep: Better accuracy, slower training. Lightweight: Fastest, lower accuracy."
        )
        
        epochs = st.slider("Number of Epochs", min_value=1, max_value=50, value=15)
        
    with col2:
        batch_size = st.selectbox("Batch Size", options=[32, 64, 128, 256], index=2)
        
        use_validation_split = st.checkbox("Use Validation Split", value=False)
        if use_validation_split:
            validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.3, value=0.2)
        else:
            validation_split = None
    
    # Model architecture preview
    st.subheader("Model Architecture Preview")
    if st.button("Show Model Architecture"):
        try:
            if model_type == 'simple':
                preview_model = st.session_state.model_builder.build_simple_cnn()
            elif model_type == 'deep':
                preview_model = st.session_state.model_builder.build_deeper_cnn()
            else:
                preview_model = st.session_state.model_builder.build_lightweight_cnn()
            
            # Display model summary
            stringlist = []
            preview_model.summary(print_fn=lambda x: stringlist.append(x))
            model_summary = "\n".join(stringlist)
            st.text(model_summary)
            
            # Model stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Parameters", f"{preview_model.count_params():,}")
            with col2:
                trainable_params = sum([np.prod(w.shape) for w in preview_model.trainable_weights])
                st.metric("Trainable Parameters", f"{trainable_params:,}")
            with col3:
                layers = len(preview_model.layers)
                st.metric("Number of Layers", layers)
                
        except Exception as e:
            st.error(f"Error creating model preview: {e}")
    
    # Training section
    st.subheader("Start Training")
    
    if st.button("Begin Training", type="primary"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training progress callback
            class StreamlitCallback:
                def __init__(self, progress_bar, status_text):
                    self.progress_bar = progress_bar
                    self.status_text = status_text
                    self.epoch = 0
                    self.epochs = epochs
                
                def on_epoch_end(self, epoch, logs=None):
                    self.epoch = epoch + 1
                    progress = self.epoch / self.epochs
                    self.progress_bar.progress(progress)
                    
                    if logs:
                        acc = logs.get('accuracy', 0)
                        val_acc = logs.get('val_accuracy', 0)
                        self.status_text.text(f"Epoch {self.epoch}/{self.epochs} - Accuracy: {acc:.4f} - Val Accuracy: {val_acc:.4f}")
            
            status_text.text("Initializing training...")
            
            # Train model
            model, history = st.session_state.trainer.train_model(
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                save_best=True
            )
            
            # Store in session state
            st.session_state.trained_model = model
            st.session_state.training_history = history
            
            progress_bar.progress(1.0)
            status_text.text("Training completed!")
            
            # Display results
            st.success("Model training completed successfully!")
            
            # Show training curves
            st.subheader("Training Results")
            
            # Plot training history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Accuracy plot
            ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1])
            
            # Loss plot
            ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
            ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Final metrics
            final_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Training Accuracy", f"{final_acc:.4f}")
            with col2:
                st.metric("Final Validation Accuracy", f"{final_val_acc:.4f}")
            with col3:
                st.metric("Final Training Loss", f"{final_loss:.4f}")
            with col4:
                st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
                
        except Exception as e:
            st.error(f"Training failed: {e}")

def load_test_model_page():
    """Load existing model and test it"""
    st.markdown('<h2 class="section-header">Load & Test Model</h2>', unsafe_allow_html=True)
    
    # List available models
    model_dir = 'models'
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        
        if model_files:
            st.subheader("Available Models")
            selected_model = st.selectbox("Choose a model:", model_files)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Model"):
                    model_path = os.path.join(model_dir, selected_model)
                    if st.session_state.predictor.load_model(model_path):
                        st.success(f"Model loaded successfully: {selected_model}")
                        
                        # Show model info
                        st.subheader("Model Information")
                        st.write(f"**Model Name:** {st.session_state.predictor.model.name}")
                        st.write(f"**Input Shape:** {st.session_state.predictor.model.input_shape}")
                        st.write(f"**Output Shape:** {st.session_state.predictor.model.output_shape}")
                        st.write(f"**Parameters:** {st.session_state.predictor.model.count_params():,}")
                    else:
                        st.error("Failed to load model")
            
            with col2:
                if st.button("Test Model on Dataset"):
                    if st.session_state.predictor.model is not None:
                        with st.spinner("Testing model on dataset..."):
                            try:
                                # Load test data
                                x_train, y_train, x_test, y_test = st.session_state.data_loader.load_data()
                                
                                # Evaluate model
                                test_loss, test_accuracy = st.session_state.predictor.model.evaluate(x_test, y_test, verbose=0)
                                
                                # Show results
                                st.subheader("Test Results")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Test Accuracy", f"{test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
                                with col2:
                                    st.metric("Test Loss", f"{test_loss:.4f}")
                                
                                # Get predictions for confusion matrix
                                y_pred_proba = st.session_state.predictor.model.predict(x_test, verbose=0)
                                y_pred = np.argmax(y_pred_proba, axis=1)
                                y_true = np.argmax(y_test, axis=1)
                                
                                # Show sample predictions
                                st.subheader("Sample Predictions")
                                sample_indices = np.random.choice(len(x_test), 12, replace=False)
                                
                                fig, axes = plt.subplots(3, 4, figsize=(12, 9))
                                for i, idx in enumerate(sample_indices):
                                    row, col = i // 4, i % 4
                                    
                                    image = x_test[idx].reshape(28, 28)
                                    true_label = y_true[idx]
                                    pred_label = y_pred[idx]
                                    confidence = y_pred_proba[idx][pred_label]
                                    
                                    axes[row, col].imshow(image, cmap='gray')
                                    
                                    true_class = st.session_state.data_loader.get_class_name(true_label)
                                    pred_class = st.session_state.data_loader.get_class_name(pred_label)
                                    
                                    if true_label == pred_label:
                                        color = 'green'
                                        title = f'Correct: {pred_class}\n{confidence:.2%}'
                                    else:
                                        color = 'red'
                                        title = f'Wrong: {pred_class}\nTrue: {true_class}\n{confidence:.2%}'
                                    
                                    axes[row, col].set_title(title, color=color, fontsize=9)
                                    axes[row, col].axis('off')
                                
                                plt.suptitle('Sample Test Predictions', fontsize=14)
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                            except Exception as e:
                                st.error(f"Error testing model: {e}")
                    else:
                        st.warning("Please load a model first")
        else:
            st.warning("No trained models found in the models directory")
    else:
        st.warning("Models directory not found")

def predict_images_page():
    """Image prediction interface"""
    st.markdown('<h2 class="section-header">Predict Images</h2>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if st.session_state.predictor.model is None:
        st.warning("Please load a model first from the 'Load & Test Model' page")
        return
    
    # Image upload options
    st.subheader("Upload Images for Prediction")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose clothing images",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        accept_multiple_files=True,
        help="Upload one or more clothing images for classification"
    )
    
    if uploaded_files:
        st.subheader("Prediction Results")
        
        # Process each uploaded file
        for i, uploaded_file in enumerate(uploaded_files):
            st.write(f"**Image {i+1}: {uploaded_file.name}**")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Display original image
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    st.write("**Original Image**")
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Size: {image.size}", use_container_width=True)
                
                # Validate image
                validation = st.session_state.preprocessor.validate_image_format(tmp_file_path)
                
                if not validation['valid']:
                    st.error("Image validation failed:")
                    for issue in validation['issues']:
                        st.write(f"- {issue}")
                    continue
                
                if validation['warnings']:
                    with st.expander("Image Warnings"):
                        for warning in validation['warnings']:
                            st.warning(warning)
                
                # Preprocess and predict
                model_input, original_image, processed_image = st.session_state.preprocessor.preprocess_image(tmp_file_path)
                
                with col2:
                    st.write("**Processed (28x28)**")
                    st.image(processed_image, caption="Model Input", use_container_width=True)
                
                # Make prediction
                results = st.session_state.predictor.predict_single_image(
                    tmp_file_path, 
                    show_visualization=False,
                    confidence_threshold=0.3
                )
                
                with col3:
                    st.write("**Prediction Results**")
                    
                    # Main prediction
                    confidence = results['confidence']
                    predicted_class = results['predicted_class_name']
                    
                    if confidence >= 0.7:
                        st.success(f"**Prediction:** {predicted_class}")
                        st.success(f"**Confidence:** {confidence:.2%}")
                    elif confidence >= 0.4:
                        st.warning(f"**Prediction:** {predicted_class}")
                        st.warning(f"**Confidence:** {confidence:.2%}")
                    else:
                        st.error(f"**Prediction:** {predicted_class}")
                        st.error(f"**Confidence:** {confidence:.2%} (Low confidence)")
                    
                    # Top 3 predictions
                    st.write("**Top 3 Predictions:**")
                    probabilities = results['all_probabilities']
                    top_3_indices = np.argsort(probabilities)[::-1][:3]
                    
                    for j, idx in enumerate(top_3_indices):
                        prob = probabilities[idx]
                        class_name = st.session_state.data_loader.class_names[idx]
                        st.write(f"{j+1}. {class_name}: {prob:.2%}")
                    
                    # Probability bar chart
                    st.write("**All Class Probabilities:**")
                    prob_df = pd.DataFrame({
                        'Class': st.session_state.data_loader.class_names,
                        'Probability': probabilities
                    })
                    st.bar_chart(prob_df.set_index('Class'))
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

def model_comparison_page():
    """Compare different trained models"""
    st.markdown('<h2 class="section-header">Model Comparison</h2>', unsafe_allow_html=True)
    
    # List available models
    model_dir = 'models'
    if not os.path.exists(model_dir):
        st.warning("No models directory found")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    if len(model_files) < 2:
        st.warning("Need at least 2 models to compare")
        return
    
    # Model selection
    st.subheader("Select Models to Compare")
    selected_models = st.multiselect(
        "Choose models:",
        options=model_files,
        default=model_files[:2] if len(model_files) >= 2 else model_files
    )
    
    if len(selected_models) < 2:
        st.warning("Please select at least 2 models")
        return
    
    if st.button("Compare Selected Models"):
        with st.spinner("Comparing models..."):
            try:
                # Load test data
                x_train, y_train, x_test, y_test = st.session_state.data_loader.load_data()
                
                comparison_results = []
                
                for model_file in selected_models:
                    model_path = os.path.join(model_dir, model_file)
                    
                    # Load model
                    if st.session_state.predictor.load_model(model_path):
                        # Evaluate on test set
                        test_loss, test_accuracy = st.session_state.predictor.model.evaluate(x_test, y_test, verbose=0)
                        
                        comparison_results.append({
                            'Model': model_file,
                            'Test Accuracy': test_accuracy,
                            'Test Loss': test_loss,
                            'Parameters': st.session_state.predictor.model.count_params()
                        })
                
                # Display comparison table
                st.subheader("Comparison Results")
                comparison_df = pd.DataFrame(comparison_results)
                comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
                
                # Format the dataframe for better display
                comparison_df['Test Accuracy'] = comparison_df['Test Accuracy'].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
                comparison_df['Test Loss'] = comparison_df['Test Loss'].apply(lambda x: f"{x:.4f}")
                comparison_df['Parameters'] = comparison_df['Parameters'].apply(lambda x: f"{x:,}")
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualization
                st.subheader("Performance Comparison")
                
                # Extract numeric values for plotting
                numeric_results = []
                for result in comparison_results:
                    numeric_results.append({
                        'Model': result['Model'],
                        'Accuracy': result['Test Accuracy'],
                        'Loss': result['Test Loss'],
                        'Parameters': result['Parameters']
                    })
                
                plot_df = pd.DataFrame(numeric_results)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(plot_df['Model'], plot_df['Accuracy'], color='skyblue')
                    ax.set_xlabel('Models')
                    ax.set_ylabel('Test Accuracy')
                    ax.set_title('Model Accuracy Comparison')
                    ax.set_ylim(0, 1)
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar, acc in zip(bars, plot_df['Accuracy']):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{acc:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(plot_df['Parameters'], plot_df['Accuracy'], 
                              s=100, c='orange', alpha=0.7)
                    
                    for i, model in enumerate(plot_df['Model']):
                        ax.annotate(model, (plot_df['Parameters'].iloc[i], plot_df['Accuracy'].iloc[i]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
                    
                    ax.set_xlabel('Number of Parameters')
                    ax.set_ylabel('Test Accuracy')
                    ax.set_title('Accuracy vs Model Size')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error comparing models: {e}")

def image_preprocessing_page():
    """Image preprocessing visualization"""
    st.markdown('<h2 class="section-header">Image Preprocessing</h2>', unsafe_allow_html=True)
    
    st.write("Upload an image to see how it gets preprocessed for the Fashion-MNIST model.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload a clothing image to see the preprocessing steps"
    )
    
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Validate image
            st.subheader("Image Validation")
            validation = st.session_state.preprocessor.validate_image_format(tmp_file_path)
            
            col1, col2 = st.columns(2)
            with col1:
                if validation['valid']:
                    st.success("Image is valid for processing")
                else:
                    st.error("Image validation failed")
                    for issue in validation['issues']:
                        st.write(f"- {issue}")
            
            with col2:
                if validation['warnings']:
                    st.warning("Warnings:")
                    for warning in validation['warnings']:
                        st.write(f"- {warning}")
            
            if validation['valid']:
                # Preprocess image
                st.subheader("Preprocessing Steps")
                model_input, original_image, processed_image = st.session_state.preprocessor.preprocess_image(tmp_file_path)
                
                # Show preprocessing pipeline
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Step 1: Original Image**")
                    st.image(original_image, caption=f"Size: {original_image.size}, Mode: {original_image.mode}", use_container_width=True)
                
                with col2:
                    st.write("**Step 2: Processed (28x28)**")
                    st.image(processed_image, caption="Grayscale, Resized, Background Corrected", use_container_width=True)
                
                with col3:
                    st.write("**Step 3: Model Input**")
                    # Show the normalized array as an image
                    normalized_array = model_input[0].reshape(28, 28)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    im = ax.imshow(normalized_array, cmap='gray')
                    ax.set_title('Normalized [0,1]')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig)
                
                # Detailed analysis
                st.subheader("Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pixel statistics
                    st.write("**Pixel Statistics**")
                    stats_df = pd.DataFrame({
                        'Statistic': ['Min Value', 'Max Value', 'Mean', 'Std Dev'],
                        'Value': [
                            f"{normalized_array.min():.3f}",
                            f"{normalized_array.max():.3f}",
                            f"{normalized_array.mean():.3f}",
                            f"{normalized_array.std():.3f}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Model input shape
                    st.write("**Model Input Shape**")
                    st.write(f"Shape: {model_input.shape}")
                    st.write(f"Data type: {model_input.dtype}")
                    st.write(f"Memory usage: {model_input.nbytes} bytes")
                
                with col2:
                    # Pixel distribution histogram
                    st.write("**Pixel Intensity Distribution**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(normalized_array.flatten(), bins=30, alpha=0.7, color='blue', edgecolor='black')
                    ax.set_xlabel('Pixel Intensity (0-1)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Pixel Values')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Pixel grid visualization
                st.subheader("Pixel-Level View")
                show_grid = st.checkbox("Show pixel grid overlay")
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(normalized_array, cmap='gray', interpolation='nearest')
                ax.set_title('28x28 Pixel Grid View')
                
                if show_grid:
                    # Add grid lines
                    for i in range(29):
                        ax.axhline(i-0.5, color='red', linewidth=0.3, alpha=0.7)
                        ax.axvline(i-0.5, color='red', linewidth=0.3, alpha=0.7)
                
                ax.set_xticks(range(0, 28, 4))
                ax.set_yticks(range(0, 28, 4))
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

def dimensionality_visualization_page():
    """Dimensionality reduction visualization interface"""
    st.markdown('<h2 class="section-header">Dimensionality Visualization (t-SNE & UMAP)</h2>', unsafe_allow_html=True)
    
    st.write("""
    Visualize the Fashion-MNIST dataset and your predictions in 2D space using t-SNE and UMAP. 
    This helps understand how the model sees different clothing items and where your predictions fit.
    """)
    
    # Configuration section
    st.subheader("Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        viz_type = st.selectbox(
            "Visualization Type",
            ["Dataset Only", "Dataset + Predictions"],
            help="Choose whether to visualize just the dataset or include your prediction images"
        )
        
        embedding_method = st.selectbox(
            "Embedding Method",
            ["Both t-SNE and UMAP", "t-SNE only", "UMAP only"],
            help="t-SNE preserves local structure, UMAP preserves both local and global structure"
        )
    
    with col2:
        n_dataset_samples = st.slider(
            "Dataset Samples",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500,
            help="Number of Fashion-MNIST samples to include (more samples = longer computation)"
        )
        
        feature_method = st.selectbox(
            "Feature Extraction",
            ["PCA (Recommended)", "Pixel Flattening", "Model Features"],
            help="Method to extract features before dimensionality reduction"
        )
    
    # Advanced parameters
    with st.expander("Advanced Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**t-SNE Parameters**")
            tsne_perplexity = st.slider("Perplexity", 5, 100, 30)
            tsne_iterations = st.slider("Iterations", 500, 2000, 1000)
        
        with col2:
            st.write("**UMAP Parameters**")
            umap_neighbors = st.slider("N Neighbors", 5, 50, 15)
            umap_min_dist = st.slider("Min Distance", 0.01, 0.5, 0.1)
    
    # Dataset-only visualization
    if viz_type == "Dataset Only":
        st.subheader("Dataset Visualization")
        
        if st.button("Generate Dataset Visualization", type="primary"):
            with st.spinner("Preparing dataset and computing embeddings..."):
                try:
                    # Update visualizer model if available
                    if st.session_state.predictor.model is not None:
                        st.session_state.visualizer.model = st.session_state.predictor.model
                    
                    # Prepare dataset
                    images, labels = st.session_state.visualizer.prepare_dataset_sample(n_dataset_samples)
                    
                    # Map feature method names
                    feature_map = {
                        "PCA (Recommended)": "pca",
                        "Pixel Flattening": "flatten", 
                        "Model Features": "model"
                    }
                    feature_method_key = feature_map[feature_method]
                    
                    # Extract features
                    features = st.session_state.visualizer.extract_features(images, method=feature_method_key)
                    
                    # Initialize embedding variable
                    embedding_for_analysis = None
                    
                    # Create embeddings and visualizations
                    if embedding_method in ["Both t-SNE and UMAP", "t-SNE only"]:
                        st.write("**t-SNE Visualization**")
                        with st.spinner("Computing t-SNE embedding..."):
                            tsne_embedding = st.session_state.visualizer.compute_tsne(
                                features, perplexity=tsne_perplexity, n_iter=tsne_iterations
                            )
                            
                            # Set this as the embedding for analysis
                            embedding_for_analysis = tsne_embedding
                            
                            # Show t-SNE plot
                            fig, ax = plt.subplots(figsize=(12, 10))
                            
                            for class_idx in range(10):
                                mask = labels == class_idx
                                if np.any(mask):
                                    ax.scatter(
                                        tsne_embedding[mask, 0], 
                                        tsne_embedding[mask, 1],
                                        c=st.session_state.visualizer.class_colors[class_idx],
                                        label=st.session_state.data_loader.class_names[class_idx],
                                        alpha=0.6,
                                        s=20
                                    )
                            
                            ax.set_title('t-SNE Visualization: Fashion-MNIST Dataset', fontsize=16, fontweight='bold')
                            ax.set_xlabel('t-SNE Dimension 1')
                            ax.set_ylabel('t-SNE Dimension 2')
                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()  # Important: close the figure to free memory
                    
                    if embedding_method in ["Both t-SNE and UMAP", "UMAP only"]:
                        st.write("**UMAP Visualization**")
                        with st.spinner("Computing UMAP embedding..."):
                            umap_embedding = st.session_state.visualizer.compute_umap(
                                features, n_neighbors=umap_neighbors, min_dist=umap_min_dist
                            )
                            
                            # If t-SNE wasn't computed, use UMAP for analysis
                            if embedding_for_analysis is None:
                                embedding_for_analysis = umap_embedding
                            
                            # Show UMAP plot
                            fig, ax = plt.subplots(figsize=(12, 10))
                            
                            for class_idx in range(10):
                                mask = labels == class_idx
                                if np.any(mask):
                                    ax.scatter(
                                        umap_embedding[mask, 0], 
                                        umap_embedding[mask, 1],
                                        c=st.session_state.visualizer.class_colors[class_idx],
                                        label=st.session_state.data_loader.class_names[class_idx],
                                        alpha=0.6,
                                        s=20
                                    )
                            
                            ax.set_title('UMAP Visualization: Fashion-MNIST Dataset', fontsize=16, fontweight='bold')
                            ax.set_xlabel('UMAP Dimension 1')
                            ax.set_ylabel('UMAP Dimension 2')
                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()  # Important: close the figure to free memory
                    
                    # Class separation analysis - only if we have an embedding
                    if embedding_for_analysis is not None:
                        st.subheader("Class Separation Analysis")
                        with st.spinner("Analyzing class separation..."):
                            from sklearn.metrics import silhouette_score
                            silhouette_avg = silhouette_score(embedding_for_analysis, labels)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                            with col2:
                                unique_labels = len(np.unique(labels))
                                st.metric("Classes Visualized", unique_labels)
                            with col3:
                                st.metric("Total Points", len(labels))
                            
                            # Distance analysis
                            within_class_distances = []
                            between_class_distances = []
                            
                            for class_idx in range(10):
                                class_mask = labels == class_idx
                                class_points = embedding_for_analysis[class_mask]
                                
                                if len(class_points) > 1:
                                    from scipy.spatial.distance import pdist
                                    within_distances = pdist(class_points)
                                    within_class_distances.extend(within_distances)
                                    
                                    other_points = embedding_for_analysis[~class_mask]
                                    for point in class_points:
                                        distances_to_others = np.linalg.norm(other_points - point, axis=1)
                                        between_class_distances.extend(distances_to_others[:100])  # Sample to avoid memory issues
                            
                            if within_class_distances and between_class_distances:
                                separation_ratio = np.mean(between_class_distances) / np.mean(within_class_distances)
                                
                                st.write("**Distance Statistics:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Avg Within-Class Distance", f"{np.mean(within_class_distances):.3f}")
                                with col2:
                                    st.metric("Avg Between-Class Distance", f"{np.mean(between_class_distances):.3f}")
                                with col3:
                                    st.metric("Separation Ratio", f"{separation_ratio:.3f}")
                                
                                # Distance distribution plot
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.hist(within_class_distances, bins=50, alpha=0.7, label='Within-class', density=True)
                                ax.hist(between_class_distances, bins=50, alpha=0.7, label='Between-class', density=True)
                                ax.set_xlabel('Distance')
                                ax.set_ylabel('Density')
                                ax.set_title('Distance Distributions')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close()  # Add this line after st.pyplot(fig)
                        
                        st.success("Dataset visualization completed!")
                        
                except Exception as e:
                    st.error(f"Error creating visualization: {e}")
    
    # Dataset + Predictions visualization
    else:
        st.subheader("Dataset + Predictions Visualization")
        
        # Check if model is loaded
        if st.session_state.predictor.model is None:
            st.warning("Please load a model first from the 'Load & Test Model' page")
            return
        
        # File uploader for prediction images
        uploaded_files = st.file_uploader(
            "Upload clothing images for prediction visualization",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            accept_multiple_files=True,
            help="Upload clothing images to see where they appear in the dataset visualization"
        )
        
        if uploaded_files and st.button("Generate Visualization with Predictions", type="primary"):
            with st.spinner("Processing images and computing embeddings..."):
                try:
                    # Save uploaded files temporarily
                    temp_paths = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_paths.append(tmp_file.name)
                    
                    # Update visualizer model
                    st.session_state.visualizer.model = st.session_state.predictor.model
                    
                    # Map embedding method
                    embedding_map = {
                        "Both t-SNE and UMAP": "both",
                        "t-SNE only": "tsne",
                        "UMAP only": "umap"
                    }
                    
                    # Map feature method
                    feature_map = {
                        "PCA (Recommended)": "pca",
                        "Pixel Flattening": "flatten",
                        "Model Features": "model"
                    }
                    
                    # Create visualization with predictions
                    st.write("**Processing predictions and creating visualization...**")
                    
                    # Update visualizer model
                    st.session_state.visualizer.model = st.session_state.predictor.model
                    
                    # Prepare dataset sample
                    dataset_images, dataset_labels = st.session_state.visualizer.prepare_dataset_sample(n_dataset_samples)
                    
                    # Process prediction images
                    prediction_images = []
                    prediction_labels = []
                    
                    for image_path in temp_paths:
                        try:
                            model_input, _, _ = st.session_state.preprocessor.preprocess_image(image_path)
                            prediction_images.append(model_input[0])
                            
                            pred_proba = st.session_state.predictor.model.predict(model_input, verbose=0)
                            pred_class_idx = np.argmax(pred_proba[0])
                            pred_class_name = st.session_state.data_loader.class_names[pred_class_idx]
                            prediction_labels.append(pred_class_name)
                        except Exception as e:
                            st.error(f"Error processing image: {e}")
                            continue
                    
                    if prediction_images:
                        prediction_images = np.array(prediction_images)
                        
                        # Combine images for feature extraction
                        all_images = np.concatenate([dataset_images, prediction_images])
                        all_features = st.session_state.visualizer.extract_features(all_images, method=feature_map[feature_method])
                        
                        dataset_features = all_features[:len(dataset_images)]
                        prediction_features = all_features[len(dataset_images):]
                        
                        # Create embeddings based on method
                        if embedding_map[embedding_method] in ['tsne', 'both']:
                            st.write("**t-SNE with Predictions**")
                            tsne_embedding = st.session_state.visualizer.compute_tsne(all_features)
                            dataset_tsne = tsne_embedding[:len(dataset_images)]
                            prediction_tsne = tsne_embedding[len(dataset_images):]
                            
                            # Create t-SNE plot
                            fig, ax = plt.subplots(figsize=(12, 10))
                            
                            # Plot dataset points
                            for class_idx in range(10):
                                mask = dataset_labels == class_idx
                                if np.any(mask):
                                    ax.scatter(
                                        dataset_tsne[mask, 0], 
                                        dataset_tsne[mask, 1],
                                        c=st.session_state.visualizer.class_colors[class_idx],
                                        label=st.session_state.data_loader.class_names[class_idx],
                                        alpha=0.6,
                                        s=20
                                    )
                            
                            # Plot prediction points
                            ax.scatter(
                                prediction_tsne[:, 0],
                                prediction_tsne[:, 1],
                                c='black',
                                marker='X',
                                s=200,
                                label='Your Predictions',
                                edgecolors='white',
                                linewidth=2
                            )
                            
                            ax.set_title('t-SNE: Dataset + Your Predictions', fontsize=16, fontweight='bold')
                            ax.set_xlabel('t-SNE Dimension 1')
                            ax.set_ylabel('t-SNE Dimension 2')
                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        if embedding_map[embedding_method] in ['umap', 'both']:
                            st.write("**UMAP with Predictions**") 
                            umap_embedding = st.session_state.visualizer.compute_umap(all_features)
                            dataset_umap = umap_embedding[:len(dataset_images)]
                            prediction_umap = umap_embedding[len(dataset_images):]
                            
                            # Create UMAP plot
                            fig, ax = plt.subplots(figsize=(12, 10))
                            
                            # Plot dataset points  
                            for class_idx in range(10):
                                mask = dataset_labels == class_idx
                                if np.any(mask):
                                    ax.scatter(
                                        dataset_umap[mask, 0], 
                                        dataset_umap[mask, 1],
                                        c=st.session_state.visualizer.class_colors[class_idx],
                                        label=st.session_state.data_loader.class_names[class_idx],
                                        alpha=0.6,
                                        s=20
                                    )
                            
                            # Plot prediction points
                            ax.scatter(
                                prediction_umap[:, 0],
                                prediction_umap[:, 1], 
                                c='black',
                                marker='X',
                                s=200,
                                label='Your Predictions',
                                edgecolors='white',
                                linewidth=2
                            )
                            
                            ax.set_title('UMAP: Dataset + Your Predictions', fontsize=16, fontweight='bold')
                            ax.set_xlabel('UMAP Dimension 1') 
                            ax.set_ylabel('UMAP Dimension 2')
                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.error("No valid prediction images to visualize")
                    
                    # Clean up temporary files
                    for temp_path in temp_paths:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                    
                    st.success("Visualization with predictions completed!")
                    
                except Exception as e:
                    st.error(f"Error creating visualization: {e}")
                    # Clean up temporary files on error
                    for temp_path in temp_paths:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
    
    # Information section
    st.subheader("Understanding the Visualizations")
    
    with st.expander("What do these visualizations show?"):
        st.write("""
        **t-SNE (t-Distributed Stochastic Neighbor Embedding):**
        - Focuses on preserving local neighborhoods
        - Similar items cluster together
        - Good for finding local patterns and clusters
        - Perplexity parameter controls local vs global structure balance
        
        **UMAP (Uniform Manifold Approximation and Projection):**
        - Preserves both local and global structure
        - Often faster than t-SNE
        - Better at maintaining global relationships
        - Good for understanding overall data structure
        
        **Feature Extraction Methods:**
        - **PCA**: Reduces dimensionality while preserving variance (recommended)
        - **Pixel Flattening**: Uses raw pixel values (may be noisy)
        - **Model Features**: Uses learned features from your trained model (best if model is good)
        """)
    
    with st.expander("How to interpret the results"):
        st.write("""
        **Good Prediction Signs:**
        - Your prediction points (black X's) appear close to similar items
        - Predictions cluster with the correct class
        - Neighborhood analysis shows majority of neighbors are the same class
        
        **Potential Issues:**
        - Prediction points appear isolated or in wrong neighborhoods
        - Low confidence predictions often appear at class boundaries
        - Outlier predictions may indicate image preprocessing issues
        
        **Class Separation Metrics:**
        - **Silhouette Score**: Higher is better (closer to 1)
        - **Separation Ratio**: Higher means better class separation
        - **Distance Distributions**: Should show clear separation between within/between class distances
        """)

def home_page():
    """Home page with overview and quick stats"""
    st.markdown('<h2 class="section-header">Welcome to Fashion-MNIST CNN Classifier</h2>', unsafe_allow_html=True)
    
    # Overview
    st.write("""
    This application provides a complete web interface for training, evaluating, and using 
    Convolutional Neural Network (CNN) models on the Fashion-MNIST dataset. The Fashion-MNIST 
    dataset contains 70,000 grayscale images of clothing items across 10 categories.
    """)
    
    # Quick stats
    st.subheader("Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Count available models
        model_count = 0
        if os.path.exists('models'):
            model_count = len([f for f in os.listdir('models') if f.endswith('.keras')])
        st.metric("Available Models", model_count)
    
    with col2:
        # Check if model is loaded
        model_loaded = "Yes" if st.session_state.predictor.model is not None else "No"
        st.metric("Model Loaded", model_loaded)
    
    with col3:
        # Count results
        result_count = 0
        if os.path.exists('results'):
            result_count = len(os.listdir('results'))
        st.metric("Generated Results", result_count)
    
    with col4:
        # Count reports
        report_count = 0
        if os.path.exists('reports'):
            report_count = len(os.listdir('reports'))
        st.metric("Training Reports", report_count)
    
    # Features overview
    st.subheader("Available Features")
    
    features = [
        ("Dataset Info", "Explore the Fashion-MNIST dataset with sample images and class distributions"),
        ("Train Model", "Train new CNN models with different architectures and hyperparameters"),
        ("Load & Test Model", "Load existing models and evaluate their performance"),
        ("Predict Images", "Upload clothing images and get real-time predictions"),
        ("Model Comparison", "Compare performance across different trained models"),
        ("Image Preprocessing", "Visualize how images are processed for the model")
    ]
    
    for feature, description in features:
        with st.expander(f"📋 {feature}"):
            st.write(description)
    
    # Getting started
    st.subheader("Getting Started")
    st.write("""
    1. **Explore the Dataset**: Start by visiting the 'Dataset Info' page to understand the Fashion-MNIST data
    2. **Train a Model**: Go to 'Train Model' to create your first CNN classifier
    3. **Make Predictions**: Once you have a trained model, use 'Predict Images' to classify clothing items
    4. **Compare Models**: Use 'Model Comparison' to evaluate different architectures
    """)
    
    # System info
    with st.expander("System Information"):
        import tensorflow as tf
        st.write(f"**TensorFlow Version:** {tf.__version__}")
        
        # Check for GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            st.write(f"**GPU Available:** Yes ({len(gpus)} device(s))")
        else:
            st.write("**GPU Available:** No (running on CPU)")
        
        st.write(f"**Python Version:** {os.sys.version}")

# Main app logic
if page == "Home":
    home_page()
elif page == "Dataset Info":
    display_dataset_info()
elif page == "Train Model":
    train_model_page()
elif page == "Load & Test Model":
    load_test_model_page()
elif page == "Predict Images":
    predict_images_page()
elif page == "Model Comparison":
    model_comparison_page()
elif page == "Image Preprocessing":
    image_preprocessing_page()
elif page == "Dimensionality Visualization":  # ADD THIS LINE
    dimensionality_visualization_page()        # ADD THIS LINE

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;'>
        Fashion-MNIST CNN Classifier | Built with Streamlit | 
        <a href='https://github.com/zalandoresearch/fashion-mnist' target='_blank'>Fashion-MNIST Dataset</a>
    </div>
    """, 
    unsafe_allow_html=True
)