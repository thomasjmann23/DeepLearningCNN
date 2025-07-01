"""
Dimensionality Reduction Visualization Module
Uses t-SNE and UMAP to visualize Fashion-MNIST dataset and prediction locations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import pandas as pd
from datetime import datetime
import os

from data_loader import FashionMNISTDataLoader
from image_preprocessor import ImagePreprocessor


class DimensionalityVisualizer:
    """Handles dimensionality reduction and visualization of Fashion-MNIST data"""
    
    def __init__(self, model=None):
        self.model = model
        self.data_loader = FashionMNISTDataLoader()
        self.preprocessor = ImagePreprocessor()
        
        # Pre-computed embeddings storage
        self.dataset_embeddings = {}
        self.dataset_labels = None
        self.dataset_images = None
        
        # Color palette for classes
        self.class_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
    
    def extract_features(self, images, method='flatten'):
        """
        Extract features from images for dimensionality reduction
        
        Args:
            images: Array of images (N, 28, 28, 1)
            method: Feature extraction method ('flatten', 'model', 'pca')
            
        Returns:
            numpy.ndarray: Feature vectors
        """
        if method == 'flatten':
            # Simple pixel flattening
            return images.reshape(len(images), -1)
        
        elif method == 'model' and self.model is not None:
            # Use model's intermediate layer as features
            # Get features from the layer before final classification
            feature_extractor = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.layers[-3].output  # Layer before final dense layers
            )
            return feature_extractor.predict(images, verbose=0)
        
        elif method == 'pca':
            # PCA preprocessing to reduce dimensionality before t-SNE/UMAP
            flattened = images.reshape(len(images), -1)
            pca = PCA(n_components=50, random_state=42)
            return pca.fit_transform(flattened)
        
        else:
            # Default to flattening
            return images.reshape(len(images), -1)
    
    def prepare_dataset_sample(self, n_samples=2000, stratified=True):
        """
        Prepare a sample from the Fashion-MNIST dataset
        
        Args:
            n_samples: Number of samples to use
            stratified: Whether to sample equally from each class
            
        Returns:
            tuple: (sample_images, sample_labels)
        """
        print(f"Preparing dataset sample ({n_samples} images)...")
        
        # Load full dataset
        x_train, y_train, x_test, y_test = self.data_loader.load_data()
        
        # Combine train and test for better representation
        all_images = np.concatenate([x_train, x_test])
        all_labels = np.concatenate([
            np.argmax(y_train, axis=1),
            np.argmax(y_test, axis=1)
        ])
        
        if stratified:
            # Sample equally from each class
            samples_per_class = n_samples // 10
            sample_indices = []
            
            for class_idx in range(10):
                class_indices = np.where(all_labels == class_idx)[0]
                selected = np.random.choice(
                    class_indices, 
                    min(samples_per_class, len(class_indices)), 
                    replace=False
                )
                sample_indices.extend(selected)
            
            sample_indices = np.array(sample_indices)
        else:
            # Random sampling
            sample_indices = np.random.choice(len(all_images), n_samples, replace=False)
        
        self.dataset_images = all_images[sample_indices]
        self.dataset_labels = all_labels[sample_indices]
        
        print(f"✓ Prepared {len(self.dataset_images)} samples")
        return self.dataset_images, self.dataset_labels
    
    def compute_tsne(self, features, perplexity=30, max_iter=1000, random_state=42):
        """
        Compute t-SNE embedding
        
        Args:
            features: Feature vectors
            perplexity: t-SNE perplexity parameter
            max_iter: Maximum number of iterations
            random_state: Random seed
            
        Returns:
            numpy.ndarray: 2D t-SNE embedding
        """
        print(f"Computing t-SNE embedding (perplexity={perplexity})...")
        
        # Handle both old and new parameter names for compatibility
        try:
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                max_iter=max_iter,
                random_state=random_state,
                verbose=1
            )
        except TypeError:
            # Fallback for older scikit-learn versions
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                n_iter=max_iter,
                random_state=random_state,
                verbose=1
            )
        
        embedding = tsne.fit_transform(features)
        print("✓ t-SNE completed")
        return embedding
    
    def compute_umap(self, features, n_neighbors=15, min_dist=0.1, random_state=42):
        """
        Compute UMAP embedding
        
        Args:
            features: Feature vectors
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            random_state: Random seed
            
        Returns:
            numpy.ndarray: 2D UMAP embedding
        """
        print(f"Computing UMAP embedding (n_neighbors={n_neighbors})...")
        
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            verbose=True
        )
        
        embedding = reducer.fit_transform(features)
        print("✓ UMAP completed")
        return embedding
    
    def create_visualization(self, embedding, labels, title, prediction_points=None, 
                           prediction_labels=None, save_path=None):
        """
        Create visualization of the embedding
        
        Args:
            embedding: 2D embedding coordinates
            labels: Class labels for dataset points
            title: Plot title
            prediction_points: Coordinates of prediction points (optional)
            prediction_labels: Labels for prediction points (optional)
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 10))
        
        # Plot dataset points
        for class_idx in range(10):
            mask = labels == class_idx
            if np.any(mask):
                plt.scatter(
                    embedding[mask, 0], 
                    embedding[mask, 1],
                    c=self.class_colors[class_idx],
                    label=self.data_loader.class_names[class_idx],
                    alpha=0.6,
                    s=20
                )
        
        # Plot prediction points if provided
        if prediction_points is not None and len(prediction_points) > 0:
            plt.scatter(
                prediction_points[:, 0],
                prediction_points[:, 1],
                c='black',
                marker='X',
                s=200,
                label='Predictions',
                edgecolors='white',
                linewidth=2
            )
            
            # Add labels for prediction points
            if prediction_labels is not None:
                for i, (point, pred_label) in enumerate(zip(prediction_points, prediction_labels)):
                    plt.annotate(
                        f'Pred {i+1}: {pred_label}',
                        (point[0], point[1]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                    )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        
        # Legend with smaller font
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        plt.show()
    
    def analyze_prediction_neighbors(self, embedding, dataset_labels, prediction_points, 
                                   prediction_labels, k=5):
        """
        Analyze nearest neighbors of prediction points in the embedding space
        
        Args:
            embedding: Dataset embedding coordinates
            dataset_labels: Dataset labels
            prediction_points: Prediction point coordinates
            prediction_labels: Predicted labels
            k: Number of nearest neighbors to analyze
        """
        from sklearn.neighbors import NearestNeighbors
        
        print(f"\nAnalyzing {k} nearest neighbors for each prediction:")
        print("=" * 60)
        
        # Fit nearest neighbors on dataset embedding
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nbrs.fit(embedding)
        
        for i, (pred_point, pred_label) in enumerate(zip(prediction_points, prediction_labels)):
            # Find nearest neighbors
            distances, indices = nbrs.kneighbors([pred_point])
            
            print(f"\nPrediction {i+1}: {pred_label}")
            print("-" * 30)
            
            # Analyze neighbor classes
            neighbor_labels = dataset_labels[indices[0]]
            neighbor_classes = [self.data_loader.class_names[label] for label in neighbor_labels]
            
            print("Nearest neighbors:")
            for j, (dist, neighbor_class) in enumerate(zip(distances[0], neighbor_classes)):
                print(f"  {j+1}. {neighbor_class} (distance: {dist:.3f})")
            
            # Calculate class distribution among neighbors
            unique_classes, counts = np.unique(neighbor_labels, return_counts=True)
            print(f"\nNeighbor class distribution:")
            for class_idx, count in zip(unique_classes, counts):
                class_name = self.data_loader.class_names[class_idx]
                percentage = (count / k) * 100
                print(f"  {class_name}: {count}/{k} ({percentage:.1f}%)")
            
            # Check if prediction matches majority of neighbors
            most_common_class = unique_classes[np.argmax(counts)]
            most_common_name = self.data_loader.class_names[most_common_class]
            
            if most_common_name == pred_label:
                print(f"  ✓ Prediction matches neighborhood majority ({most_common_name})")
            else:
                print(f"  ⚠ Prediction differs from majority (predicted: {pred_label}, majority: {most_common_name})")
    
    def visualize_predictions(self, image_paths, model, feature_method='pca', 
                            embedding_method='both', n_dataset_samples=2000):
        """
        Main method to visualize predictions in embedding space
        
        Args:
            image_paths: List of paths to prediction images
            model: Trained model for feature extraction
            feature_method: Method for feature extraction
            embedding_method: 'tsne', 'umap', or 'both'
            n_dataset_samples: Number of dataset samples to include
        """
        self.model = model
        
        # Prepare dataset sample
        dataset_images, dataset_labels = self.prepare_dataset_sample(n_dataset_samples)
        
        # Process prediction images
        print(f"Processing {len(image_paths)} prediction images...")
        prediction_images = []
        prediction_labels = []
        
        for image_path in image_paths:
            try:
                # Preprocess image
                model_input, _, _ = self.preprocessor.preprocess_image(image_path)
                prediction_images.append(model_input[0])  # Remove batch dimension
                
                # Get prediction
                pred_proba = model.predict(model_input, verbose=0)
                pred_class_idx = np.argmax(pred_proba[0])
                pred_class_name = self.data_loader.class_names[pred_class_idx]
                prediction_labels.append(pred_class_name)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        if not prediction_images:
            print("No valid prediction images to process")
            return
        
        prediction_images = np.array(prediction_images)
        print(f"✓ Processed {len(prediction_images)} prediction images")
        
        # Combine dataset and prediction images for consistent feature extraction
        all_images = np.concatenate([dataset_images, prediction_images])
        
        # Extract features
        print(f"Extracting features using method: {feature_method}")
        all_features = self.extract_features(all_images, method=feature_method)
        
        # Split back into dataset and prediction features
        dataset_features = all_features[:len(dataset_images)]
        prediction_features = all_features[len(dataset_images):]
        
        # Create embeddings
        if embedding_method in ['tsne', 'both']:
            print("\n" + "="*50)
            print("Creating t-SNE Visualization")
            print("="*50)
            
            tsne_embedding = self.compute_tsne(all_features, perplexity=30, max_iter=1000)
            dataset_tsne = tsne_embedding[:len(dataset_images)]
            prediction_tsne = tsne_embedding[len(dataset_images):]
            
            # Create t-SNE visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tsne_path = f"results/tsne_predictions_{timestamp}.png"
            
            self.create_visualization(
                dataset_tsne, 
                dataset_labels,
                't-SNE Visualization: Dataset + Predictions',
                prediction_tsne,
                prediction_labels,
                tsne_path
            )
            
            # Analyze neighbors
            self.analyze_prediction_neighbors(
                dataset_tsne, dataset_labels, 
                prediction_tsne, prediction_labels
            )
        
        if embedding_method in ['umap', 'both']:
            print("\n" + "="*50)
            print("Creating UMAP Visualization")
            print("="*50)
            
            umap_embedding = self.compute_umap(all_features)
            dataset_umap = umap_embedding[:len(dataset_images)]
            prediction_umap = umap_embedding[len(dataset_images):]
            
            # Create UMAP visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            umap_path = f"results/umap_predictions_{timestamp}.png"
            
            self.create_visualization(
                dataset_umap,
                dataset_labels,
                'UMAP Visualization: Dataset + Predictions',
                prediction_umap,
                prediction_labels,
                umap_path
            )
            
            # Analyze neighbors
            self.analyze_prediction_neighbors(
                dataset_umap, dataset_labels,
                prediction_umap, prediction_labels
            )
    
    def create_class_separation_analysis(self, embedding, labels, save_path=None):
        """
        Analyze how well classes are separated in the embedding space
        
        Args:
            embedding: 2D embedding coordinates
            labels: Class labels
            save_path: Path to save analysis plot
        """
        from sklearn.metrics import silhouette_score
        from scipy.spatial.distance import pdist, squareform
        
        print("\nClass Separation Analysis:")
        print("=" * 40)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(embedding, labels)
        print(f"Average Silhouette Score: {silhouette_avg:.3f}")
        
        # Calculate within-class and between-class distances
        within_class_distances = []
        between_class_distances = []
        
        for class_idx in range(10):
            class_mask = labels == class_idx
            class_points = embedding[class_mask]
            
            if len(class_points) > 1:
                # Within-class distances
                within_distances = pdist(class_points)
                within_class_distances.extend(within_distances)
                
                # Between-class distances
                other_points = embedding[~class_mask]
                for point in class_points:
                    distances_to_others = np.linalg.norm(other_points - point, axis=1)
                    between_class_distances.extend(distances_to_others)
        
        # Create analysis plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Distance distributions
        ax1.hist(within_class_distances, bins=50, alpha=0.7, label='Within-class', density=True)
        ax1.hist(between_class_distances, bins=50, alpha=0.7, label='Between-class', density=True)
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Density')
        ax1.set_title('Distance Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Class centroids
        centroids = []
        for class_idx in range(10):
            class_mask = labels == class_idx
            if np.any(class_mask):
                centroid = np.mean(embedding[class_mask], axis=0)
                centroids.append(centroid)
                ax2.scatter(centroid[0], centroid[1], 
                           c=self.class_colors[class_idx], 
                           s=200, marker='*', 
                           edgecolors='black', linewidth=2,
                           label=self.data_loader.class_names[class_idx])
        
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        ax2.set_title('Class Centroids')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Class separation analysis saved: {save_path}")
        
        plt.show()
        
        # Print statistics
        print(f"Average within-class distance: {np.mean(within_class_distances):.3f}")
        print(f"Average between-class distance: {np.mean(between_class_distances):.3f}")
        print(f"Separation ratio: {np.mean(between_class_distances) / np.mean(within_class_distances):.3f}")


def main():
    """Example usage of the dimensionality visualizer"""
    print("Fashion-MNIST Dimensionality Reduction Visualizer")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = DimensionalityVisualizer()
    
    # Create dataset-only visualization for exploration
    print("Creating dataset visualization...")
    images, labels = visualizer.prepare_dataset_sample(n_samples=1000)
    features = visualizer.extract_features(images, method='pca')
    
    # t-SNE visualization
    tsne_embedding = visualizer.compute_tsne(features, perplexity=30, max_iter=1000)
    visualizer.create_visualization(
        tsne_embedding, labels, 
        't-SNE Visualization: Fashion-MNIST Dataset',
        save_path='results/tsne_dataset_only.png'
    )
    
    # UMAP visualization
    umap_embedding = visualizer.compute_umap(features, n_neighbors=15)
    visualizer.create_visualization(
        umap_embedding, labels,
        'UMAP Visualization: Fashion-MNIST Dataset', 
        save_path='results/umap_dataset_only.png'
    )
    
    # Class separation analysis
    visualizer.create_class_separation_analysis(
        tsne_embedding, labels,
        save_path='results/class_separation_analysis.png'
    )
    
    print("\n✓ Dataset visualization completed!")
    print("To visualize predictions, use visualize_predictions() with trained model and image paths")


if __name__ == "__main__":
    main()