# Brain Tumor Detection using Deep Learning - MacBook Optimized
# Based on the presentation by Tejaswini Tatikonda

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import cv2
from glob import glob
import pandas as pd
import seaborn as sns
import pathlib
from datetime import datetime
import multiprocessing

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Optimizations for MacBook
# Use CPU cores effectively (adjust based on your MacBook's capabilities)
NUM_CORES = multiprocessing.cpu_count()
print(f"Number of CPU cores available: {NUM_CORES}")

# Configure TensorFlow for Mac - Metal GPU support if available
# Check if Metal plugin is available (Apple Silicon M1/M2/etc)
if tf.config.list_physical_devices('GPU'):
    print("Metal GPU acceleration is available")
    # Configure memory growth to avoid memory allocation errors
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("Running on CPU only")


class BrainTumorDetector:
    def __init__(self, data_dir=None, img_size=224, model_path=None):
        """
        Initialize the Brain Tumor Detector
        
        Args:
            data_dir: Directory containing the dataset
            img_size: Size to which all images will be resized
            model_path: Path to a pre-trained model (if available)
        """
        if data_dir is None:
            # Use home directory as base for macOS
            home_dir = str(pathlib.Path.home())
            self.data_dir = os.path.join(home_dir, 'brain_mri_data')
        else:
            self.data_dir = data_dir
            
        self.img_size = img_size
        self.model = None
        self.history = None
        self.class_indices = None
        
        # Create results directory
        self.results_dir = os.path.join(os.getcwd(), 'brain_tumor_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load existing model if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            self.model = load_model(model_path)
        
    def load_and_prepare_data(self, augment=True):
        """
        Load and prepare data for training and validation
        
        Args:
            augment: Whether to apply data augmentation
        
        Returns:
            train_generator, validation_generator
        """
        print("Loading and preparing data...")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'train/normal'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'train/tumor'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'val/normal'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'val/tumor'), exist_ok=True)
        
        # Data augmentation
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
            
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='binary',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'val'),
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )
        
        self.class_indices = train_generator.class_indices
        print(f"Class indices: {self.class_indices}")
        
        print(f"Found {train_generator.samples} training images")
        print(f"Found {val_generator.samples} validation images")
        
        return train_generator, val_generator
    
    def check_for_dataset(self):
        """Check if dataset exists or guide user to create one"""
        train_path = os.path.join(self.data_dir, 'train')
        val_path = os.path.join(self.data_dir, 'val')
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print(f"\nDataset not found at {self.data_dir}")
            print("\nTo create a dataset:")
            print(f"1. Create the following directory structure:")
            print(f"   {self.data_dir}/")
            print(f"   ├── train/")
            print(f"   │   ├── normal/   # Place normal brain MRI images here")
            print(f"   │   └── tumor/    # Place tumor brain MRI images here")
            print(f"   └── val/")
            print(f"       ├── normal/   # Place normal brain MRI images for validation")
            print(f"       └── tumor/    # Place tumor brain MRI images for validation")
            print("\n2. You can download brain tumor MRI datasets from:")
            print("   - Kaggle: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
            print("   - Figshare: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427")
            print("\nRunning in demonstration mode instead...")
            return False
            
        # Check if directories have images
        normal_train_images = glob(os.path.join(train_path, 'normal', '*.*'))
        tumor_train_images = glob(os.path.join(train_path, 'tumor', '*.*'))
        normal_val_images = glob(os.path.join(val_path, 'normal', '*.*'))
        tumor_val_images = glob(os.path.join(val_path, 'tumor', '*.*'))
        
        if len(normal_train_images) == 0 or len(tumor_train_images) == 0 or \
           len(normal_val_images) == 0 or len(tumor_val_images) == 0:
            print("One or more dataset directories are empty!")
            print("Running in demonstration mode instead...")
            return False
            
        return True
    
    def build_model(self):
        """
        Build the CNN model architecture optimized for MacBook
        
        Returns:
            The compiled model
        """
        print("Building CNN model...")
        
        # For Apple Silicon Macs, smaller model can work better
        is_apple_silicon = False
        try:
            import platform
            is_apple_silicon = platform.processor() == 'arm'
        except:
            pass
        
        # Adjust model complexity based on hardware
        filters_multiplier = 0.75 if is_apple_silicon else 1.0
        
        model = Sequential([
            # First convolutional block
            Conv2D(int(32 * filters_multiplier), (3, 3), activation='relu', padding='same', 
                   input_shape=(self.img_size, self.img_size, 3)),
            BatchNormalization(),
            Conv2D(int(32 * filters_multiplier), (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(int(64 * filters_multiplier), (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(int(64 * filters_multiplier), (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(int(128 * filters_multiplier), (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(int(128 * filters_multiplier), (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Flatten and fully connected layers
            Flatten(),
            Dense(int(512 * filters_multiplier), activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(int(128 * filters_multiplier), activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification (tumor vs normal)
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        self.model = model
        return model
    
    def train_model(self, train_generator, val_generator, epochs=20):
        """
        Train the model with the given data generators
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs to train for
            
        Returns:
            Training history
        """
        print("Training the model...")
        
        # Create timestamp for model saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.results_dir, f'brain_tumor_model_{timestamp}.h5')
        
        # Define callbacks
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[checkpoint, early_stopping],
            workers=NUM_CORES,  # Use multiple CPU cores for data loading
            use_multiprocessing=True
        )
        
        self.history = history
        print(f"Model saved to {model_path}")
        return history
    
    def evaluate_model(self, val_generator):
        """
        Evaluate the model performance
        
        Args:
            val_generator: Validation data generator
            
        Returns:
            Validation accuracy and ROC AUC score
        """
        print("Evaluating the model...")
        
        # Get model accuracy on validation set
        val_loss, val_accuracy = self.model.evaluate(val_generator)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        
        # Predict on validation data
        val_generator.reset()
        y_pred = self.model.predict(val_generator)
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        y_true = val_generator.classes
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=['Normal', 'Tumor']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Tumor'], 
                    yticklabels=['Normal', 'Tumor'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.results_dir, 'roc_curve.png'))
        plt.close()
        
        return val_accuracy, roc_auc
    
    def plot_training_history(self):
        """
        Plot training and validation accuracy/loss
        """
        if self.history is None:
            print("No training history available.")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_history.png'))
        plt.close()
    
    def segment_tumor(self, image_path):
        """
        Segment tumor from brain MRI using multi-level thresholding
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Original image, segmentation mask, segmented image, and tumor density
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        mask = np.zeros_like(gray)
        
        # Draw contours of minimum size (to filter noise)
        min_contour_area = 100
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Apply mask to original image
        segmented = cv2.bitwise_and(img, img, mask=mask)
        
        # Calculate tumor density (percentage of white pixels in the mask)
        tumor_density = np.sum(mask > 0) / (self.img_size * self.img_size) * 100
        
        return img, mask, segmented, tumor_density
    
    def predict_and_segment(self, image_path):
        """
        Predict whether image contains tumor and segment if positive
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        if self.model is None:
            raise ValueError("Model not built or loaded. Call build_model() first.")
            
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_array = np.expand_dims(img / 255.0, axis=0)
        
        # Predict
        prediction = self.model.predict(img_array)[0][0]
        has_tumor = prediction > 0.5
        
        result = {
            'has_tumor': has_tumor,
            'confidence': float(prediction) if has_tumor else float(1 - prediction),
            'tumor_density': None,
            'image': img,
            'mask': None,
            'segmented': None
        }
        
        # If tumor detected, perform segmentation
        if has_tumor:
            _, mask, segmented, tumor_density = self.segment_tumor(image_path)
            result['tumor_density'] = tumor_density
            result['mask'] = mask
            result['segmented'] = segmented
        
        return result
    
    def display_results(self, result):
        """
        Display detection and segmentation results
        
        Args:
            result: Dictionary with prediction results
        """
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB))
        plt.title('Original MRI')
        plt.axis('off')
        
        # Results text
        status = "TUMOR DETECTED" if result['has_tumor'] else "NO TUMOR"
        confidence = result['confidence'] * 100
        
        if result['has_tumor']:
            # Segmentation mask
            plt.subplot(1, 3, 2)
            plt.imshow(result['mask'], cmap='gray')
            plt.title('Tumor Segmentation Mask')
            plt.axis('off')
            
            # Segmented image
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(result['segmented'], cv2.COLOR_BGR2RGB))
            plt.title(f'Segmented Tumor\nDensity: {result["tumor_density"]:.2f}%')
            plt.axis('off')
        else:
            plt.subplot(1, 3, 2)
            plt.text(0.5, 0.5, f"{status}\nConfidence: {confidence:.2f}%", 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14)
            plt.axis('off')
        
        plt.suptitle(f"Brain Tumor Detection Result: {status} (Confidence: {confidence:.2f}%)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'detection_result.png'))
        plt.show()
    
    def create_demo_image(self, has_tumor=True):
        """
        Create a demo image for testing when no real data is available
        
        Args:
            has_tumor: Whether to create an image with a tumor
            
        Returns:
            Path to the created image
        """
        # Create a demo directory
        demo_dir = os.path.join(self.results_dir, 'demo')
        os.makedirs(demo_dir, exist_ok=True)
        
        # Create a blank image (simulating a brain MRI)
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Draw brain-like shape
        center = (self.img_size // 2, self.img_size // 2)
        axes = (self.img_size // 3, self.img_size // 2)
        cv2.ellipse(img, center, axes, 0, 0, 360, (150, 150, 150), -1)  # Brain outline
        
        # Add random textures to make it look more like a brain
        for _ in range(30):
            x = np.random.randint(center[0] - axes[0] + 10, center[0] + axes[0] - 10)
            y = np.random.randint(center[1] - axes[1] + 10, center[1] + axes[1] - 10)
            r = np.random.randint(5, 15)
            c = np.random.randint(130, 170)
            cv2.circle(img, (x, y), r, (c, c, c), -1)
        
        # Add a tumor if specified
        if has_tumor:
            # Random position within the brain
            tumor_x = np.random.randint(center[0] - axes[0] // 2, center[0] + axes[0] // 2)
            tumor_y = np.random.randint(center[1] - axes[1] // 2, center[1] + axes[1] // 2)
            tumor_size = np.random.randint(15, 30)
            
            # Draw tumor with a distinctive color
            cv2.circle(img, (tumor_x, tumor_y), tumor_size, (50, 50, 200), -1)  # Red tumor
        
        # Save the image
        image_path = os.path.join(demo_dir, f"demo_{'tumor' if has_tumor else 'normal'}.jpg")
        cv2.imwrite(image_path, img)
        
        return image_path


def main():
    """Main function to run the brain tumor detection system"""
    print("=" * 60)
    print("Brain Tumor Detection System for MacBook")
    print("Based on the presentation by Tejaswini Tatikonda")
    print("=" * 60)
    
    # Create detector instance - use ~/Documents/brain_mri_data as default on Mac
    home_dir = str(pathlib.Path.home())
    data_dir = os.path.join(home_dir, 'Documents', 'brain_mri_data')
    detector = BrainTumorDetector(data_dir=data_dir)
    
    # Check if dataset exists
    has_dataset = detector.check_for_dataset()
    
    if has_dataset:
        # Normal workflow with real data
        print("\n1. Loading and preparing data...")
        train_generator, val_generator = detector.load_and_prepare_data()
        
        print("\n2. Building model...")
        detector.build_model()
        
        print("\n3. Training model...")
        detector.train_model(train_generator, val_generator, epochs=15)
        
        print("\n4. Evaluating model...")
        accuracy, roc_auc = detector.evaluate_model(val_generator)
        
        print("\n5. Plotting training history...")
        detector.plot_training_history()
        
        # Get a test image from validation set
        test_files = []
        for class_dir in os.listdir(os.path.join(data_dir, 'val')):
            class_path = os.path.join(data_dir, 'val', class_dir)
            if os.path.isdir(class_path):
                test_files.extend(glob(os.path.join(class_path, '*.*')))
        
        if test_files:
            test_image_path = np.random.choice(test_files)
            print(f"\n6. Testing on a sample image: {test_image_path}")
            result = detector.predict_and_segment(test_image_path)
            detector.display_results(result)
        else:
            print("\n6. No test images found in validation directory.")
    
    else:
        # Demo mode with simulated data
        print("\nRunning in DEMO mode with simulated data...")
        
        print("\n1. Building a model for demonstration...")
        detector.build_model()
        
        print("\n2. Creating demo images...")
        normal_image_path = detector.create_demo_image(has_tumor=False)
        tumor_image_path = detector.create_demo_image(has_tumor=True)
        
        print("\n3. Since we don't have real data to train the model:")
        print("   - We'll create a simple demonstration to show what the system would do")
        print("   - In a real scenario, the model would be trained on actual MRI images")
        
        print("\n4. Simulating prediction and segmentation results...")
        
        # Create simulated results
        normal_result = {
            'has_tumor': False,
            'confidence': 0.93,
            'tumor_density': None,
            'image': cv2.imread(normal_image_path),
            'mask': None,
            'segmented': None
        }
        
        tumor_img = cv2.imread(tumor_image_path)
        # Create a mask for tumor demo
        gray = cv2.cvtColor(tumor_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        segmented = cv2.bitwise_and(tumor_img, tumor_img, mask=mask)
        
        tumor_result = {
            'has_tumor': True,
            'confidence': 0.89,
            'tumor_density': 15.3,
            'image': tumor_img,
            'mask': mask,
            'segmented': segmented
        }
        
        print("\n5. Displaying results for normal MRI (simulated):")
        detector.display_results(normal_result)
        
        print("\n6. Displaying results for MRI with tumor (simulated):")
        detector.display_results(tumor_result)
        
        print("\nDEMO COMPLETE!")
        print("\nTo use this system with real data:")
        print(f"1. Prepare a dataset at: {data_dir}")
        print("2. Structure your data as follows:")
        print(f"   {data_dir}/")
        print(f"   ├── train/")
        print(f"   │   ├── normal/   # Place normal brain MRI images here")
        print(f"   │   └── tumor/    # Place tumor brain MRI images here")
        print(f"   └── val/")
        print(f"       ├── normal/   # Place normal brain MRI images for validation")
        print(f"       └── tumor/    # Place tumor brain MRI images for validation")
    
    print("\nResults have been saved to:", detector.results_dir)
    print("\nBrain Tumor Detection System execution complete!")


if __name__ == "__main__":
    main()