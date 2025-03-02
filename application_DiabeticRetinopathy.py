import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sv_ttk
import pywinstyles
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model

class InceptionV3GradCAM:
    def __init__(self, model_path):
        """
        Initialize Grad-CAM for Inception V3 Diabetic Retinopathy model
        
        Args:
            model_path (str): Path to saved Keras model
        """
        # Load the model
        self.model = load_model(model_path)
        
        # Get the InceptionV3 submodel
        inception_v3_model = self.model.get_layer("inception_v3")

        # Identify the target layer for Grad-CAM
        target_layer = inception_v3_model.get_layer("mixed10")

        # Create a gradient model using the InceptionV3 submodel
        self.grad_model = Model(
            inputs=inception_v3_model.input,
            outputs=[inception_v3_model.output, target_layer.output]
        )
    
    def compute_heatmap(self, img, class_index=None):
        """
        Compute Grad-CAM heatmap for Inception V3
        
        Args:
            img (numpy.ndarray): Preprocessed input image
            class_index (int, optional): Target class index
        
        Returns:
            numpy.ndarray: Grad-CAM heatmap
        """
        # Ensure input is a tensor with batch dimension
        inputs = tf.cast(tf.expand_dims(img, 0), tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs, feature_maps = self.grad_model(inputs)
            
            # If no class specified, use the most probable class
            if class_index is None:
                class_index = tf.argmax(outputs[0])
            
            class_output = outputs[0][class_index]
        
        # Compute gradients
        grads = tape.gradient(class_output, feature_maps)
        
        # Global Average Pooling of gradients
        if len(grads.shape) > 3:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        else:
            pooled_grads = tf.reduce_mean(grads)
        
        # Cast feature maps and pooled grads to float32
        feature_maps = tf.cast(feature_maps, tf.float32)
        pooled_grads = tf.cast(pooled_grads, tf.float32)
        
        # Weight feature maps
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, feature_maps), 
            axis=-1
        )
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap[0], 0)
        heatmap /= tf.math.reduce_max(heatmap)
        
        # Convert to numpy and resize
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (img.shape[1] * 2, img.shape[0] * 2))
        heatmap = np.uint8(255 * heatmap)
        
        return heatmap
    
    def overlay_heatmap(self, original_img, heatmap, alpha=0.4):
        """
        Overlay heatmap on original image
        
        Args:
            original_img (numpy.ndarray): Original input image
            heatmap (numpy.ndarray): Computed heatmap
        
        Returns:
            numpy.ndarray: Image with overlaid heatmap
        """
        # Ensure the original image is in RGB format
        original_img_rgb = cv2.cvtColor(original_img.astype('uint8'), cv2.COLOR_RGB2BGR)
        
        # Convert grayscale heatmap to a color map (BGR format)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Resize heatmap to match original image size
        colored_heatmap = cv2.resize(colored_heatmap, (original_img.shape[1], original_img.shape[0]))
        
        # Overlay the heatmap on the original image
        overlay = cv2.addWeighted(original_img_rgb, 1 - alpha, colored_heatmap, alpha, 0)
        
        # Convert back to RGB
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

class DiabeticRetinopathyApp:
    def __init__(self, model_path):
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("Diabetic Retinopathy Detector")
        self.root.geometry("1920x1080")  # Increase the window size
        self.root.configure(bg="#f0f0f0")  # Use a lighter, calmer background color

        # Apply a modern Windows 11 style
        pywinstyles.apply_style(self.root, "acrylic")
        sv_ttk.set_theme("light")

        # Load the pre-trained model and create Grad-CAM handler
        self.model = tf.keras.models.load_model(model_path)
        self.grad_cam_handler = InceptionV3GradCAM(model_path)

        # Create UI components
        self.create_ui()

    def create_ui(self):
    # Create a main scrollable frame for all content
        self.main_scroll = ttk.Scrollbar(self.root, orient="vertical")
        self.main_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.main_canvas = tk.Canvas(self.root, yscrollcommand=self.main_scroll.set)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.main_scroll.config(command=self.main_canvas.yview)
        
        # Create a frame inside the canvas for all content
        self.content_frame = ttk.Frame(self.main_canvas)
        self.main_canvas.create_window((0, 0), window=self.content_frame, anchor='n')
        
        # Configure grid weight to enable centering
        self.content_frame.grid_columnconfigure(0, weight=1)
        
        # Main frame
        main_frame = ttk.Frame(self.content_frame, padding="30")
        main_frame.grid(row=0, column=0, sticky='nsew')
        main_frame.grid_columnconfigure(0, weight=1)

        # Image selection section
        image_frame = ttk.LabelFrame(main_frame, text="Select Retina Image")
        image_frame.grid(row=0, column=0, pady=20, sticky='ew')
        image_frame.grid_columnconfigure(0, weight=1)

        self.image_path = tk.StringVar()
        path_entry = ttk.Entry(image_frame, textvariable=self.image_path)
        path_entry.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        browse_btn = ttk.Button(image_frame, text="Browse", command=self.browse_image)
        browse_btn.grid(row=0, column=1, padx=10, pady=10)

        # Create a frame for image displays
        self.display_frame = ttk.Frame(main_frame)
        self.display_frame.grid(row=1, column=0, pady=20, sticky='nsew')
        self.display_frame.grid_columnconfigure(0, weight=1)
        self.display_frame.grid_columnconfigure(1, weight=1)
        
        # Create a centered frame for images
        image_display_frame = ttk.Frame(self.display_frame)
        image_display_frame.grid(row=0, column=0, columnspan=2, sticky='nsew')
        image_display_frame.grid_columnconfigure(0, weight=1, uniform='group1')
        image_display_frame.grid_columnconfigure(1, weight=1, uniform='group1')

        # Original image display
        self.image_label = ttk.Label(self.display_frame, text="Selected Image will appear here")
        self.image_label.grid(row=0, column=0, padx=20, pady=20)

        # GRAD-CAM display
        self.grad_cam_label = ttk.Label(self.display_frame, text="GRAD-CAM visualization will appear here")
        self.grad_cam_label.grid(row=0, column=1, padx=20, pady=20)

        # Analyze button
        analyze_btn = ttk.Button(main_frame, text="Analyze Image", command=self.analyze_image)
        analyze_btn.grid(row=2, column=0, pady=20)

        # Results section
        self.result_frame = ttk.LabelFrame(main_frame, text="Analysis Results")
        self.result_frame.grid(row=3, column=0, pady=20, sticky='ew')
        self.result_frame.grid_columnconfigure(0, weight=1)

        self.result_label = ttk.Label(self.result_frame, text="", justify='center')
        self.result_label.grid(row=0, column=0, padx=20, pady=20)
        
        # Create a frame to hold both results and metrics side by side
        results_container = ttk.Frame(main_frame)
        results_container.grid(row=3, column=0, pady=20, sticky='ew')
        results_container.grid_columnconfigure(0, weight=1)
        results_container.grid_columnconfigure(1, weight=1)

        # Results section
        self.result_frame = ttk.LabelFrame(results_container, text="Analysis Results")
        self.result_frame.grid(row=0, column=0, pady=20, padx=10, sticky='nsew')
        self.result_frame.grid_columnconfigure(0, weight=1)

        self.result_label = ttk.Label(self.result_frame, text="", justify='center')
        self.result_label.grid(row=0, column=0, padx=20, pady=20)

        # Metrics section
        self.metrics_frame = ttk.LabelFrame(results_container, text="Performance Metrics")
        self.metrics_frame.grid(row=0, column=1, pady=20, padx=10, sticky='nsew')
        self.metrics_frame.grid_columnconfigure(0, weight=1)

        self.metrics_label = ttk.Label(self.metrics_frame, text="", justify='left')
        self.metrics_label.grid(row=0, column=0, padx=20, pady=20)

        # Bind configure event to update scroll region
        self.content_frame.bind('<Configure>', self.on_configure)
        
    def on_configure(self, event):
        """
        Update the scroll region when the content size changes
        """
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox('all'))
        
    def browse_image(self):
        """Open file dialog to select an image"""
        filename = filedialog.askopenfilename(
            title="Select Retina Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if filename:
            self.image_path.set(filename)
            self.display_image(filename)
    
    def display_image(self, image_path):
        """Display selected image in the UI"""
        try:
            # Open and resize image
            img = Image.open(image_path)
            img.thumbnail((400, 400))  # Resize to fit UI
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Image Error", str(e))
    
    def preprocess_image(self, image_path, target_size=(299, 299)):
        """
        Preprocess image for Inception V3 model input
        
        Args:
            image_path (str): Path to input image
            target_size (tuple): Target image size
        
        Returns:
            numpy.ndarray: Preprocessed image
        """
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        return img_array
    
    def display_grad_cam(self, original_img, heatmap):
        """
        Display the GRAD-CAM visualization next to the original image.
        
        Args:
            original_img (numpy.ndarray): Original input image
            heatmap (numpy.ndarray): Computed GRAD-CAM heatmap
        """
        for widget in self.grad_cam_container.winfo_children():
            widget.destroy()

        # Create a new frame to hold the original image and heatmap
        grad_cam_frame = ttk.Frame(self.grad_cam_container)
        grad_cam_frame.pack(fill=tk.BOTH, expand=True)

        # Display the original image
        original_img = Image.fromarray(original_img.astype('uint8'))
        original_img.thumbnail((400, 400))
        original_photo = ImageTk.PhotoImage(original_img)
        original_label = ttk.Label(grad_cam_frame, image=original_photo, style="GradCam.TLabel")
        original_label.image = original_photo
        original_label.pack(side=tk.LEFT, padx=20, pady=20)

        # Display the GRAD-CAM heatmap
        heatmap_img = Image.fromarray(heatmap)
        heatmap_img.thumbnail((400, 400))
        heatmap_photo = ImageTk.PhotoImage(heatmap_img)
        heatmap_label = ttk.Label(grad_cam_frame, image=heatmap_photo, style="GradCam.TLabel")
        heatmap_label.image = heatmap_photo
        heatmap_label.pack(side=tk.RIGHT, padx=20, pady=20)

        # Add explanatory text
        explanation_label = ttk.Label(self.grad_cam_container, text="The GRAD-CAM heatmap highlights the regions of the image that the model is focusing on to make the prediction.", style="GradCam.TLabel")
        explanation_label.pack(side=tk.BOTTOM, pady=20)

        # Configure the canvas scrollregion
        self.grad_cam_canvas.configure(scrollregion=self.grad_cam_canvas.bbox("all"))
    
    def analyze_image(self):
        """Analyze the selected image"""
        img_path = self.image_path.get()
        if not img_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            # Preprocess and predict
            processed_img = self.preprocess_image(img_path)
            prediction = self.model.predict(np.expand_dims(processed_img, 0))
            
            # Calculate metrics for this prediction
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Get prediction probabilities for each class
            class_probabilities = prediction[0]
            
            # Calculate metrics based on model's prediction
            metrics = {
                'confidence': confidence,
                'entropy': -np.sum(class_probabilities * np.log2(class_probabilities + 1e-10)),
                'top_2_diff': np.sort(class_probabilities)[-1] - np.sort(class_probabilities)[-2],
                'probability_distribution': class_probabilities,
                'prediction_certainty': 1 - (-np.sum(class_probabilities * np.log2(class_probabilities + 1e-10)) / np.log2(len(class_probabilities)))
            }
            
            # Format metrics text with actual values from the prediction
            metrics_text = (
                f"Model Prediction Metrics:\n\n"
                f"Confidence Score: {metrics['confidence']:.4f}\n"
                f"Prediction Certainty: {metrics['prediction_certainty']:.4f}\n"
                f"Decision Entropy: {metrics['entropy']:.4f}\n"
                f"Margin (Top 2 Difference): {metrics['top_2_diff']:.4f}\n\n"
                f"Class Probabilities:\n"
            )
            
            # Add individual class probabilities
            severity_levels = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
            for i, prob in enumerate(metrics['probability_distribution']):
                metrics_text += f"{severity_levels[i]}: {prob:.4f}\n"
            
            # Update metrics display
            self.metrics_label.config(text=metrics_text)
            
            # Determine result
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Generate Grad-CAM visualization
            heatmap = self.grad_cam_handler.compute_heatmap(processed_img, class_index=predicted_class)
            
            # Load original image for heatmap overlay
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Resize original image while maintaining aspect ratio
            desired_width = 400
            aspect_ratio = original_img.shape[1] / original_img.shape[0]
            desired_height = int(desired_width / aspect_ratio)
            
            original_img_resized = cv2.resize(original_img, (desired_width, desired_height))
            
            # Generate heatmap overlay at the same size
            heatmap_overlay = self.grad_cam_handler.overlay_heatmap(original_img_resized, heatmap)
            
            # Convert to PIL Images for display
            original_pil = Image.fromarray(original_img_resized)
            heatmap_pil = Image.fromarray(heatmap_overlay)
            
            # Convert to PhotoImage
            original_photo = ImageTk.PhotoImage(original_pil)
            heatmap_photo = ImageTk.PhotoImage(heatmap_pil)
            
            # Update labels with new images
            self.image_label.configure(image=original_photo)
            self.image_label.image = original_photo
            
            self.grad_cam_label.configure(image=heatmap_photo)
            self.grad_cam_label.image = heatmap_photo
            
            # Center align the images in their frames
            self.image_label.grid(row=0, column=0, padx=20, pady=20)
            self.grad_cam_label.grid(row=0, column=1, padx=20, pady=20)
            
            # Display results
            severity_levels = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
            result_text = (
                f"Diabetic Retinopathy Detection:\n"
                f"Severity: {severity_levels[predicted_class]}\n"
                f"Confidence: {confidence:.2%}\n\n"
                + self.get_dr_explanation(predicted_class)
            )
            
            self.result_label.config(text=result_text)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
            
    def get_dr_explanation(self, has_dr):
        """Provide a binary explanation of Diabetic Retinopathy based on classification"""
        explanations = {
            0: {
                "description": "No Diabetic Retinopathy Detected\n",
                "details": (
                    "- Your retina appears healthy\n"
                    "- Continue regular eye check-ups\n"
                    "- Maintain good diabetes management"
                )
            },
            1: {
                "description": "Diabetic Retinopathy Detected\n",
                "details": (
                    "- Signs of retinal damage detected\n"
                    "- Diabetic Retinopathy is present and requires medical attention\n"
                    "- Recommended actions:\n"
                    "  1. Consult an ophthalmologist for further evaluation\n"
                    "  2. Monitor blood sugar, blood pressure, and cholesterol levels\n"
                    "  3. Follow comprehensive eye care plans to prevent progression\n"
                    "  4. Maintain regular eye screenings as advised"
                )
            }
        }
        
        explanation = explanations.get(has_dr, explanations[0])
        return explanation["description"] + explanation["details"]


    def run(self):
        """Start the application"""
        pywinstyles.apply_style(self.root, "acrylic")  # Apply a modern Windows 11 style
        sv_ttk.set_theme("dark")  # or "dark"
        self.root.mainloop()

def main():
    # Update this with your actual model path
    MODEL_PATH = r"D:\\AOL_AI\\Models\\fine_tuned_phase\\diabetic_retinopathy_model_99.h5"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    app = DiabeticRetinopathyApp(MODEL_PATH)
    app.run()

if __name__ == '__main__':
    main()