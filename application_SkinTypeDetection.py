import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from threading import Thread
import time

class SkinTypeApp:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.root = tk.Tk()
        self.root.title("Skin Type Predictor")
        self.root.geometry("1400x800")
        self.root.configure(bg="#f0f0f0")

        self.model = self.load_model(model_path)
        
        # Camera variables
        self.cap = None
        self.is_camera_running = False
        self.camera_thread = None
        self.camera_list = self.get_available_cameras()
        self.last_prediction_time = 0
        self.prediction_delay = 1  # Delay between predictions in seconds
        
        self.create_ui()

    def get_available_cameras(self):
        camera_list = []
        for i in range(2):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_list.append(i)
                    cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {str(e)}")
        return camera_list if camera_list else [0]

    def load_model(self, model_path):
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 3)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def create_ui(self):
        # Create main container
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

        # Left panel for file input and analysis
        left_panel = ttk.Frame(container)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10)

        # Input selection frame
        input_frame = ttk.LabelFrame(left_panel, text="Input Selection", padding=10)
        input_frame.pack(fill=tk.X, pady=10)

        # File selection
        file_frame = ttk.Frame(input_frame)
        file_frame.pack(fill=tk.X, pady=5)

        self.image_path = tk.StringVar()
        ttk.Label(file_frame, text="Image File:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(file_frame, textvariable=self.image_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_image).pack(side=tk.LEFT, padx=5)

        # Camera selection
        camera_frame = ttk.Frame(input_frame)
        camera_frame.pack(fill=tk.X, pady=5)

        ttk.Label(camera_frame, text="Camera:").pack(side=tk.LEFT, padx=5)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(
            camera_frame,
            textvariable=self.camera_var,
            values=[f"Camera {i}" for i in self.camera_list]
        )
        self.camera_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        if self.camera_list:
            self.camera_combo.set(f"Camera {self.camera_list[0]}")

        self.camera_button = ttk.Button(
            camera_frame,
            text="Start Camera",
            command=self.toggle_camera
        )
        self.camera_button.pack(side=tk.LEFT, padx=5)

        # Image display frames
        display_frame = ttk.Frame(left_panel)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Input image
        input_frame = ttk.LabelFrame(display_frame, text="Input Image")
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.image_label = ttk.Label(input_frame)
        self.image_label.pack(padx=10, pady=10)

        # Grad-CAM image
        gradcam_frame = ttk.LabelFrame(display_frame, text="Grad-CAM Heatmap")
        gradcam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.grad_cam_label = ttk.Label(gradcam_frame)
        self.grad_cam_label.pack(padx=10, pady=10)

        ttk.Button(left_panel, text="Analyze Image", command=self.analyze_image).pack(pady=10)
        
        self.result_label = ttk.Label(left_panel, text="", justify="center", font=("Helvetica", 12))
        self.result_label.pack(pady=10)

        # Right panel for camera feed
        right_panel = ttk.Frame(container)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=10)
        
        camera_frame = ttk.LabelFrame(right_panel, text="Live Camera Feed")
        camera_frame.pack(fill=tk.BOTH, expand=True)
        
        self.camera_canvas = tk.Canvas(camera_frame, width=640, height=480)
        self.camera_canvas.pack(padx=10, pady=10)
        
        self.live_result_label = ttk.Label(right_panel, text="", justify="center", font=("Helvetica", 12))
        self.live_result_label.pack(pady=10)

    def toggle_camera(self):
        if not self.is_camera_running:
            try:
                selected_camera = self.camera_combo.current()
                if selected_camera < 0:
                    selected_camera = 0

                self.cap = cv2.VideoCapture(selected_camera, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    raise ValueError("Failed to open camera")

                self.is_camera_running = True
                self.camera_button.config(text="Stop Camera")
                self.camera_thread = Thread(target=self.update_camera, daemon=True)
                self.camera_thread.start()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
                self.cleanup_camera()
        else:
            self.cleanup_camera()

    def cleanup_camera(self):
        self.is_camera_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.camera_button.config(text="Start Camera")
        self.camera_canvas.delete("all")
        self.live_result_label.config(text="")

    def update_camera(self):
        while self.is_camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Real-time prediction
                current_time = time.time()
                if current_time - self.last_prediction_time >= self.prediction_delay:
                    self.predict_from_frame(frame_rgb)
                    self.last_prediction_time = current_time
                
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil.thumbnail((640, 480))
                photo = ImageTk.PhotoImage(frame_pil)
                
                self.camera_canvas.config(width=photo.width(), height=photo.height())
                self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.camera_canvas.photo = photo
                
            time.sleep(0.03)
    def predict_from_frame(self, frame):
        try:
            img = Image.fromarray(frame)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_transformed = transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_transformed)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                predicted_class = output.argmax(dim=1).item()

            index_label = {0: "Dry", 1: "Normal", 2: "Oily"}
            prediction_label = index_label[predicted_class]
            confidence = probabilities[predicted_class].item() * 100

            self.live_result_label.config(
                text=f"Predicted Skin Type: {prediction_label}\n"
                     f"Confidence: {confidence:.2f}%"
            )
        except Exception as e:
            print(f"Prediction error: {str(e)}")

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.image_path.set(file_path)
            self.display_image(file_path)

    def display_image(self, file_path):
        try:
            img = Image.open(file_path).convert("RGB")
            img.thumbnail((400, 400))  # Resize for consistent UI layout
            photo = ImageTk.PhotoImage(img)

            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Image Error", str(e))

    def analyze_image(self):
        file_path = self.image_path.get()
        if not file_path:
            self.result_label.config(text="Please select an image first.")
            return

        try:
            img = Image.open(file_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_transformed = transform(img).unsqueeze(0).to(self.device)

            # Perform Grad-CAM analysis
            heatmap, predicted_class = self.grad_cam(img_transformed)
            index_label = {0: "dry", 1: "normal", 2: "oily"}
            prediction_label = index_label[predicted_class]

            # Generate heatmap overlay
            heatmap_resized = cv2.resize(heatmap, img.size)
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            img_np = np.array(img)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
            overlay_img = Image.fromarray(overlay)

            self.display_result_images(img, overlay_img)

            # Display results
            self.result_label.config(
                text=f"Prediction: {prediction_label}\n"
                     f"Explanation: The heatmap highlights the regions contributing to the prediction."
            )
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))

    def display_result_images(self, original_img, heatmap_img):
        original_img.thumbnail((400, 400))
        original_photo = ImageTk.PhotoImage(original_img)
        self.image_label.config(image=original_photo)
        self.image_label.image = original_photo

        heatmap_img.thumbnail((400, 400))
        heatmap_photo = ImageTk.PhotoImage(heatmap_img)
        self.grad_cam_label.config(image=heatmap_photo)
        self.grad_cam_label.image = heatmap_photo

    def grad_cam(self, input_image):
        target_layer = self.model.layer4[-1]
        feature_maps, gradients = None, None

        def forward_hook(module, input, output):
            nonlocal feature_maps
            feature_maps = output

        def backward_hook(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0]

        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_backward_hook(backward_hook)

        output = self.model(input_image)
        target_class = output.argmax(dim=1).item()
        loss = output[:, target_class]

        self.model.zero_grad()
        loss.backward()

        handle_forward.remove()
        handle_backward.remove()

        pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(feature_maps.shape[1]):
            feature_maps[:, i, :, :] *= pooled_grads[i]

        heatmap = feature_maps.squeeze().detach().cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap, target_class

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.cleanup_camera()
        self.root.destroy()

if __name__ == "__main__":
    app = SkinTypeApp(model_path="D:\\AI_Face\\model\\best_model.pth")
    app.run()
