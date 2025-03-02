import cv2
import numpy as np
from deepface import DeepFace
import time
from threading import Thread
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk
from datetime import datetime

class EnhancedSecurityCheckSystem:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.reference_image = None
        self.reference_photo = None
        self.current_fps = 0
        self.optimal_resolution = (640, 480)
        self.camera_list = self.get_available_cameras()
        self.current_camera_index = 0
        self.verification_history = []
        self.setup_gui()

    def get_available_cameras(self):
        camera_list = []
        max_cameras_to_check = 2  # Reduce the number of cameras to check to avoid unnecessary errors
        
        for i in range(max_cameras_to_check):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Specify DirectShow backend for Windows
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_list.append(i)
                    cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {str(e)}")
                continue
        
        return camera_list if camera_list else [0]

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Enhanced Security Check System")
        self.root.geometry("1200x800")

        # Main container
        self.container = ttk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for video and controls
        self.left_panel = ttk.Frame(self.container)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for video and reference image
        self.images_frame = ttk.Frame(self.left_panel)
        self.images_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create separate frames for video and reference
        self.video_container = ttk.LabelFrame(self.images_frame, text="Camera Feed")
        self.video_container.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        self.video_frame = ttk.Label(self.video_container)
        self.video_frame.pack(padx=5, pady=5)

        # Reference image frame
        self.reference_container = ttk.LabelFrame(self.images_frame, text="Reference Photo")
        self.reference_container.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        self.reference_label = ttk.Label(self.reference_container)
        self.reference_label.pack(padx=5, pady=5)

        # Camera selection
        self.camera_frame = ttk.Frame(self.left_panel)
        self.camera_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.camera_frame, text="Select Camera:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(
            self.camera_frame,
            textvariable=self.camera_var,
            values=[f"Camera {i}" for i in self.camera_list]
        )
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        if self.camera_list:
            self.camera_combo.set(f"Camera {self.camera_list[0]}")

        # Controls frame
        self.controls_frame = ttk.Frame(self.left_panel)
        self.controls_frame.pack(fill=tk.X, pady=5)

        self.upload_btn = ttk.Button(
            self.controls_frame,
            text="Upload Reference ID/Passport",
            command=self.upload_reference
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5)

        self.start_btn = ttk.Button(
            self.controls_frame,
            text="Start Recognition",
            command=self.toggle_recognition
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel for status and verification history
        self.right_panel = ttk.Frame(self.container)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        # Status section
        self.status_frame = ttk.LabelFrame(self.right_panel, text="Status")
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(
            self.status_frame,
            text="Status: Waiting for reference image"
        )
        self.status_label.pack(pady=5)
        
        self.match_label = ttk.Label(self.status_frame, text="Match: N/A")
        self.match_label.pack(pady=5)
        
        self.fps_label = ttk.Label(self.status_frame, text="FPS: N/A")
        self.fps_label.pack(pady=5)
        
        # Verification history
        self.history_frame = ttk.LabelFrame(
            self.right_panel,
            text="Verification History"
        )
        self.history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.history_tree = ttk.Treeview(
            self.history_frame,
            columns=("Timestamp", "Result", "Confidence"),
            show="headings"
        )
        self.history_tree.heading("Timestamp", text="Timestamp")
        self.history_tree.heading("Result", text="Result")
        self.history_tree.heading("Confidence", text="Confidence")
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        
    def upload_reference(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            try:
                # Load image for processing
                self.reference_image = cv2.imread(file_path)
                if self.reference_image is None:
                    raise ValueError("Failed to load image")

                # Display reference image while maintaining aspect ratio
                reference_display = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
                display_height = 240
                aspect_ratio = self.reference_image.shape[1] / self.reference_image.shape[0]
                display_width = int(display_height * aspect_ratio)
                
                reference_display = cv2.resize(reference_display, (display_width, display_height))
                reference_photo = Image.fromarray(reference_display)
                self.reference_photo = ImageTk.PhotoImage(reference_photo)
                self.reference_label.configure(image=self.reference_photo)

                self.status_label.config(text="Status: Reference image loaded")
                messagebox.showinfo("Success", "Reference image loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load reference image: {str(e)}")
                
    def optimize_camera_settings(self):
        try:
            # Set lower resolution first
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Try to optimize FPS
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Disable auto focus to improve performance (if supported)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            # Read a test frame to ensure settings are applied
            ret, _ = self.cap.read()
            if not ret:
                raise ValueError("Failed to read frame after optimization")
                
            self.optimal_resolution = (640, 480)
            self.current_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
        except Exception as e:
            print(f"Camera optimization error: {str(e)}")
            # Fall back to default settings if optimization fails
            self.optimal_resolution = (640, 480)
            self.current_fps = 30
        
    def verify_identity(self, frame):
        try:
            result = DeepFace.verify(
                frame,
                self.reference_image,
                model_name="VGG-Face",
                enforce_detection=False,
                distance_metric="cosine"
            )
            
            verified = result["verified"]
            confidence = 1 - result["distance"]
            
            # Log verification attempt
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.verification_history.append({
                "timestamp": timestamp,
                "result": "Match" if verified else "No Match",
                "confidence": f"{confidence:.2%}"
            })
            
            # Update history display
            self.history_tree.insert(
                "",
                0,
                values=(
                    timestamp,
                    "Match" if verified else "No Match",
                    f"{confidence:.2%}"
                )
            )
            
            # Keep only last 10 entries
            if len(self.history_tree.get_children()) > 10:
                self.history_tree.delete(self.history_tree.get_children()[-1])
            
            return verified, confidence
            
        except Exception as e:
            print(f"Verification error: {str(e)}")
            return False, 0.0
            
    def process_frame(self):
        frame_count = 0
        fps_start_time = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # FPS calculation
                frame_count += 1
                if frame_count >= 30:
                    current_time = time.time()
                    fps = frame_count / (current_time - fps_start_time)
                    self.fps_label.config(text=f"FPS: {fps:.2f}")
                    frame_count = 0
                    fps_start_time = current_time
                
                # Face detection
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    if self.reference_image is not None:
                        try:
                            is_match, confidence = self.verify_identity(frame)
                            match_text = f"Match: {'Yes' if is_match else 'No'}"
                            conf_text = f"Confidence: {confidence:.2%}"
                            
                            self.match_label.config(
                                text=f"{match_text}\n{conf_text}",
                                foreground="green" if is_match else "red"
                            )
                        except Exception as e:
                            print(f"Verification error: {str(e)}")
                
                # Convert frame for display while maintaining aspect ratio
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_height = 480
                aspect_ratio = frame.shape[1] / frame.shape[0]
                display_width = int(display_height * aspect_ratio)
                
                frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
                frame_pil = Image.fromarray(frame_resized)
                frame_tk = ImageTk.PhotoImage(image=frame_pil)
                
                self.video_frame.imgtk = frame_tk
                self.video_frame.configure(image=frame_tk)
                    
    def toggle_recognition(self):
        if not self.is_running:
            if self.reference_image is None:
                messagebox.showwarning("Warning", "Please upload a reference image first")
                return

            try:
                # Get selected camera index
                selected_camera = self.camera_combo.current()
                if selected_camera < 0:
                    selected_camera = 0
                
                # Initialize camera with DirectShow backend
                self.cap = cv2.VideoCapture(selected_camera, cv2.CAP_DSHOW)
                
                # Set buffer size to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if not self.cap.isOpened():
                    # Try default camera if selected camera fails
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        raise ValueError("Failed to open camera")

                self.optimize_camera_settings()
                self.is_running = True
                self.start_btn.config(text="Stop Recognition")
                self.status_label.config(text="Status: Running")

                Thread(target=self.process_frame, daemon=True).start()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
                self.cleanup()
        else:
            self.cleanup()
            self.start_btn.config(text="Start Recognition")
            self.status_label.config(text="Status: Stopped")
            
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def cleanup(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            
    def on_closing(self):
        self.cleanup()
        self.root.destroy()
        
if __name__ == "__main__":
    app = EnhancedSecurityCheckSystem()
    try:
        app.run()
    except Exception as e:
        print(f"Application error: {str(e)}")
        app.cleanup()