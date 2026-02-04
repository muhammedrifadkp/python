import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import mediapipe as mp
import math
import numpy as np
import pyautogui
import time

# ---------------- CONFIG ----------------
CAMERA_INDEX = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX
FRAME_R = 100  # Frame Reduction for active area
SMOOTHING = 5  # Smoothing factor
CLICK_COOLDOWN = 0.8 # Seconds between clicks
EYE_AR_THRESH = 0.20 # Increased threshold for easier detection
# ---------------------------------------

class HandControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Mouse: Hand & Eye Control")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)

        self.video_label = Label(root)
        self.video_label.pack(expand=True, fill="both")

        self.status_label = Label(
            root,
            text="Initializing AI Models...",
            font=("Arial", 14),
            fg="blue"
        )
        self.status_label.pack(pady=10)

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible")

        self.cap.set(3, 1280) # Increased resolution for better mesh accuracy
        self.cap.set(4, 720)

        # Mediapipe Hands Init
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Mediapipe Face Mesh Init (For Eyes)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.mp_drawing = mp.solutions.drawing_utils

        # Screen Dimensions
        self.w_scr, self.h_scr = pyautogui.size()
        
        # Smoothing & Relative Movement variables
        self.last_hand_pos = None  # To store the last hand position for relative movement
        self.sensitivity = 1.6     # Adjust mouse speed

        # Click State
        self.last_left_click = 0
        self.last_right_click = 0

        self.running = True
        self.update_frame()

    def get_blink_ratio(self, landmarks, top_idx, bottom_idx, left_idx, right_idx):
        try:
            # Coordinates (Screen-space normalized)
            pts = [np.array([landmarks[idx].x, landmarks[idx].y]) for idx in [top_idx, bottom_idx, left_idx, right_idx]]
            
            # Vertical distance
            v_dist = np.linalg.norm(pts[0] - pts[1])
            # Horizontal distance
            h_dist = np.linalg.norm(pts[2] - pts[3])
            
            if h_dist == 0: return 1.0
            return v_dist / h_dist
        except:
            return 1.0

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Failed to read camera")
            return

        h_cam, w_cam, _ = frame.shape
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Absolute Mapping: Entire camera frame to Entire screen (No box)
        
        # ---------------------------------------------------------
        # 1. FACE MESH (Eye Clicks)
        # ---------------------------------------------------------
        face_results = self.face_mesh.process(frame_rgb)
        click_status = ""
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                lms = face_landmarks.landmark
                
                # Standard indexes for height/width:
                # Left Eye (anatomical): 159 (top), 145 (bottom), 33 (left), 133 (right)
                # Right Eye (anatomical): 386 (top), 374 (bottom), 362 (left), 263 (right)
                
                left_ear = self.get_blink_ratio(lms, 159, 145, 33, 133)
                right_ear = self.get_blink_ratio(lms, 386, 374, 362, 263)
                
                current_time = time.time()
                
                # Debug Info for user to see their "Eye Ratio"
                cv2.putText(frame_rgb, f"L-Eye: {left_ear:.3f}", (30, 100), FONT, 0.7, (0, 255, 255), 2)
                cv2.putText(frame_rgb, f"R-Eye: {right_ear:.3f}", (w_cam - 200, 100), FONT, 0.7, (0, 255, 255), 2)
                
                # Detect Left Click (Left Eye)
                if left_ear < EYE_AR_THRESH:
                    cv2.putText(frame_rgb, "BLAST!", (100, 150), FONT, 1, (0, 0, 255), 2)
                    if current_time - self.last_left_click > CLICK_COOLDOWN:
                        pyautogui.click()
                        self.last_left_click = current_time
                        click_status = "L-CLICKED"

                # Detect Right Click (Right Eye)
                if right_ear < EYE_AR_THRESH:
                    cv2.putText(frame_rgb, "BLAST!", (w_cam - 250, 150), FONT, 1, (0, 0, 255), 2)
                    if current_time - self.last_right_click > CLICK_COOLDOWN:
                        pyautogui.rightClick()
                        self.last_right_click = current_time
                        click_status = "R-CLICKED"

        # ---------------------------------------------------------
        # 2. HAND TRACKING (Mouse Move)
        # ---------------------------------------------------------
        hand_results = self.hands.process(frame_rgb)
        mouse_status = "Mouse: Hand Not Detected"

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_rgb, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )

                # Get coordinates for Index (8) and Thumb (4)
                x1, y1 = hand_landmarks.landmark[8].x * w_cam, hand_landmarks.landmark[8].y * h_cam
                x2, y2 = hand_landmarks.landmark[4].x * w_cam, hand_landmarks.landmark[4].y * h_cam

                # Calculate distance
                length = math.hypot(x2 - x1, y2 - y1)

                # Mouse Logic (Relative movement)
                if length < 40: # Pinched
                    if self.last_hand_pos is None:
                        # Starting a new move: set the initial anchor point
                        self.last_hand_pos = (x1, y1)
                    else:
                        # Calculate movement from previous frame
                        dx = (x1 - self.last_hand_pos[0]) * self.sensitivity
                        dy = (y1 - self.last_hand_pos[1]) * self.sensitivity
                        
                        # Move mouse relatively from current position
                        pyautogui.moveRel(dx, dy)
                        
                        # Update the anchor point for the next frame
                        self.last_hand_pos = (x1, y1)

                    cv2.circle(frame_rgb, (int(x1), int(y1)), 15, (0, 255, 0), cv2.FILLED)
                    mouse_status = "Mouse: Moving (Relative)"
                else:
                    # Fingers not touching: clear the anchor point
                    self.last_hand_pos = None
                    cv2.circle(frame_rgb, (int(x1), int(y1)), 15, (255, 0, 0), cv2.FILLED)
                    mouse_status = "Mouse: Stopped (Clutch)"

                cv2.line(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)

        # UI Overlay
        cv2.putText(frame_rgb, "AI Hand & Eye Control (Relative)", (30, 50), FONT, 1.0, (255, 255, 255), 2)
        
        info_str = f"{mouse_status}"
        if click_status:
            info_str += f" | {click_status}"
            
        self.status_label.config(text=info_str)
        
        # Convert to Tkinter image and scale to window size
        img = Image.fromarray(frame_rgb)
        
        # Get current window dimensions
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height() - 80 # Leave space for the status label
        
        # Resize image to fit window (maintaining performance)
        if win_w > 100 and win_h > 100:
            img = img.resize((win_w, win_h), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = HandControlApp(root)

    def on_close():
        app.stop()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
