import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import mediapipe as mp
import math
import screen_brightness_control as sbc
import numpy as np

# ---------------- CONFIG ----------------
CAMERA_INDEX = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ---------------------------------------

class HandControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Brightness Control")
        # Increase size for better visibility
        self.root.geometry("1000x800")
        self.root.resizable(False, False)

        self.video_label = Label(root)
        self.video_label.pack()

        self.status_label = Label(
            root,
            text="Initializing camera & hand tracking...",
            font=("Arial", 14),
            fg="blue"
        )
        self.status_label.pack(pady=10)

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible")

        # Mediapipe Hands Init
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.running = True
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Failed to read camera")
            return

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = self.hands.process(frame_rgb)

        img_h, img_w, _ = frame.shape
        
        # Default status
        status_text = "Status: Detecting Hands..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_rgb, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )

                # Get coordinates for Thumb (4) and Index (8)
                x1, y1 = int(hand_landmarks.landmark[4].x * img_w), int(hand_landmarks.landmark[4].y * img_h)
                x2, y2 = int(hand_landmarks.landmark[8].x * img_w), int(hand_landmarks.landmark[8].y * img_h)
                
                # Midpoint for center of line
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw points and line
                cv2.circle(frame_rgb, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame_rgb, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(frame_rgb, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(frame_rgb, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

                # Calculate length
                length = math.hypot(x2 - x1, y2 - y1)

                # Map length to brightness range [0, 100]
                # Adjust min/max length based on camera distance usually 30-250 range works for webcams
                min_len = 30
                max_len = 250
                brightness = np.interp(length, [min_len, max_len], [0, 100])
                
                # Set System Brightness
                # Wrap in try-except to avoid spamming errors if library fails on some systems
                try:
                    sbc.set_brightness(int(brightness))
                except Exception:
                    pass

                # Visual Feedback: Brightness Bar
                bar_x, bar_y = 50, 150
                bar_w, bar_h = 35, 400
                
                # Calculate filled height
                filled_height = np.interp(brightness, [0, 100], [0, bar_h])
                
                # Draw Bar background
                cv2.rectangle(frame_rgb, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), 3)
                # Draw Bar fill (inverted Y because 0,0 is top-left)
                cv2.rectangle(frame_rgb, (bar_x, int(bar_y + bar_h - filled_height)), (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), cv2.FILLED)
                
                # Percentage Text
                cv2.putText(frame_rgb, f'{int(brightness)}%', (bar_x, bar_y - 20), FONT, 1, (255, 0, 0), 3)
                
                status_text = f"Status: Brightness Control Active | Level: {int(brightness)}%"

        # UI Overlay (Title)
        cv2.putText(
            frame_rgb,
            "Hand Gesture Brightness Control",
            (30, 50),
            FONT,
            1.0,
            (255, 255, 255),
            2
        )

        # Convert to Tkinter image
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.status_label.config(text=status_text)
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
