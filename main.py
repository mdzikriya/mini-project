import cv2
import time
import tkinter as tk
from PIL import Image, ImageTk
from gesture_detection import detect_hand_gesture, detect_head_gesture
from text_to_speech import speak
from communication_dict import communication_dict

class GestureApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 0

        self.vid = VideoCapture(self.video_source)
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        self.btn_start = tk.Button(window, text="Start", width=50, command=self.start_video)
        self.btn_start.pack(anchor=tk.CENTER, expand=True)
        
        self.btn_stop = tk.Button(window, text="Stop", width=50, command=self.stop_video)
        self.btn_stop.pack(anchor=tk.CENTER, expand=True)
        
        self.delay = 15
        self.update()
        self.window.mainloop()

    def start_video(self):
        self.vid.start()

    def stop_video(self):
        self.vid.stop()

    def update(self):
        is_opened, frame = self.vid.get_frame()

        if is_opened:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

class VideoCapture:
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.vid = cv2.VideoCapture(video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.running = False
        self.last_gesture_time = time.time()
        self.last_response_time = time.time()
        self.gesture = "Unknown Gesture"

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def get_frame(self):
        if self.vid.isOpened() and self.running:
            success, frame = self.vid.read()
            if success:
                frame = cv2.flip(frame, 1)
                hand_gesture, frame = detect_hand_gesture(frame)
                head_gesture, frame = detect_head_gesture(frame)

                if hand_gesture:
                    self.gesture = communication_dict.get(str(hand_gesture), "Unknown Gesture")
                elif head_gesture:
                    self.gesture = communication_dict.get(head_gesture, "Unknown Gesture")
                else:
                    self.gesture = "Unknown Gesture"

                x, y = 50, 50
                cv2.putText(frame, self.gesture, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                current_time = time.time()
                if current_time - self.last_gesture_time > 1.5:
                    if current_time - self.last_response_time > 4:
                        speak(self.gesture)
                        self.last_gesture_time = current_time
                        self.last_response_time = current_time

                return (success, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (success, None)
        else:
            return (False, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

if __name__ == '__main__':
    root = tk.Tk()
    app = GestureApp(root, "Gesture Recognition App")
