import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Initialize Hand Detector and Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

# Parameters
offset = 20
imgSize = 300
labels = ["A", "B", "C"]

# Tkinter App Setup
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.geometry("800x600")

# Labels and Buttons in UI
label_text = tk.StringVar()
label_text.set("Recognized Gesture: None")
gesture_label = Label(root, textvariable=label_text, font=("Helvetica", 20))
gesture_label.pack()

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

cap = cv2.VideoCapture(0)

def update_frame():
    success, img = cap.read()
    if success:
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop.size > 0:
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                label_text.set(f"Recognized Gesture: {labels[index]}")
                cv2.rectangle(
                    imgOutput,
                    (x - offset, y - offset - 50),
                    (x - offset + 200, y - offset),
                    (34, 139, 34),
                    cv2.FILLED
                )
                cv2.putText(imgOutput, labels[index], (x, y - 27), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (34, 139, 34), 4)

        # Convert image to PhotoImage format
        imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        imgRGB = Image.fromarray(imgRGB)
        imgtk = ImageTk.PhotoImage(image=imgRGB)

        # Update canvas with new frame
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.imgtk = imgtk

    # Schedule next frame update
    root.after(10, update_frame)

# Start and Stop Buttons
def start_camera():
    update_frame()

def stop_camera():
    cap.release()
    root.destroy()

start_button = tk.Button(root, text="Start", command=start_camera, font=("Helvetica", 14), bg="green", fg="white")
start_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=stop_camera, font=("Helvetica", 14), bg="red", fg="white")
exit_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
