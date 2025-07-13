import tkinter as tk
from tkinter import simpledialog, messagebox
import os
import sys
sys.path.append('/mnt/data')
import face_detect1
import faceRecognition

def face_detect():
    folder_name = simpledialog.askstring("Input", "Enter the folder name:")
    if folder_name:
        facedetect1.run_face_detection(folder_name)
    else:
        messagebox.showwarning("Input Error", "Folder name cannot be empty!")

def face_recognize():
    faceRecognition.run_face_recognition()
root = tk.Tk()
root.title("Face Detection & Recognition")
root.geometry("400x200")
label = tk.Label(root, text="Choose an Option", font=("Arial", 16))
label.pack(pady=20)
detect_button = tk.Button(root, text="Face Detect", command=face_detect, width=20, height=2)
detect_button.pack(pady=10)

recognize_button = tk.Button(root, text="Face Recognition", command=face_recognize, width=20, height=2)
recognize_button.pack(pady=10)
root.mainloop()
