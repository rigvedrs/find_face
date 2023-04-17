import sys
import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

def pdf_to_image(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return faces

def save_faces(image, faces, output_folder, base_name):
    for i, (x, y, w, h) in enumerate(faces):
        face = image.crop((x, y, x+w, y+h))
        face.save(os.path.join(output_folder, f"{base_name}_face_{i}.png"))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} PDF_FILE OUTPUT_FOLDER")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = pdf_to_image(pdf_path)

    for idx, img in enumerate(images):
        faces = detect_faces(img)
        if len(faces) > 0:
            save_faces(img, faces, output_folder, f"page_{idx}")
