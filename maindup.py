import pytesseract
import cv2
import numpy as np

# Set your Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

myconfig = r"--psm 6 --oem 3"

def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)

    # Resize for better OCR accuracy
    scale_percent = 150  # Increase size by 150%
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise image
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )

    # Morphological operations to remove small noise
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Save preprocessed images for debugging
    cv2.imwrite("temp/gray.png", gray)
    cv2.imwrite("temp/denoised.png", denoised)
    cv2.imwrite("temp/thresh.png", thresh)
    cv2.imwrite("temp/clean.png", clean)

    return img, clean

def extract_text(image_path):
    img, preprocessed_img = preprocess_image(image_path)

    # Finding contours
    cnts, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > 500:  # Ignore small areas
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("temp/boxes.png", img)

    # Extract text using Tesseract
    text = pytesseract.image_to_string(preprocessed_img, config=myconfig)

    print("Extracted Text:\n", text)
    return img, text

