import tkinter as tk
from tkinter import filedialog, messagebox
import cv2 as cv
import numpy as np
import imutils
import easyocr



class NumberPlateDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Number Plate Detection")
        self.create_widgets()
        self.blur_factors = range(1, 20, 2)  # Adjust range of blur factors as needed

    def create_widgets(self):
        self.camera_button = tk.Button(self.root, text="Use Camera", command=self.use_camera)
        self.camera_button.pack(pady=20)

        self.file_button = tk.Button(self.root, text="Select Image File", command=self.select_file)
        self.file_button.pack(pady=20)

        self.quit_button = tk.Button(self.root, text="Quit", command=self.root.quit)
        self.quit_button.pack(pady=20)

    def rescale_frame(self, frame, scale=0.5):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)
        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    def correct_ocr_mistakes_with_confidence(self, result):
        corrections = {}
        corrected_texts = []

        for detection in result:
            text = detection[-2]
            confidence = detection[-1]

            if confidence > 0.5:  # Adjust threshold as needed
                # Generate all possible corrected texts based on corrections dictionary
                possible_texts = self.generate_possible_texts(text, corrections)
                corrected_texts.extend(possible_texts)
            else:
                corrected_texts.append(text)

        return corrected_texts

    def generate_possible_texts(self, text, corrections):
        possible_texts = [text]

        for char_index, char in enumerate(text):
            if char in corrections:
                new_char = corrections[char]
                new_texts = []
                for possible_text in possible_texts:
                    new_text = possible_text[:char_index] + new_char + possible_text[char_index + 1:]
                    new_texts.append(new_text)
                possible_texts.extend(new_texts)

        return possible_texts

    def draw_stylish_text(self, img, text, position, font=cv.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
        """Draw stylish text with a background rectangle."""
        text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x, text_y = position
        bg_x1, bg_y1 = text_x - 10, text_y - 10
        bg_x2, bg_y2 = text_x + text_size[0] + 10, text_y + text_size[1] + 10
        cv.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        cv.putText(img, text, (text_x, text_y + text_size[1]), font, font_scale, text_color, font_thickness, cv.LINE_AA)

    def process_image(self, img, run_ocr):
        # Convert the image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        best_result = None
        best_blur_factor = None

        blur_factors = range(3,31,2)

        for blur_factor in self.blur_factors:
            # Reduce noise with Gaussian blur
            blur = cv.GaussianBlur(gray, (blur_factor, blur_factor), 0)

            # Edge detection
            edged = cv.Canny(blur, 60, 100)



            # Find contours
            keypoints = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(keypoints)
            contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

            # Find the number plate contour
            location = None
            for contour in contours:
                approx = cv.approxPolyDP(contour, 15, True)
                if len(approx) == 4:
                    location = approx
                    break

            if location is not None:
                # Create a mask for the number plate
                mask = np.zeros(gray.shape, np.uint8)
                cv.drawContours(mask, [location], 0, 255, -1)
                new_image = cv.bitwise_and(img, img, mask=mask)

                # Crop the number plate region
                (x, y) = np.where(mask == 255)
                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

                if run_ocr:
                    # Read text using EasyOCR
                    reader = easyocr.Reader(['en'], gpu=False)
                    result = reader.readtext(cropped_image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

                    if result:
                        corrected_texts = self.correct_ocr_mistakes_with_confidence(result)

                        # Determine best result based on criteria (e.g., number of contours, size of contour, etc.)
                        if best_result is None or len(contours) > len(best_result):
                            best_result = corrected_texts
                            best_blur_factor = blur_factor

            if best_result and len(best_result) >= 1:
                break  # Exit loop early if a satisfactory result is found

        if best_result:
            # Draw the bounding box for the best result
            cv.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 2)

            # Display detected text
            for idx, text in enumerate(best_result):
                text_x = location[0][0][0] + (location[2][0][0] - location[0][0][0]) // 2
                text_y = location[0][0][1] - 30 if location[0][0][1] - 30 > 10 else location[0][0][1] + 30
                self.draw_stylish_text(img, text, (text_x, text_y), font_scale=0.8, font_thickness=2, text_color=(255, 255, 255), bg_color=(0, 255, 0))

            return img, best_result
        else:
            return img, None

    def use_camera(self):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return

        frame_count = 0
        detected_text = None

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Could not read frame from camera.")
                break

            # Resize the frame to improve performance
            frame = self.rescale_frame(frame, scale=0.5)

            # Process the frame for number plate detection
            run_ocr = frame_count % 10 == 0  # Run OCR every 10 frames
            processed_frame, detected_text = self.process_image(frame, run_ocr)

            # Display the frame with detection
            cv.imshow('Number Plate Detection', processed_frame)

            # Break loop on 'q' key press or if number plate is detected
            if cv.waitKey(1) & 0xFF == ord('q') or detected_text:
                break

            frame_count += 1

        cap.release()
        cv.destroyAllWindows()

        if detected_text:
            self.display_detected_text(detected_text)
        else:
            messagebox.showinfo("No Number Plate Detected", "No Number Plate Detected!")

    def select_file(self):
        file_path = filedialog.askopenfilename(title="Select an image file",
                                               filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if not file_path:
            messagebox.showinfo("No File Selected", "No file selected. Exiting.")
            return
        img = cv.imread(file_path)
        if img is None:
            messagebox.showerror("Error", f"Unable to load image from {file_path}.")
            return

        # Display the original image
        cv.imshow('Original Image', img)
        cv.waitKey(0)

        # Process the image to detect number plate
        processed_img, detected_text = self.process_image(img, run_ocr=True)

        # Display the processed image
        cv.imshow('Processed Image', processed_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

        if detected_text:
            self.display_detected_text(detected_text)
        else:
            messagebox.showinfo("No Number Plate Detected", "No Number Plate Detected!")

    def display_detected_text(self, detected_text):
        result_window = tk.Toplevel(self.root)
        result_window.title("Detected Number Plate")

        detected_label = tk.Label(result_window, text="Detected Number Plate:")
        detected_label.pack(pady=10)

        for text in detected_text:
            text_label = tk.Label(result_window, text=text, font=("Helvetica", 16, "bold"))
            text_label.pack(pady=5)

        close_button = tk.Button(result_window, text="Close", command=result_window.destroy)
        close_button.pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = NumberPlateDetectorApp(root)
    root.mainloop()
