import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import os
import pytesseract  # OCR library
import sympy as sp  # Symbolic mathematics library
import re  # Regular expressions for parsing
import warnings
import google.generativeai as genai

# Suppress deprecated warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Configure Gemini API securely using the environment variable
api_key = os.getenv("GEMINI_API_KEY")  # Retrieve your API key from the environment variable
if not api_key:
    raise ValueError("API key not found. Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro')

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Initialize variables
prev_pos = None
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Initialize canvas with the same size as the frame
expression_submitted = False

def get_hand_info(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lm_list = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lm_list
    return None

def draw(info, prev_pos, canvas):
    fingers, lm_list = info
    current_pos = None

    if fingers == [0, 1, 1, 0, 0]:  # Index and middle fingers up for drawing
        current_pos = (int(lm_list[8][0]), int(lm_list[8][1]))  # Position of index finger tip
        if prev_pos is not None and np.linalg.norm(np.array(prev_pos) - np.array(current_pos)) < 50:
            cv2.line(canvas, prev_pos, current_pos, (255, 255, 255), 10)  # Draw in white for visibility
        prev_pos = current_pos
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up for submission
        return None, canvas  # Do not clear canvas; just indicate submission
    elif fingers == [0, 0, 0, 0, 1]:  # Thumb down for clearing
        return None, np.zeros_like(canvas)  # Clear canvas

    return prev_pos, canvas

def extract_text_from_image(canvas):
    # Convert the canvas to grayscale for better OCR accuracy
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Enhance the image for better OCR results
    gray_canvas = cv2.threshold(gray_canvas, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Use pytesseract to extract text from the image
    extracted_text = pytesseract.image_to_string(gray_canvas, config='--psm 6')
    return extracted_text.strip()

def parse_matrix(text):
    try:
        pattern = r'\[\[.*?\]\]'
        match = re.search(pattern, text)
        if match:
            matrix_str = match.group(0)
            matrix = eval(matrix_str)
            if isinstance(matrix, list):
                return sp.Matrix(matrix)
        return None
    except Exception as e:
        print(f"Error parsing matrix: {e}")
        return None

def parse_expression(text):
    try:
        # Check for matrix pattern first
        matrix = parse_matrix(text)
        if matrix:
            return matrix
        
        # Handle other expressions
        expr = sp.sympify(text)
        return expr
    except Exception as e:
        print(f"Error parsing expression: {e}")
        return None

def calculate_expression(expr):
    try:
        if isinstance(expr, sp.Matrix):
            # Calculate the determinant of the matrix
            determinant = expr.det()
            return f"Determinant: {determinant}"
        else:
            # Evaluate the expression
            result = expr.evalf()
            return f"Result: {result}"
    except Exception as e:
        return f"Error calculating expression: {e}"

def send_to_llm(text):
    try:
        response = model.some_correct_method(text)  
        return response
    except AttributeError:
        print("The method used to generate text is incorrect. Please check the library documentation.")
        return None

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera.")
        break

    img = cv2.flip(img, 1)

    info = get_hand_info(img)
    if info:
        fingers, lm_list = info
        prev_pos, canvas = draw(info, prev_pos, canvas)

        if fingers == [1, 0, 0, 0, 0] and not expression_submitted:  # Thumbs up for submission
            if np.any(canvas):  # Check if the canvas is not blank
                extracted_text = extract_text_from_image(canvas)
                print(f"Extracted Text: {extracted_text}")
                
                # Validate extracted text
                if re.match(r'^[0-9+\-*/().\s]+$', extracted_text):
                    expr = parse_expression(extracted_text)
                    if expr is not None:
                        result = calculate_expression(expr)
                        print(result)
                        # Send the result as text to LLM for further processing
                        llm_response = send_to_llm(result)
                        print(f"LLM Response: {llm_response}")
                    else:
                        print("Failed to parse the expression.")
                else:
                    print("Extracted text is not a valid mathematical expression.")
                
                expression_submitted = True  # Mark expression as submitted
            else:
                print("Canvas is blank. Please draw something before submitting.")

        # Reset expression submitted status if thumbs up is not shown
        if not (fingers == [1, 0, 0, 0, 0]):
            expression_submitted = False

    # Overlay the canvas on the video feed
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    cv2.imshow("Hand Tracking & Drawing", image_combined)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
