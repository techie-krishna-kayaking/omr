#  pip install pandas numpy cv2
import cv2
import numpy as np
import pandas as pd

# Load the image
image_path = "C:\\Users\\Administrator\\PycharmProjects\\omr\\img\\test4.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply GaussianBlur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological operations to close gaps in the circles
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define a function to check if a contour is a filled bubble
def is_filled_bubble(contour):
    area = cv2.contourArea(contour)
    if 50 < area < 300:  # These values might need tuning
        return True
    return False

# Sort contours from top to bottom, then left to right
def sort_contours(cnts):
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes),
                                         key=lambda b: (b[1][1], b[1][0])))
    return cnts, bounding_boxes

contours, bounding_boxes = sort_contours(contours)

# Initialize an empty list to store the results
results = []

# Define question parameters
num_questions = 10
num_choices_per_question = 4
choices = ['A', 'B', 'C', 'D']

# Process the contours
question_number = 1
answers = []

for i in range(0, len(contours), num_choices_per_question):
    current_question_contours = contours[i:i+num_choices_per_question]
    bubble_filled = [0] * num_choices_per_question
    for j, contour in enumerate(current_question_contours):
        if is_filled_bubble(contour):
            bubble_filled[j] = 1
    if sum(bubble_filled) == 1:
        answer_index = bubble_filled.index(1)
        answer = choices[answer_index]
    else:
        answer = ''  # In case of no fill or multiple fills
    answers.append((question_number, answer))
    question_number += 1

# Create a DataFrame from the results
df = pd.DataFrame(answers, columns=['Question', 'Answer'])

# Save the DataFrame to a CSV file
csv_path = "C:\\Users\\Administrator\\PycharmProjects\\omr\\output.csv"
df.to_csv(csv_path, index=False)

print(f"Answers saved to {csv_path}")
