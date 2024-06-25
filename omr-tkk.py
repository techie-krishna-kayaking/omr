import cv2
import numpy as np
import os
import csv

# When the four corners are identified, we will do a four-point
# perspective transform such that the outmost points of each
# corner map to a TRANSF_SIZE x TRANSF_SIZE square image.
TRANSF_SIZE = 512

# Answer sheet properties.
N_QUESTIONS = 10

ANSWER_SHEET_WIDTH = 740
ANSWER_SHEET_HEIGHT = 1049

ANSWER_PATCH_HEIGHT = 50
ANSWER_PATCH_HEIGHT_WITH_MARGIN = 80
ANSWER_PATCH_LEFT_MARGIN = 200
ANSWER_PATCH_RIGHT_MARGIN = 90
FIRST_ANSWER_PATCH_TOP_Y = 200

ALTERNATIVE_HEIGHT = 50
ALTERNATIVE_WIDTH = 50
ALTERNATIVE_WIDTH_WITH_MARGIN = 100

def calculate_contour_features(contour):
    moments = cv2.moments(contour)
    return cv2.HuMoments(moments)

def calculate_corner_features():
    corner_img = cv2.imread('img/corner.png')
    corner_img_gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        corner_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 2:
        raise RuntimeError(
            'Did not find the expected contours when looking for the corner')
    corner_contour = next(ct
                          for i, ct in enumerate(contours)
                          if hierarchy[0][i][3] != -1)
    return calculate_contour_features(corner_contour)

def normalize(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(im_gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 77, 10)

def get_approx_contour(contour, tol=.01):
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def get_contours(image_gray):
    contours, _ = cv2.findContours(
        image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return map(get_approx_contour, contours)

def get_corners(contours):
    corner_features = calculate_corner_features()
    return sorted(
        contours,
        key=lambda c: features_distance(
                corner_features,
                calculate_contour_features(c)))[:4]

def get_bounding_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)

def features_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))

def draw_point(point, img, radius=5, color=(0, 0, 255)):
    cv2.circle(img, tuple(point), radius, color, -1)

def sort_points_counter_clockwise(points):
    origin = np.mean(points, axis=0)
    def positive_angle(p):
        x, y = p - origin
        ang = np.arctan2(y, x)
        return 2 * np.pi + ang if ang < 0 else ang
    return sorted(points, key=positive_angle)

def get_outmost_points(contours):
    all_points = np.concatenate(contours)
    return get_bounding_rect(all_points)

def perspective_transform(img, points):
    source = np.array(
        points,
        dtype="float32")
    dest = np.array([
        [TRANSF_SIZE, TRANSF_SIZE],
        [0, TRANSF_SIZE],
        [0, 0],
        [TRANSF_SIZE, 0]],
        dtype="float32")
    transf = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(img, transf, (TRANSF_SIZE, TRANSF_SIZE))
    return warped

def sheet_coord_to_transf_coord(x, y):
    return list(map(lambda n: int(np.round(n)), (
        TRANSF_SIZE * x / ANSWER_SHEET_WIDTH,
        TRANSF_SIZE * y / ANSWER_SHEET_HEIGHT
    )))

def get_question_patch(transf, question_index):
    tl = sheet_coord_to_transf_coord(
        ANSWER_PATCH_LEFT_MARGIN,
        FIRST_ANSWER_PATCH_TOP_Y + ANSWER_PATCH_HEIGHT_WITH_MARGIN * question_index
    )
    br = sheet_coord_to_transf_coord(
        ANSWER_SHEET_WIDTH - ANSWER_PATCH_RIGHT_MARGIN,
        FIRST_ANSWER_PATCH_TOP_Y +
        ANSWER_PATCH_HEIGHT +
        ANSWER_PATCH_HEIGHT_WITH_MARGIN * question_index
    )
    return transf[tl[1]:br[1], tl[0]:br[0]]

def get_question_patches(transf):
    for i in range(N_QUESTIONS):
        yield get_question_patch(transf, i)

def get_alternative_patches(question_patch):
    for i in range(5):
        x0, _ = sheet_coord_to_transf_coord(ALTERNATIVE_WIDTH_WITH_MARGIN * i, 0)
        x1, _ = sheet_coord_to_transf_coord(ALTERNATIVE_WIDTH +
                                            ALTERNATIVE_WIDTH_WITH_MARGIN * i, 0)
        yield question_patch[:, x0:x1]

def draw_marked_alternative(question_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        ALTERNATIVE_WIDTH * (2 * index + .5),
        ALTERNATIVE_HEIGHT / 2)
    draw_point((cx, cy), question_patch, radius=5, color=(255, 0, 0))

def get_marked_alternative(alternative_patches):
    means = list(map(np.mean, alternative_patches))
    sorted_means = sorted(means)
    if sorted_means[0]/sorted_means[1] > .7:
        return None
    return np.argmin(means)

def get_letter(alt_index):
    return ["A", "B", "C", "D", "E"][alt_index] if alt_index is not None else "N/A"

def get_answers(source_file):
    im_orig = cv2.imread(source_file)
    im_normalized = normalize(im_orig)
    contours = get_contours(im_normalized)
    corners = get_corners(contours)
    outmost = sort_points_counter_clockwise(get_outmost_points(corners))
    normalized_transf = perspective_transform(im_normalized, outmost)
    answers = []
    for i, q_patch in enumerate(get_question_patches(normalized_transf)):
        alt_index = get_marked_alternative(get_alternative_patches(q_patch))
        answers.append(get_letter(alt_index))
    return answers

def process_images(input_folder, output_csv):
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    all_answers = []
    for image_file in image_files:
        file_path = os.path.join(input_folder, image_file)
        answers = get_answers(file_path)
        all_answers.append([image_file] + answers)
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['NAME'] + [f'Q{i+1}' for i in range(N_QUESTIONS)])
        csvwriter.writerows(all_answers)
    return all_answers

if __name__ == '__main__':
    answers = process_images('img', 'output.csv')
    print("Answers stored in 'output.csv':")
    for answer in answers:
        print(answer)
