import cv2
import numpy as np
import os

def perform_image_subtraction(image_path, save_folder):

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found:", image_path)
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.absdiff(gray, template)
    _, thresh = cv2.threshold(
        diff, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )


    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    os.makedirs(save_folder, exist_ok=True)

    # Save with original filename
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]

    cv2.imwrite(os.path.join(save_folder, f"{name_without_ext}_diff.jpg"), diff)
    cv2.imwrite(os.path.join(save_folder, f"{name_without_ext}_mask.jpg"), cleaned)

    print("Subtraction done for:", base_name)
