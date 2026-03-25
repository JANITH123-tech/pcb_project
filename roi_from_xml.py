import cv2
import os
import xml.etree.ElementTree as ET

def extract_rois_from_xml(xml_path, image_folder, save_folder, annotated_folder):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    folder_name = root.find("folder").text

    image_path = os.path.join(image_folder, folder_name, filename)

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found:", image_path)
        return 0

    count = 0

    for obj in root.findall("object"):

        class_name = obj.find("name").text
        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        
        roi = image[ymin:ymax, xmin:xmax]

        if roi.size == 0:
            continue

        
        roi = cv2.resize(roi, (128, 128))

        
        class_folder = os.path.join(save_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

        save_path = os.path.join(class_folder, f"{filename}_{count}.jpg")
        cv2.imwrite(save_path, roi)

        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

        count += 1

    
    os.makedirs(annotated_folder, exist_ok=True)
    annotated_path = os.path.join(annotated_folder, filename)
    cv2.imwrite(annotated_path, image)

    print(f"{count} ROIs extracted from {filename}")

    return count