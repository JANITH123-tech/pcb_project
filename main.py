import os
from roi_from_xml import extract_rois_from_xml
from subtraction import perform_image_subtraction

# ==============================
# PATH CONFIGURATION
# ==============================

xml_root = "PCB_DATASET/Annotations"
image_folder = "PCB_DATASET/images"

roi_save_folder = "outputs/rois"
annotated_folder = "outputs/annotated"
subtraction_root = "outputs/subtraction"

# ==============================
# ROI EXTRACTION FROM ALL XML FILES
# ==============================

print("Starting ROI Extraction...")

total_rois = 0

for root, dirs, files in os.walk(xml_root):
    for file in files:
        if file.endswith(".xml"):
            xml_path = os.path.join(root, file)

            total_rois += extract_rois_from_xml(
                xml_path,
                image_folder,
                roi_save_folder,
                annotated_folder
            )

print("===================================")
print("ROI Extraction Completed!")
print("Total ROIs extracted:", total_rois)
print("===================================")


# ==============================
# IMAGE SUBTRACTION FOR ALL IMAGES
# ==============================

print("Starting Subtraction for All Images...")

for root, dirs, files in os.walk(image_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            image_path = os.path.join(root, file)

            # Maintain same class-wise folder structure
            relative_path = os.path.relpath(root, image_folder)
            save_path = os.path.join(subtraction_root, relative_path)

            perform_image_subtraction(image_path, save_path)

print("===================================")
print("All Images Subtraction Completed!")
print("===================================")
