from ultralytics import YOLO
import numpy as np
import cv2


CHAR_MAP = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L',
            'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']

def get_char_from_box(box):
    class_id = int(box[5])
    return CHAR_MAP[class_id]

def group_characters_by_rows(boxes, threshold=0.5):
    boxes.sort(key=lambda x: x[1])  
    rows = []
    current_row = [boxes[0]]

    for box in boxes[1:]:
        last_box = current_row[-1]
        y_overlap = min(last_box[1] + last_box[3], box[1] + box[3]) - max(last_box[1], box[1])
        min_height = min(last_box[3], box[3])

        if y_overlap / min_height > threshold:
            current_row.append(box)
        else:
            current_row.sort(key=lambda x: x[0])  # trái → phải
            rows.append(current_row)
            current_row = [box]

    if current_row:
        current_row.sort(key=lambda x: x[0])
        rows.append(current_row)

    return rows

def identify_upper_lower_rows(rows):
    avg_y = [np.mean([b[1] for b in row]) for row in rows]
    order = np.argsort(avg_y)  # nhỏ hơn = hàng trên
    upper_row = rows[order[0]]
    lower_row = rows[order[1]]
    return upper_row, lower_row

def combine_license_plate(upper_row, lower_row=None):

    upper_chars = ''.join(get_char_from_box(b) for b in upper_row)
    if lower_row is not None:
        lower_chars = ''.join(get_char_from_box(b) for b in lower_row)
        return f"{upper_chars}{lower_chars}"  # có dấu "-"
    else:
        return upper_chars


def process_license_plate(results):

    boxes = []
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        w, h = x2 - x1, y2 - y1
        x_center = x1 + w / 2
        y_center = y1 + h / 2
        boxes.append((x_center, y_center, w, h, conf, int(cls)))

    if len(boxes) == 0:
        print("❌ Không phát hiện ký tự nào!")
        return None

    rows = group_characters_by_rows(boxes)
    len_rows = len(rows)
    if len(rows) == 1:

        rows[0].sort(key=lambda x: x[0])
        print(f"🔹 Phát hiện {len(rows)} hàng ký tự, tiến hành phân loại...")
        license_plate = combine_license_plate(rows[0])
       
    else:

        print(f"🔹 Phát hiện {len(rows)} hàng ký tự, tiến hành phân loại...")
        upper_row, lower_row = identify_upper_lower_rows(rows)
        license_plate = combine_license_plate(upper_row, lower_row)
       
    return license_plate, len_rows

