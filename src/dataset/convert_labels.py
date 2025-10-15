import os
from glob import glob

def polygon_to_yolo_bbox(line):
    parts = line.strip().split()
    if len(parts) < 9:
        return None  
    
    cls = parts[0]
    coords = list(map(float, parts[1:]))
    xs = coords[0::2]
    ys = coords[1::2]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin

    return f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def convert_folder_labels(folder):
    txt_files = glob(os.path.join(folder, "*.txt"))
    print(f"Found {len(txt_files)} label files in {folder}")

    for txt_path in txt_files:
        with open(txt_path, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            bbox_line = polygon_to_yolo_bbox(line)
            if bbox_line:
                new_lines.append(bbox_line + "\n")

        # ghi đè lại file
        with open(txt_path, "w") as f:
            f.writelines(new_lines)

        print(f"✅ Converted {txt_path} ({len(lines)} -> {len(new_lines)} boxes)")



convert_folder_labels(r"D:\Model\LicensePlate\LicensePlateDataset\labels\test")
convert_folder_labels(r"D:\Model\LicensePlate\LicensePlateDataset\labels\train")
convert_folder_labels(r"D:\Model\LicensePlate\LicensePlateDataset\labels\val")
print("✅ All done!")
