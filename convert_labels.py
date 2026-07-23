import os
import xml.etree.ElementTree as ET

# --- CONFIGURATION ---
xml_folder = "train_data/annotations" # Where your old XMLs are
output_folder = "train_data/yolo_labels" # Where the new TXTs will go

# Notice we dropped 'background' and started at 0
classes = {'car': 0, 'bus': 1, 'bike': 2, 'ambulance': 3}

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def convert_box(size, box):
    """Converts absolute [xmin, ymin, xmax, ymax] to normalized [x_center, y_center, width, height]"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    width = box[1] - box[0]
    height = box[3] - box[2]
    
    x_center = x_center * dw
    width = width * dw
    y_center = y_center * dh
    height = height * dh
    return (x_center, y_center, width, height)

print("Starting Conversion...")
for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith('.xml'):
        continue
        
    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    txt_filename = xml_file.replace('.xml', '.txt')
    txt_path = os.path.join(output_folder, txt_filename)
    
    with open(txt_path, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
                
            cls_id = classes[cls]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            
            bb = convert_box((w, h), b)
            # Write to TXT file: class_id x_center y_center width height
            out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

print(f"✅ Converted XMLs to YOLO TXT format in '{output_folder}'")