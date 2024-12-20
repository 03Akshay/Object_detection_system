import cv2
import argparse
import numpy as np
from sort import Sort

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='path to input video file')
ap.add_argument('-o', '--output', default='', help='path to output video file')
ap.add_argument('-c', '--config', required=True, help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True, help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True, help='path to text file containing class names')
args = ap.parseArgs()

# Function to get output layers
def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Function to draw prediction
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Read class names
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate colors for each class
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Read pre-trained model and config file
net = cv2.dnn.readNet(args.weights, args.config)

# Open video capture
cap = cv2.VideoCapture(args.input)

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
if args.output != '':
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Initialize SORT tracker
tracker = Sort()

# Dictionary to keep track of unique objects
unique_objects = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Prepare input blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set input blob for the network
    net.setInput(blob)

    # Run inference through the network and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # Initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # For each detection from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Prepare detections for tracking
    dets = []
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        x, y, w, h = box
        dets.append([x, y, x + w, y + h, confidences[i]])

    # Update tracker
    dets = np.array(dets)
    tracks = tracker.update(dets)

    # Track unique objects
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = class_ids[indices[0]]  # Assuming the same class for simplicity
        if track_id not in unique_objects:
            unique_objects[track_id] = classes[class_id]
        draw_prediction(frame, class_id, 1, x1, y1, x2, y2)

    # Display and write frame to output video
    cv2.imshow("object detection", frame)
    if args.output != '':
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print the total counts of each unique object class
print("\nTotal unique objects detected:")
object_counts = {class_name: 0 for class_name in classes}
for obj_class in unique_objects.values():
    object_counts[obj_class] += 1

for class_name, count in object_counts.items():
    print(f"{class_name}: {count}")

# Release video capture and writer objects
cap.release()
if args.output != '':
    out.release()

cv2.destroyAllWindows()