from flask import Flask, render_template, Response
import cv2
import torch
import socket
import pytesseract
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression

app = Flask(__name__)

# Load YOLOv5 model
model_path = './runs/train/exp19/weights/best.pt'  # Update with actual model path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(model_path)
model.eval()

# OpenCV video capture
video_capture = cv2.VideoCapture(0)  # Default camera

# Tesseract OCR configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\DSWD\AppData\Local\Programs\Tesseract-OCR\tesseract'

# Object classes
classes = ["plates", "vehicle"]

def generate_frames():
    processing_img = False
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            if not processing_img:
                # Preprocess the frame for inference
                img = frame.copy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).float() / 255.0
                img = img.permute(2, 0, 1).unsqueeze(0)

                # Run inference with YOLOv5
                with torch.no_grad():
                    detections = model(img.to(device))[0]
                    detections = non_max_suppression(detections, conf_thres=0.8, iou_thres=0.4)[0]

                if detections is not None:
                    for detection in detections:
                        x1, y1, x2, y2, conf, cls = map(int, detection[:6])

                        class_name = classes[cls] if cls < len(classes) else 'Unknown'

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 58, 32), 1)
                        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 58, 32), 2)
                        
                        if class_name == "plates":
                            processing_img = True
                            # Crop the detected object from the frame
                            cropped_object = frame[y1+8:y2-8, x1+6:x2-5]
                            cropped_object_gray = enhance_image(cropped_object)
                            cv2.imwrite(f'cropped_{class_name}.jpg', cropped_object_gray)
                            extracted_text = pytesseract.image_to_string(cropped_object_gray, lang='eng', config='--psm 6')
                            print('Plate Number:', extracted_text)
                            processing_img = False

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('bootstrap-5.3.html', host_ip=get_host_ip())

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def enhance_image(image):
    # Convert image to grayscale and apply sharpening filter
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    return sharpened

def get_host_ip():
    # Get host IP address
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

@app.route('/host-ip-addr')
def get_ip_address():
    return get_host_ip()

if __name__ == '__main__':
    app.run(debug=True)
