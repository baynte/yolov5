from flask import Flask, render_template, Response
import cv2
import torch
import socket
import time
from torchvision.transforms import functional as F
from models.experimental import attempt_load
from utils.general import non_max_suppression
# from utils.plots import plot_one_box

app = Flask(__name__)

# Load the YOLOv5 model
model_path = './runs/train/exp19/weights/best.pt'  # Update with the actual path to your model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(model_path)
model.eval()

# OpenCV video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera, change to a different number if needed

classes = ["plates", "vehicle"]


def generate_frames():
    processing_img = False
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            if not processing_img:
                # Preprocess the frame
                img = frame.copy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).float()
                img /= 255.0
                img = img.permute(2, 0, 1).unsqueeze(0)

                # Run inference
                with torch.no_grad():
                    detections = model(img.to(device))[0]
                    detections = non_max_suppression(detections, conf_thres=0.9, iou_thres=0.4)[0]

                if detections is not None:
                    for detection in detections:
                        x1, y1, x2, y2, conf, cls = detection
                        x1, y1, x2, y2 = map(int, detection[:4])

                        class_name = classes[int(cls)] if int(cls) < len(classes) else 'Unknown'


                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 58, 32), 1)

                        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 58, 32), 2)
                        
                        if class_name == "plates":
                            processing_img = True
                            time.sleep(10)
                            processing_img = False
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # yield frame in byte format

@app.route('/')
def index():
    return render_template('bootstrap-5.3.html', host_ip=get_host_ip())

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_host_ip():
    # Get the hostname of the server
    hostname = socket.gethostname()
    # Get the IP address corresponding to the hostname
    ip_address = socket.gethostbyname(hostname)
    return ip_address

@app.route('/host-ip-addr')
def get_ip_address():
    return get_host_ip()

if __name__ == '__main__':
    app.run(debug=True)
