
from contextlib import redirect_stderr
from email.mime import image
from unittest.result import failfast
from urllib import request
from flask import Flask, render_template, Response, url_for, redirect, request
import cv2
import numpy as np
import time
import random
import string
from paho.mqtt import client as mqtt_client
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
from PIL import Image
from face_capture import capture
from update_faces import update_face
from firebase import putData, getFBData


app = Flask(__name__)

power = pow(10, 6)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)
model.eval()

mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)
camera_url = "http://192.168.0.102:6677/videofeed?username=&password="
camera = cv2.VideoCapture(camera_url)
camera1 = cv2.VideoCapture(camera_url)


# Create arrays of known face encodings and their names
known_face_names = []
known_face_path = []
known_face_access = []
# Initialize some variables
face_locations = []
process_this_frame = True
face_names = []
face_detected_time = []
time_detected = 0
start_recog = False
run_bg = False
lastest_name = ""
admin_open = False
data = []
datas = []
ctime = [0, 0]


broker = '192.168.0.101'
port = 1883
topic1 = "doorlock/open"
topic2 = "doorlock/face_infor"
subTopic = "doorlock/capture"

frame_size = (640, 480)
IMG_PATH = 'static/data/test_images'
DATA_PATH = 'static/data'

# generate client ID with pub prefix randomly
sub_client_id = f'python-mqtt-{random.randint(0, 1000)}'
pub_client_id = f'python-mqtt-{random.randint(0, 1000)}'
username = 'emqx'
password = 'public'


def connect_mqtt(id: string):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(id + " Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def getData():
    data.clear()
    known_face_names.clear()
    known_face_path.clear()
    known_face_access.clear()
    with open('static/name.txt') as f:
        for line in f:
            item = [i for i in line.split(",")]
            data.append(item)
            known_face_names.append(item[0])
            known_face_path.append(item[1])
            known_face_access.append(item[2])

def publish(client, message: string, topic: string):
    msg_count = 0
    # while True:
    #     time.sleep(1)
    result = client.publish(topic, message)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{message}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")
    msg_count += 1


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        data = msg.payload
        message = data.decode("utf-8")
        # if(msg.topic == topic2):
        if(msg.topic == topic1):
            if message == "open":
                print("test!!")
                if not(admin_open):
                    admin_open = True
                else:
                    success, frame = camera1.read()
                    if success:
                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame = buffer.tobytes()
                        i = round(time.time() * 1000)
                                # print(i)
                        f = open(f"static/temp_image/temp.jpg", "wb")
                        f.write(frame)
                        f.close()
                        putData("Open by admin", "Admin", "static/temp_image/temp.jpg")
            if message == "start":
                    print("bg run")
                    run_bg = True
                    start_recog = True
                    background()
        if(msg.topic == subTopic):
            sub_msg = message.split(",")
            updateFaces(sub_msg[0], sub_msg[1])
        print(f"From topic {msg.topic} got message {message}")
        # f = open('output.jpg', "wb")
        # f.write(msg.payload)
        # print("Image Received")
        # f.close()

    client.subscribe(subTopic)
    client.subscribe(topic1)
    client.on_message = on_message

pubclient = connect_mqtt(pub_client_id)
subclient = connect_mqtt(sub_client_id)
subscribe(subclient)
subclient.loop_start()
pubclient.loop_start()

def trans(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)


def load_faceslist():
    if device == 'cpu':
        embeds = torch.load(DATA_PATH+'/faceslistCPU.pth')
    else:
        embeds = torch.load(DATA_PATH+'/faceslist.pth')
    names = np.load(DATA_PATH+'/usernames.npy')
    return embeds, names


def inference(model, face, local_embeds, threshold=3):
    #local: [n,512] voi n la so nguoi trong faceslist
    embeds = []
    # print(trans(face).unsqueeze(0).shape)
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds)  # [1,512]
    # print(detect_embeds.shape)
                  #[1,512,1]                                      [1,512,n]
    norm_diff = detect_embeds.unsqueeze(-1) - \
    torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    # print(norm_diff)
    # (1,n), moi cot la tong khoang cach euclide so vs embed moi
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1)

    min_dist, embed_idx = torch.min(norm_score, dim=1)
    # print(min_dist*power, names[embed_idx])
    # print(min_dist.shape)
    if min_dist*power > threshold:
        return -1, -1
    else:
        return embed_idx, min_dist.double()


def extract_face(box, img, margin=20):
    face_size = 160
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ]  # tạo margin bao quanh box cũ
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img, (face_size, face_size),
                      interpolation=cv2.INTER_CUBIC)
    face = Image.fromarray(face)
    return face

# Update khuôn mặt
def updateFaces(new_name, rules):
    save_path = str("static/dataset"+'/{}.jpg'.format(new_name))
    with open("static/name.txt", "a") as a_file:
                a_file.write("\n")
                a_file.write(new_name + "," + save_path + "," + rules)
    state = capture(new_name)
    if state:
            getData()
            complete = update_face()
    return complete

# Frame cho video stream
def gen_frames():
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    ptime = 0
    while True:
        isSuccess, frame = camera.read()
        if isSuccess:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            if cv2.waitKey(1)&0xFF == 27:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Frame chạy ở backgound
def background():
    camera1 = cv2.VideoCapture(camera_url)
    # camera1.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    # camera1.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    while True:
        isSuccess, frame = camera1.read()
        # print(isSuccess)
        if isSuccess:
            # print("start")
            if start_recog:
                recog(frame, embeddings, names)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            if not(run_bg):
                break
    
def recog(frame, embeddings, names):
    if(time_detected < 50):
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            name = "Unknown"
            for box in boxes:
                bbox = list(map(int,box.tolist()))
                face = extract_face(bbox, frame)
                idx, score = inference(model, face, embeddings)
                if idx != -1:
                            frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                            score = torch.Tensor.cpu(score[0]).detach().numpy()*power
                            frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                            name = names[idx]
                            
                else:
                    frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                    frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                for i in range(len(face_names)):
                    if(name == face_names[i]):
                        face_detected_time[i] = face_detected_time[i] + 1
                        break
                    if(name != face_names[i] and i == len(face_names)-1):
                        face_names.append(name)
                time_detected = time_detected + 1 
                print(name)
                print(time_detected)
    else:
        name_detected = ""
        print(face_detected_time)
        if (max(face_detected_time)/time_detected) > 0.65:
                    name_detected = face_names[face_detected_time.index(
                        max(face_detected_time))]
        else:
                    name_detected = "Unknown"
        time_detected = 0
        print(face_names)
        face_names.clear()
        face_names.append("Unknown")
        print(face_names)
        for i in range(len(face_detected_time)):
                    face_detected_time[i] = 1
        print(face_detected_time)
        i = round(time.time() * 1000)
        if not(lastest_name == name_detected):
                    lastest_name = name_detected
                    state = False
                    if not(lastest_name == "Unknown"):
                        if (known_face_access[known_face_names.index(lastest_name)] == "admin") or (known_face_access[known_face_names.index(lastest_name)] == "admin\n"):
                            state = True
                    message = '{"time": ' + str(i) + ',"name": "' + \
                        lastest_name + '","state": ' + str(state).lower() + '}'
                    run_bg = False
                    publish(pubclient, message, topic2)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    f = open(f"static/temp_image/temp.jpg", "wb")
                    f.write(frame)
                    f.close()
                    if state:
                        publish(pubclient, "open", topic1)
                        putData(lastest_name, "Admin", "static/temp_image/temp.jpg")
                    else:
                        putData(lastest_name, "Guess", "static/temp_image/temp.jpg")
                    lastest_name = ""
                    start_recog = False
                    admin_open = False
                    
        
@app.route('/start')
def start():
    start_recog = False
    lastest_name = ""
    publish(pubclient, "open", topic1)
    return "Nothing"

@app.route('/reload')
def reload():
    update_face()
    return redirect(url_for("storage"))

def myFunc(e):
    return e[2]


@app.route('/')
@app.route('/index.html')
def main_page():
    return render_template('index.html')

@app.route('/add_new.html')
def history():
    datas = getFBData(20)
    return render_template('add_new.html', data= datas)

@app.route('/storage.html')
def storage():
    data.sort(key=myFunc)
    return render_template('storage.html', data=data)


@app.route('/capture.html', methods=["POST", "GET"])
def addNew():    
    if request.method == "POST":
        new_name = request.form["nm"]
        rules = request.form["rules"]
        complete = updateFaces(new_name, rules)
        if complete:
            return redirect(url_for("storage"))
        else:
            return render_template('capture.html')
    else: 
        return render_template('capture.html')
        
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    embeddings, names = load_faceslist()
    getData()
    face_detected_time = [1]*(len(known_face_names)+1)
    face_names.append("Unknown")
    app.run()
