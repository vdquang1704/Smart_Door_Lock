import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

IMG_PATH = "static/data/test_images/"


def capture(usr_name):
    count = 20
    USR_PATH = os.path.join(IMG_PATH, usr_name)
    leap = 1
    mtcnn = MTCNN(
        margin=20,
        keep_all=False,
        select_largest=True,
        post_process=False,
        device=device,
    )
    cap = cv2.VideoCapture("http://192.168.0.123:81/stream")
    saved = False
    while cap.isOpened() and count:
        isSuccess, frame = cap.read()
        if not (saved):
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            save_path = str("static/dataset" + "/{}.jpg".format(usr_name))
            f = open(save_path, "wb")
            f.write(frame)
            f.close()
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            saved = True
            continue
        if mtcnn(frame) is not None and leap % 2:
            path = str(
                USR_PATH
                + "/{}.jpg".format(
                    str(datetime.now())[:-7].replace(":", "-").replace(" ", "-")
                    + str(count)
                )
            )
            face_img = mtcnn(frame, save_path=path)
            count -= 1
        leap += 1
        # print(count)
        # cv2.imshow('Face Capturing', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    return True
