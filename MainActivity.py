import cv2
import numpy as np

# 웹캠 신호 받기
#videoSignal = cv2.VideoCapture(0)

# while True: 
#     success,img=VideoSignal.read()
#     cv2.imshow('Image',img)
#     cv2.waitKey(1)

#모델 불러오는 부분 / yolo 가중치 파일과 CFG 파일 로드
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg") 

# YOLO NETWORK 재구성
classes = []
with open("yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names=net.getLayerNames()
output_layers=[layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))

img=cv2.imread("room_ser.jpg")
img=cv2.resize(img,None,fx=0.2,fy=0.2)
height,width,channels= img.shape

blob=cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)

# for b in blob:
#     for n,img_blob in enumerate(b):
#         cv2.imshow(str(n),img_blob)

net.setInput(blob)
outs=net.forward(output_layers)

#showing informations on the screen
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 : 
            #Object detected 
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w= int(detection[2]*width)
            h= int(detection[3]*height)

            # Rectangle coordinates
            x= int(center_x - w /2)
            y= int(center_y - h /2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
# print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        color=colors[i]
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2) 
        cv2.putText(img,label,(x,y+30),font,1,color,3)
    
    
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# layer_names = YOLO_net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()] 

# while True:
#     # 웹캠 프레임
#     ret, frame = VideoSignal.read()
#     h, w, c = frame.shape

#     # YOLO 입력


#     class_ids = []
#     confidences = []
#     boxes = []

#     for out in outs:

#         for detection in out:

#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.5:
#                 # Object detected
#                 center_x = int(detection[0] * w)
#                 center_y = int(detection[1] * h)
#                 dw = int(detection[2] * w)
#                 dh = int(detection[3] * h)
#                 # Rectangle coordinate
#                 x = int(center_x - dw / 2)
#                 y = int(center_y - dh / 2)
#                 boxes.append([x, y, dw, dh])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)


#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             score = confidences[i]

#             # 경계상자와 클래스 정보 이미지에 입력
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
#             cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, 
#             (255, 255, 255), 1)

#     cv2.imshow("YOLOv3", frame)

#     if cv2.waitKey(100) > 0:
#         break