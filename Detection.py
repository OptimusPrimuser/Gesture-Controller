import cv2
import numpy as np
import argparse
import imutils
import time

import os

LABELS = ["cursor", "fist", "Left Click", "Palm","Right Click"]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = "yolov4-obj_best.weights"
configPath = "yolov4-obj.cfg"

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detection(file):
    """ import pafy
    url="https://www.youtube.com/watch?v=seG9J49bBYI"
    vid=pafy.new(url)
    vid=vid.videostreams[4]
    vs = cv2.VideoCapture(vid.url)  """
    vs = cv2.VideoCapture(file)
    writer = None
    (W, H) = (None, None)
    
    headingPOSText=open("heading.txt","w")

    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1
    no = 0
    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame,
                                    1 / 255.0, (320, 320),
                                    swapRB=True,
                                    crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []
        layerOutputs=np.concatenate(layerOutputs,axis=0)
        scores = layerOutputs[:,5:]
        observe=np.max(scores)
        confidence =np.max(scores,axis=1)
        temp=confidence>0.3
        layerOutputs=layerOutputs[temp]
        confidence=confidence[temp]
        scores = scores[temp]
        classID = np.argmax(scores,axis=1)
        box=(layerOutputs[:,:4] * np.array([W, H, W, H])).astype(np.int) 
        box[:,0]=(box[:,0]-(box[:,2] / 2)).astype(np.int)
        box[:,1]=(box[:,1]-(box[:,3] / 2)).astype(np.int)
        if cv2.waitKey(1)==ord("q"):
                headingPOSText.write("X\n")
                break
        try:
            idxs = cv2.dnn.NMSBoxes(box.tolist(), confidence.tolist(), 0.3, 0.1) 
            print(idxs)
        except :
            writer.write(frame)
            cv2.imshow("",frame)
            print(no)
            no = no + 1
            continue
        
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (box[i][0], box[i][1])
                (w, h) = (box[i][2], box[i][3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

                #for heading.txt not needed to be used in this project
                if LABELS[classID[i]]=="heading":
                    tempText=str(no)+","+str(y)+","+str(y+h)+","+str(x)+","+str(x+w)+"\n"
                    headingPOSText.write(tempText)
                text = "{}: {:.4f}".format(LABELS[classID[i]], confidence[i])
                
                yield { 
                        "Action":LABELS[classID[i]] ,
                        "frameSize":(W,H), 
                        "postion":(W-int(x+(w/2)), int(y+(h/2))) 
                      }
                
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            255, 2)

        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("result.mp4", fourcc, 30,
                                    (frame.shape[1], frame.shape[0]), True)

            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total))

        # write the output frame to disk
        writer.write(frame)
        #print(no)
        no = no + 1
        cv2.imshow("",frame)
    # release the file pointers
    print("[INFO] cleaning up...")
    headingPOSText.write("X\n")
    headingPOSText.close()
    writer.release()
    vs.release()
    cv2.destroyAllWindows()
    quit()


