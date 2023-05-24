import cv2
import os
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def ext(x,i):
    cam = cv2.VideoCapture(x)
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    tfps = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    # print(fps)

    try:
        # creating a folder named data
        dest = 'VIDEOS\\'+'vid'+str(i)
        if not os.path.exists(dest):
            os.makedirs(dest)
    except OSError:
        print ('Error: Creating directory of data')

    # frame
    currentframe = 0
    IMAGE_FILES = []

    while(True):
        
        # reading from frame
        ret,frame = cam.read()

        if ret:
            if (currentframe%fps) == 0 and currentframe <= 300:
                # if video is still left continue creating images
                name = 'PRE' + str(currentframe) + '.jpg'
                print ('Creating...' + name)
                IMAGE_FILES.append(name)

                # writing the extracted images
                cv2.imwrite(name, frame)
                # cv2.imwrite(os.path.join(dest, name), frame)
                # increasing counter so that it will
                # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.8) as face_detection:
      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        if not results.detections:
          continue
        annotated_image = image.copy()
        for detection in results.detections:
          mp_drawing.draw_detection(annotated_image, detection)
        cv2.imwrite(os.path.join(dest, 'res' + str(idx) + '.jpg'), annotated_image)

    for idx, file in enumerate(IMAGE_FILES):
        os.remove(file)


vFiles = 0
vPath = []
for path in os.listdir('VIDEOS'):
    if os.path.isfile(os.path.join("C:\\VS\\projects\\trilogy\\fin\\VIDEOS", path)):
        vFiles += 1
        vPath.append(os.path.join("C:\\VS\\projects\\trilogy\\fin\\VIDEOS", path))

n=0
for vid in vPath:
    # print(vid)
    # print(n)
    ext(vid,n)
    n += 1