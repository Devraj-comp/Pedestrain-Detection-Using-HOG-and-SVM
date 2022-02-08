"""
PEDESTRAIN DETECTION USING HOG FEATURE EXTRACTION AND SVM

"""

import cv2
video_path = "pedestrian.mp4"

# importing hog objects and detector

hog_obj = cv2.HOGDescriptor()
ppl_detector = cv2.HOGDescriptor_getDefaultPeopleDetector()

hog_obj.setSVMDetector(ppl_detector)

confidence = 0.3
scale_factor = 0.9


capture = cv2.VideoCapture(video_path)

while True:
    valid_frame, image = capture.read()
    if valid_frame:
        # scaling images to fasten the image processing
        image_scale = cv2.resize(image, (0,0), None, scale_factor, scale_factor)


        image_rgb = cv2.cvtColor(image_scale, cv2.COLOR_BGR2RGB)
        
        # get hog object
        regions, weights = hog_obj.detectMultiScale(image_rgb,
                                winStride = (4,4),
                                padding = (8, 8),
                                scale = 1.01)

        # plot rectangle

        print(weights)
        
        for region, weight in zip(regions, weights):
            if weight > confidence: #if weight is greater than confidence then detect
                left, top, width, height = region
                left, top, width, height = int(left/scale_factor), int(top/scale_factor), int(width/scale_factor), int(height/scale_factor)

                cv2.rectangle(image, (left, top), (left+width, top+height), (30, 255, 30), 2)
        cv2.imshow("HOG-SVM", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

capture.release()
cv2.destroyAllWindows()