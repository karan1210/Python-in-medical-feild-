from imageai.Detection import ObjectDetection
import os
import matplotlib.pyplot as plt
import cv2


# Configer path
execution_path = os.getcwd()

# Detect Object
# Call inbuilt project
detector = ObjectDetection()

# Call RetinaNet pre train model.
detector.setModelTypeAsRetinaNet()

# Reading model
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))

# Load the model
detector.loadModel()

# Load the Image 
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "karan.jpg"), output_image_path=os.path.join(execution_path , "Object_Detect_karan.jpg"))


# print the class of object with percentage_probability
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

Original_img = cv2.imread("karan.jpg")
Object_detect = cv2.imread("Object_Detect_karan.jpg")

Original_img = cv2.cvtColor(Original_img, cv2.COLOR_BGR2RGB)
Object_detect = cv2.cvtColor(Object_detect, cv2.COLOR_BGR2RGB)

#Plotting Images
titles = ['Original img', 'Object detected image']
images = [Original_img,Object_detect]

for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
