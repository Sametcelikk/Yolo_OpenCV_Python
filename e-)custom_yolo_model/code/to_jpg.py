#bu kodda tüm png ve jpeg formatında olan tüm resimleri jpg formatına çevirdik


import cv2
import os
from glob import glob


png = glob("C:/Users/samet/Desktop/YOLO_OpenCV_Python/e-)custom_yolo_model/spot_data/spot_images/*.png")#glob komutu verdiğim
#pathdeki tüm png uzantılı resimleri okuyup değişkenime atıyacak
jpeg = glob("C:/Users/samet/Desktop/YOLO_OpenCV_Python/e-)custom_yolo_model/spot_data/spot_images/*.jpeg")#glob komutu verdiğim
#pathdeki tüm jpeg uzantılı resimleri okuyup değişkenime atıyacak

for j in png:
    print(j)
    img = cv2.imread(j)
    cv2.imwrite(j[:-3]+"jpg",img) #bu datırda resmin uzantısını jpg yaptıkve klasöre kaydettik
    os.remove(j)#bu satırda png uzantılı resmi slidik
    
for j in jpeg:
    print(j)
    img = cv2.imread(j)
    cv2.imwrite(j[:-4]+"jpg",img)
    os.remove(j)    