

import cv2
import numpy as np


cap = cv2.VideoCapture("C:/Users/samet/Desktop/YOLO_OpenCV_Python/yolo_pretrained_video/videos/people.mp4") 

fourcc = cv2.VideoWriter_fourcc(*"MPV+")
writer = cv2.VideoWriter("kayit.mp4",fourcc,25,(720,1280))

while True:
    ret,frame = cap.read()
    
    

    frame_width = frame.shape[1] #bu satırda resmin enini tespit ettik
    frame_height = frame.shape[0] #bu satırda resmin boyunu tespit ettik


    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True , crop=False) 
    #cv2.dnn.blobFromImage komutunu kullanmamızın sebebi resmimizi yolo algoritmasına vereceğimiz için o resmi blob formatına
    #çevirmemiz gerekiyor. 1. değişken kaynağımızı 2. değişkende 1/255 yazdık bu en verimli sonucu vermesi için yazmamız 
    #gereken değer 3. değişkende blobun kaça kaçlık olacağını giriyoruz (416,416) girmemizin sebebi kullanacağımız 
    #yolo modeli 416 ya 416 lık resimlerle eğitildiği için bu değeri girdik (yolov3 416 modelini indirdik)
    #4. değişkende resmi BGR formattan RGB formata çeviriyoruz çünkü cv2 BGR olarak tanıyor ama yolo RGB resimlerle çalışıyor
    #5 değişkende resmi kırpmak için kullanıyoruz ama biz resmi kırpmayacağımız için false değerini girdik


    labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
              "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
              "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
              "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
              "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
              "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
              "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
              "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
              "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
              "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
    #labels etiket demek burada yolo algoritmasını kullanarak neleri tanımasını istiyorsak onların listesini olusştruyoruz

    colors = ["255,0,0","0,255,0","0,0,255","100,10,255","255,0,255"]
    #burada ise tespit ettiğimiz nesnelerin etrafını saran dikdörtgenlerin renklerini belirleyip listeledik
    #fakat bu değerler hala string ve kullanabileceğimiz tarzda değiller bunları bir takım işleme tabi tutup matrisler halinde
    #kullanabileceğimiz bir format haline getireceğiz
    colors=[np.array(color.split(",")).astype("int") for color in colors]
    #bu satırda for döngüsüyle renk kodlarını listeden teker terek alıp bumları numpy.array komutunu kullanarak matris şeklinde
    #listeledik split komutuyla ana listemizdeki elemanları ayırdık ve astype komutuyla bu değerleri str formatından 
    #int formatına çevirdik.Fakat hala sayılar 3 erli listeler halinde elimizde. onlara teker teker erişmek için bir kez
    #daha numpy.array komutunu kullanıyoruz ve sayılar elimizde int biçimde teker teker ve matris halinde bulunacak
    colors=np.array(colors)
    colors=np.tile(colors,(18,1))
    #bu satırda ise numpy.tile komutunu kullanarak elimizdeki matrisi uzattık 2. değişkende kaç satır aşağı ve kaç
    #sütun sağa ekleme yapacağımızı belirliyoruz 



    model = cv2.dnn.readNetFromDarknet("C:/Users/samet/Desktop/YOLO_OpenCV_Python/pretrained_model/yolov3.cfg",
                                        "C:/Users/samet/Desktop/YOLO_OpenCV_Python/pretrained_model/yolov3.weights")
    #bu satırda cv2.dnn.readNetFromDarknet komutunu kullanarak cfg ve wights dosyamızı model değişkeni içerisinde tutuyoruz
    #1. değişken cfg 2. değişken weights dosyası için. Bu dosyalar görüntüyü tanımamız ve nesneleri tespit etmemizi sağlayan
    #kütüphanenin bulunduğu dosyalardı
    
    layers = model.getLayerNames()
    #bu satırda modelin içindeli tüm layerları layers değişkenine .getLayerNames() komutunu kullanarak atadık
    #yanlız burda modelimizdeki tüm katmanlar var ama ben hepsini değil sadece tespit yapılan katmanları istiyoruz
    #diğer bir deyişle çıktı katmanları işimize yarayacak layerları ise .getUnconnectedOutLayers komutuyla buluyoruz
    
    output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
    #burada .getUnconnectedOutLayers() komutunu kullanarak gerekli layerların indexlerinin bir fazlasını bulduk. Yukarda kullandığımız
    #yöntemle ise bu layerların asıl indexlerini taspit ettik ve liste halinde output_layer değişkenine atadık
    
    #Artık modele blob formatına çevirdiğimiz resmi verebiliriz
    
    model.setInput(frame_blob)
    #bu satırda artık modelimize blob formatındaki resmimizi verdik
    
    detection_layers = model.forward(output_layer)
    #bu satırda output layerlarının değerlerine .forward komutunu kullanarak ulaştım bu değerlere ulaşmam lazım çünkü 
    #output_layer değişkeninde sadece layer isimleri mevcut artık detection_layers değişkeninde bu layerların değerleri var



    ############### NON-MAXIMUM SUPPRESSION - OPERATION 1 ###############
    
    #ilk kısımda 3 adet boş liste oluşturacağız. Bunlar:
    ids_list = []  #bu listede predicted id leri tutucaz
    boxes_list = [] #bu liste bounding boxlar ile ilgili özellikleri tutucak
    confidences_list = [] #bu liste ise doğruluk skorlarını tutacak
    
    ######################### END OF OPERATION 1 ########################



    for detection_layer in detection_layers:     #bu for dögüleriyle detection layer larındaki layerlara ve ardından o layerlar içindeki
        for object_detection in detection_layer: #objelere ulaşıyorum bu objeler içinden de güven skorlarına ulaşıcam
            
            scores = object_detection[5:]  #bu değerler arasından ilk 5 i bounding boxlarla alakalı bu yüzden ilk beşinden sonraki değerleri
                                          #alıcam. Onlar güven skorlarıyla ilgili olan değerler. Buradaki değerlerin anlamı ise şu:
                                          #tespit ettiği nesnelerin güven skorları oluyo maksimum değer olan nesne hangisine denk 
                                          #geliyorsa tespit ettiği nesne o demek oluyo.Buraraki maks değeri tespit etmemiz lazım.
                                          
            
            predicted_id = np.argmax(scores) #burada np.argmax komutunu kullanarak en yüksek değerin indeksini buluyoruz
            confidence = scores[predicted_id] #bu en yüksek değeri confidence değerine atıyoruz
            
            if confidence > 0.30: #burada doğruluk skoru %30 dan azsa o nesneyi tanımaya çalışma demek istedik
                
                label = labels[predicted_id] #burada predicted_id değişkenini kullanarak nesnemizin ne olduğunu tespit ettik
                bounding_box = object_detection[0:4]*np.array([frame_width,frame_height,frame_width,frame_height]) #burada boindeing box ı oluşturduk 
                #ve boyutunu ayarladık
                (box_center_x, box_center_y, box_width, box_height)=bounding_box.astype("int")#burada yukardaki float değerleri int formatına çevirip
                #bounding box merkez kordinatlarını , genişlik ve yüksekliğini belirlediğimiz değişkenlere atadık 
                #Sırada dikdörgenimizin yani bounding boxımızın başlangıç ve bitiş koordinatlarını belirliyeceğiz
                #başlangıç noktası sol alt bitiş noktası sağ üst köşe
                
                start_x = int(box_center_x - (box_width/2)) #başlangıç ve bitiş noktalarını int formatında almamız gerekir
                start_y = int(box_center_y - (box_height/2)) #bu satırlarda sol üst köşenin koordinantlarını bulduk
                
                
                ############### NON-MAXIMUM SUPPRESSION - OPERATION 2 ###############
                
                #bu adımda listelerimizin içini dolduracağız
                ids_list.append(predicted_id) #burada predicted id leri listeye ekledik
                boxes_list.append([start_x,start_y,int(box_width),int(box_height)]) #burada box özelliklerini listeye ekledik
                #çizim için gerekli değerler int formatında olması gerektiği için genişlik ve yüksekliği int formatına çevirdik
                confidences_list.append(float(confidence)) #burda da doğruluk skorlarını absürt bir rakam çıkmasın diye
                #float cinsine çevirip listeye ekledik
                
                ######################### END OF OPERATION 2 ########################
            
            
            
    ############### NON-MAXIMUM SUPPRESSION - OPERATION 3 ###############          
    
    #bu adımda en yüksek güvenilirliğe sahip boxları buluca bunu cv2.dnn.NMSBoxes komutuyla yapıcaz
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4) #bu komut max doğrulama skoruna sahip boxları bir liste array biçiminde 
    #bize döndürüyo. 1. parametre box listemizi 2 parametre confidence yani doğruluk skorlarımızın listesini 3. değişkende confidence ve 4. değişkende
    #treshold değerini giriyoruz 3. ve 4. değişken optimum değerlerdir projeye göre değiştirilebilir
    
    for max_id in max_ids: #burada bu max doğruluk değerlerine sahip box değerlerine teker tekerk ulaştık for döngüsü sayesinde
        max_class_id = max_id[0] #max güvenilirliğe sahip boxları max_class_id değişkeninde tutuyoruz
        box = boxes_list[max_class_id] #max günelirliğe sahip boxları boxes_list  listesinden çektik ve box değişkenine atadık
         
        start_x = box[0] #ve bu box özelliklerini start_x start_x box_width box_height değişkenlerine kaydettim
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]
               
        
        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]
           
    ######################### END OF OPERATION 3 ########################          
                
        end_x = start_x + box_width  #bu satırlarda da bounding boxun sağ alt köşesini belirledik
        end_y = start_y + box_height  
                
        #şimdi kutularımızın rengini belirlicez
        
        box_color = colors[predicted_id] #burada predicted_id deki değer 0 sa mesela 0 colors matrisindeki 0 değerini çekip onu box_color
        #değişkenine atıyacak bu sayede her bir label yani nesne için farklı bir renk seçicez. Ama çektiğimiz bu değerler bir liste halinde
        #tutulmalı
        box_color = [int(each) for each in box_color] #burada değerleri liste halinde tek tek sakladık
        
        label = str(f"{label}: {confidence*100:.2f}%")
        print(f"predicted object {label}") #bu satırda terminale hangi labelın yüzde kaç 
        #doğruluk değeri olduğunu bulup yazdırdık
        
        
        #şimdi cv2.rectangle komutu ile dikdörtgen çizdiricez
        
        cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),box_color, 2)
        cv2.putText(frame,label,(start_x,start_y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, box_color, 2) 
        #bu satırda ise boxımızın üstüne ismini yani labelını yaydırıcaz 
                            
    writer.write(frame)
    cv2.imshow("Detection Window",frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.relase()
cv2.destroyAllWindows()


