import cv2
import mediapipe as mp
import numpy as np

# Ortak ayarlar
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Kamera ile yüz mozaikleme
kamera = cv2.VideoCapture(0)
while True:
    ret, frame = kamera.read()
    if not ret:
        break
    frame_mozaik = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sonuc = face_mesh.process(rgb)
    if sonuc.multi_face_landmarks:
        for yuz in sonuc.multi_face_landmarks:
            h, w, _ = frame.shape
            x_min, x_max, y_min, y_max = w, 0, h, 0
            for nokta in yuz.landmark:
                x, y = int(nokta.x * w), int(nokta.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

            # Kenarlara %20 pay ekle (mozaik alanını büyüt)
            genislik = x_max - x_min
            yukseklik = y_max - y_min
            pay = 0.2
            x_min = max(0, int(x_min - genislik * pay))
            x_max = min(w, int(x_max + genislik * pay))
            y_min = max(0, int(y_min - yukseklik * pay))
            y_max = min(h, int(y_max + yukseklik * pay))

            if x_max > x_min and y_max > y_min:
                yuz_bolgesi = frame[y_min:y_max, x_min:x_max]
                
                # Mozaik efekti
                kucuk_boyut = (max(1, (x_max - x_min) // 20), max(1, (y_max - y_min) // 20))
                kucuk_yuz = cv2.resize(yuz_bolgesi, kucuk_boyut, interpolation=cv2.INTER_LINEAR)
                mozaik_yuz = cv2.resize(kucuk_yuz, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
                
                # Dairesel maske oluştur
                mask = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
                merkez = ((x_max - x_min) // 2, (y_max - y_min) // 2)
                yaricap = min((x_max - x_min), (y_max - y_min)) // 2
                cv2.circle(mask, merkez, yaricap, (255, 255, 255), -1)

                # Dairesel alanı karıştır
                bolge = frame_mozaik[y_min:y_max, x_min:x_max]
                mozaik_yuvarlak = np.where(mask == 255, mozaik_yuz, bolge)
                frame_mozaik[y_min:y_max, x_min:x_max] = mozaik_yuvarlak

    
    cv2.imshow('Yuz Mozaik', frame_mozaik)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
