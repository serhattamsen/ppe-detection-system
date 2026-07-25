import cv2, os, time
from ultralytics import YOLO

MODEL = "baret_model.pt"
KAYNAK = 0          # webcam. IP kamera icin RTSP adresini yaz
ESIK = 0.45
KARE_ATLA = 2       # her 2 karede bir analiz
COOLDOWN = 10       # ayni ihlal icin bekleme suresi (saniye)

def ihlal_mi(ad):
    a = ad.lower()
    return a.startswith("no-") or a in {"head", "fall-detected"}

model = YOLO(MODEL)
cap = cv2.VideoCapture(KAYNAK)
os.makedirs("alarmlar", exist_ok=True)

sayac, son_alarm, son_kare = 0, 0, None
print("Cikis icin ESC")

while True:
    ok, kare = cap.read()
    if not ok:
        print("Kamera goruntusu alinamadi")
        break

    sayac += 1
    if sayac % KARE_ATLA == 0:
        r = model.predict(kare, conf=ESIK, verbose=False)[0]
        son_kare = r.plot()
        isimler = [r.names[int(c)] for c in r.boxes.cls]
        ihlaller = sorted({i for i in isimler if ihlal_mi(i)})

        if ihlaller and time.time() - son_alarm > COOLDOWN:
            son_alarm = time.time()
            ad = time.strftime("alarmlar/%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(ad, son_kare)
            print(f"[{time.strftime('%H:%M:%S')}] ALARM: {', '.join(ihlaller)} -> {ad}")

        if ihlaller:
            cv2.rectangle(son_kare, (0, 0), (son_kare.shape[1], 50), (0, 0, 200), -1)
            cv2.putText(son_kare, "IHLAL: " + ", ".join(ihlaller),
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if son_kare is not None:
        cv2.imshow("KKD Denetim - Canli", son_kare)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()