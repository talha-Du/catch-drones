import cv2
import numpy as np
import tensorflow as tf

# Laden Sie Ihr trainiertes Modell
model = tf.keras.models.load_model('trained_model.h5')

# Öffnen Sie das Video
cap = cv2.VideoCapture('your_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Hier können Sie Frame-Verarbeitung, Größenänderung, Farbkonvertierung usw. hinzufügen

    # Vorhersagen
    input_image = cv2.resize(frame, (224, 224))  # Größe an Ihr Modell anpassen
    input_image = np.expand_dims(input_image, axis=0)
    predictions = model.predict(input_image)

    # Wenn eine Drohne erkannt wurde, können Sie hier entsprechende Aktionen ausführen
    if predictions[0][0] > 0.5:  # Beispiel: Schwellenwert für die Erkennung
        cv2.putText(frame, "Drohne erkannt", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Frame anzeigen
    cv2.imshow('Drohnen-Erkennung', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()