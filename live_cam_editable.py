import keras.src.saving
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

camera_nr = 0

# Funkcja do wczytywania modelu
def load_model():
    model = keras.src.saving.load_model("gest_recogn_data2.h5")
    return model

# Funkcja umożliwiająca wprowadzenie rozmiaru klatki obrazu
def get_frame_size():
    frame_size = 128
    return frame_size

# Funkcja do wykrywania dłoni
def preprocess_hand(image, frame_size):
    # Konwertowanie obrazu na skali szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rozmycie obrazu, aby zredukować szumy
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)

    # Ustawienie progowania, aby wykryć dłoń (możesz dostosować próg)
    _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Znajdowanie konturów
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Znalezienie największego konturu (zakładamy, że to dłoń)
        max_contour = max(contours, key=cv2.contourArea)

        # Tworzenie maski dla dłoni
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)

        # Nałożenie maski na obraz
        result = np.ones_like(image) * 255  # Ustawienie tła na białe
        result[mask > 0] = [0, 0, 0]  # Czarny kolor dla dłoni
    else:
        result = np.ones_like(image) * 255  # Jeśli brak konturów, ustaw cały obraz na biały

    # Skalowanie do rozmiaru wejściowego modelu
    result_resized = cv2.resize(result, (frame_size, frame_size))

    return result_resized

# Funkcja do predykcji z obrazu
def predict_image(image, model, mapper, frame_size):
    # Preprocess dłoni (wybielenie dłoni i czarne tło)
    processed_image = preprocess_hand(image, frame_size)

    # Normalizacja obrazu do zakresu [0, 1]
    img_array = processed_image / 255.0

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Konwersja na tensor TensorFlow
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Predykcja modelu
    prediction = model(img_tensor, training=False)
    predicted_class = tf.argmax(prediction, axis=1)[0]

    # Pobranie etykiety klasy
    predicted_label = mapper(predicted_class.numpy())

    return predicted_label, prediction[0][predicted_class].numpy(), processed_image

# Funkcja do uruchomienia kamery
def capture_from_camera(frame_size, model, mapper):
    # Otwieranie kamery
    cap = cv2.VideoCapture(camera_nr)
    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        return

    while True:
        # Pobranie klatki
        ret, frame = cap.read()
        if not ret:
            break

        # Zmieniamy format obrazu na RGB (dla PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predykcja
        predicted_label, confidence, model_input = predict_image(frame_rgb, model, mapper, frame_size)

        # Wyświetlanie oryginalnego podglądu z etykietą
        cv2.putText(frame, f"Predicted: {predicted_label} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera', frame)

        # Wyświetlanie danych wejściowych do modelu
        model_input_display = cv2.cvtColor(model_input, cv2.COLOR_RGB2BGR)
        cv2.imshow('Model Input', model_input_display)

        # Wyjście z pętli po naciśnięciu 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Zamknięcie kamery
    cap.release()
    cv2.destroyAllWindows()

# Przykład funkcji mapper (można dostosować do swoich etykiet)
def mapper(class_idx):
    labels = ['fist', 'five', 'none', 'okay', 'peace', 'rad', 'straight', 'thumbs']
    return labels[class_idx]

# Główna funkcja
if __name__ == "__main__":
    model = load_model()
    frame_size = get_frame_size()
    capture_from_camera(frame_size, model, mapper)
