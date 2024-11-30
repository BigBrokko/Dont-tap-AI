import pyautogui
import numpy as np
import cv2
from ultralytics import YOLO


# model
model = YOLO('runs/detect/best/weights/best.pt')


def capture_screen():
    """
    Functie voor het maken van een screenshot van het scherm
    :return: frame van het scherm als numpy array
    """
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def display(frame, predictions):
        """
        Functie voor het tonen van de voorspellingen op het scherm
        :param frame: frame van het scherm
        :param predictions: voorspellingen van het model
        :return: None
        """
        # Pak frame
        for result in predictions:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = model.names[int(box.cls)]  # Pak de classnaam
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Teken een boundingbox
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # Label met classnaam

        # Display
        cv2.imshow('prediction', frame)

def main():
    while True:
        frame = capture_screen()
        results = model(frame)

        display(frame, results)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

