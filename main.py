import cv2
from fer import FER

def detect_emotion(frame):
    detector = FER(mtcnn=True)
    result = detector.detect_emotions(frame)
    if result:
        emotion, score = max(result[0]['emotions'].items(), key=lambda item: item[1])
        return emotion, score
    return None, None

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion, score = detect_emotion(frame)
        if emotion:
            text = f"{emotion}: {score:.2f}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
