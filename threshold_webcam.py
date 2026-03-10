import cv2

def on_trackbar(value):
    pass


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: não foi possível acessar a webcam.")
        return

    cv2.namedWindow("Threshold")
    cv2.createTrackbar("Limiar", "Threshold", 127, 255, on_trackbar)

    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erro: não foi possível capturar o frame da webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        limiar = cv2.getTrackbarPos("Limiar", "Threshold")

        _, binary = cv2.threshold(gray, limiar, 255, cv2.THRESH_BINARY)

        cv2.imshow("Original", frame)
        cv2.imshow("Cinza", gray)
        cv2.imshow("Threshold", binary)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()