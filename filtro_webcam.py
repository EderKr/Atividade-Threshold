import cv2
import numpy as np
import os


def overlay_png(background, overlay, x, y, w, h):
    if overlay is None:
        return background

    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    bg_h, bg_w = background.shape[:2]

    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, bg_w)
    y2 = min(y + h, bg_h)

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    roi = background[y1:y2, x1:x2]
    overlay_crop = overlay_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    if overlay_crop.shape[2] == 4:
        overlay_rgb = overlay_crop[:, :, :3]
        alpha = overlay_crop[:, :, 3] / 255.0

        for c in range(3):
            roi[:, :, c] = (
                alpha * overlay_rgb[:, :, c] +
                (1 - alpha) * roi[:, :, c]
            ).astype(np.uint8)
    else:
        roi[:] = overlay_crop[:, :, :3]

    background[y1:y2, x1:x2] = roi
    return background


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: não foi possível acessar a webcam.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("Erro: não foi possível carregar o classificador de rosto.")
        cap.release()
        return

    png_path = os.path.join("img", "rostinho_lindo.png")
    filtro_png = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)

    if filtro_png is None:
        print(f"Erro: não foi possível carregar o PNG em: {png_path}")
        cap.release()
        return

    print("Pressione 'q' para sair.")

    limiar = 200

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erro: não foi possível capturar o frame da webcam.")
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) > 0:
            _, binary = cv2.threshold(gray, limiar, 255, cv2.THRESH_BINARY)
            frame_effect = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            frame_effect = frame.copy()

        for (x, y, w, h) in faces:
            filtro_w = int(w * 0.8)
            filtro_h = int(h * 0.8)

            filtro_x = x - (filtro_w - w) // 2
            filtro_y = y - (filtro_h - h) // 2

            frame_effect = overlay_png(
                frame_effect,
                filtro_png,
                filtro_x,
                filtro_y,
                filtro_w,
                filtro_h
            )

        cv2.imshow("Filtro Webcam", frame_effect)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()