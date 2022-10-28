import cv2


video = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
foreground_background = cv2.createBackgroundSubtractorKNN()
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    foreground_mask = foreground_background.apply(frame)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Original", frame)
    cv2.imshow("Maks", foreground_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
