import cv2


def extract_ine(ine):
    copy_ine = ine.copy()
    ine_front_gray = cv2.cvtColor(ine, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(ine_front_gray, 50, 200)
    blur = cv2.GaussianBlur(ine_front_gray, (5, 5), 0)
    _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    area = 0
    xc, yc, wc, hc = 0, 0, 0, 0
    for c in sorted_contours:
        x, y, w, h = cv2.boundingRect(c)
        n_area = (x + w) * (y * h)
        if n_area > area:
            area = n_area
            xc, yc, wc, hc = x, y, w, h
    roi_template = copy_ine[yc:yc + hc, xc:xc + wc]

    contours, hierarchy = cv2.findContours(th3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    area = 0
    xc, yc, wc, hc = 0, 0, 0, 0
    for c in sorted_contours:
        x, y, w, h = cv2.boundingRect(c)
        n_area = (x + w) * (y * h)
        if n_area > area:
            area = n_area
            xc, yc, wc, hc = x, y, w, h
    th_template = copy_ine[yc:yc + hc, xc:xc + wc]

    return roi_template, th_template


ine_front = cv2.imread("../../SRC/images/fondo_blanco_2_B.jpeg")
cv2.imshow("original", ine_front)
roi, th = extract_ine(ine_front)
cv2.imshow("roi", roi)
cv2.imshow("roi th", th)
cv2.waitKey(0)
