import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    # Захватываем видео с камеры
    ret, frame = cap.read()

    # Отражаем изображение по горизонтали
    frame = cv2.flip(frame, 1)

    # Изменяем размер окна
    # frame = cv2.resize(frame, (800, 600))

    # Конвертируем кадр в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ищем лица на кадре
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Для каждого найденного лица
    for (x,y,w,h) in faces:
        # Рисуем прямоугольную рамку вокруг лица
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # Обрезаем кадр, чтобы распознать улыбку только на лице
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Ищем улыбки на лице
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        # Для каждой найденной улыбки
        for (sx,sy,sw,sh) in smiles:
            # Рисуем прямоугольную рамку вокруг улыбки
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)

    # Отображаем кадр с лицами и улыбками
    cv2.imshow('Facial Recognition', frame)

    # Для выхода нажмите ESC
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
