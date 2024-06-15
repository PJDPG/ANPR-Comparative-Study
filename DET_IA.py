from ultralytics import YOLO
import cv2
import numpy as np

#Se utiliza el modelo entrenado
modelo = YOLO('DetMatriculasV1.pt')

#Apertura de imagen
#path = 'imagenesTEST/Coche.jpg'
path = 'imagenesTEST/Camion.jpg'
#path = 'imagenesTEST/Furgoneta.jpg'
#path = 'imagenesTEST/MatriculaCerca.jpg'

imagen = cv2.imread(path)

#Se aplica el modelo de la red neuronal
resultados = modelo.predict(imagen, conf=0.2, max_det=1)[0]

#Se dibuja la bounding box sobre la imagen original, para ver lo que se va a recortar
boundingbox = resultados.plot(line_width=1)

#Se extraen de los resultados las coordenadas de la boundingbox de la detección,
#en formato XY,XY (esquinas de la bounding box)
XYXY = resultados.boxes.xyxy
X1bb = int(XYXY[0,0].item())
Y1bb = int(XYXY[0,1].item())
X2bb = int(XYXY[0,2].item())
Y2bb = int(XYXY[0,3].item())

#Se crea una máscara vacía donde se cambia a negro todos los pixeles menos los de la boundingbox,
#para ver con mas claridad la imagen
mascara_matricula = np.zeros(resultados.orig_img.shape[:2], np.uint8)
mascara_matricula[Y1bb:Y2bb, X1bb:X2bb] = 255
masked_imagen = cv2.bitwise_and(resultados.orig_img, resultados.orig_img, mask = mascara_matricula)

#Se recortan el resto de pixeles
imagen_final=resultados.orig_img[Y1bb:Y2bb, X1bb:X2bb]

#Se muestran los resultados
cv2.imshow('imshow', boundingbox)
cv2.waitKey(0)
cv2.imshow('imshow', masked_imagen)
cv2.waitKey(0)
cv2.imshow('imshow', imagen_final)
cv2.waitKey(0)