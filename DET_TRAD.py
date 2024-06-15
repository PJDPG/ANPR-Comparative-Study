import cv2
import imutils
import numpy as np
import time
import math

################################################################################################
# IMPORTANTE: Esta versión del algoritmo requiere definir MUCHAS condiciones iniciales         #
# especificas para cada imagen, como los puntos de la transf. afín, la máscara a emplear, etc. #
# Estos parámetros se muestran a contiuación, para poder ajustarlos rápidamente.               #
################################################################################################

# Elegir imagen inicial
path = 'imagenesTEST/Coche5.jpg'
#path = 'imagenesTEST/Camion.jpg'
#path = 'imagenesTEST/Furgoneta.jpg'

# Elegir máscara
path2 = 'mascaras/MascaraCircular1.jpg'
#path2 = 'mascaras/MascaraCircular2.jpg'
#path2 = 'mascaras/MascaraCuadrada1.jpg'
#path2 = 'mascaras/MascaraCuadrada2.jpg'

# Asignar puntos afines (específicos para cada instalación física)
P1a, P2a, P3a = [152, 134], [253, 139], [150, 158]
P1b, P2b, P3b = [0, 0], [300, 0], [0, 150]

# Parámetros de Canny
ParamCanny1 = 0
ParamCanny2 = 200

# Iteraciones de dilatación/erosión y potencia de dilatacion/erosión
potDilate = 2  # (Mínimo 2)
itDilate = 2   # Default 1-2
potErode = 2   # (Mínimo 2)
itErode = 1    # Default 1

# Parámetro de epsilon en el aproxPolyDp
paramEpsilon = 0.04

# Parámetros de tamaño de la matrícula. Cambian según el tamaño de la imagen fuente y cambian
# según el país de procedencia (ej. las matrículas en EEUU tienen dimensiones distintas que en Europa)
UmbrMaxVer = 10  # Umbral máximo vertical de diferencia entre longitudes de rectas iguales 2 a 2
UmbrMaxHor = 10  # Umbral máximo horizontal de diferencia entre longitudes de rectas iguales 2 a 2
DistMinVer = 20  # Longitud mínima en píxeles de los laterales de la matrícula
DistMinHor = 40  # Longitud máxima en píxeles de la parte superior e inferior de la matrícula


##########################
# Comienzo del algoritmo #
##########################

# Se lee el tiempo del comienzo
T1 = time.time()

# Se lee la imagen y sus características
imagen = cv2.imread(path)
filas, columnas,canal = imagen.shape

# Se definen los puntos para la transformación afin
pts1 = np.float32([P1a, P2a, P3a])
pts2 = np.float32([P1b, P2b, P3b])

# Se mueve el origen de la transformación hasta la esquina superior derecha.
# El motivo: La imagen resultante comienza desde ese primer punto, es para poder representar la imagen completa.
pts1 = pts1 - np.float32([[152, 134],[152, 134],[152, 134]])

# Se define la matriz de transformación afin
Matriz = cv2.getAffineTransform(pts1, pts2)

# Se aplica la transformacion afin
imagen_afin = cv2.warpAffine(imagen, Matriz,(1500, 1500))
imagen_afin = cv2.resize(imagen_afin, (columnas, filas))

# Se elige y se lee la mejor máscara
mascara_preprocesado = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

# Se ajusta el tamaño de la mascara a la imagen
Xmask = int(imagen_afin.shape[0])
Ymask = int(imagen_afin.shape[1])
mascara_preprocesado = cv2.resize(mascara_preprocesado, (Ymask, Xmask), interpolation=cv2.INTER_LINEAR)
ret, mascara_preprocesado = cv2.threshold(mascara_preprocesado, thresh=180, maxval=255, type=cv2.THRESH_BINARY)

# Se aplica la máscara sobre la imagen
imagen_mascara = cv2.bitwise_and(imagen_afin, imagen_afin, mask=mascara_preprocesado)

# Imagen resultante a grises
grises = cv2.cvtColor(imagen_mascara, cv2.COLOR_RGB2GRAY)

# Filtro de reduccion de ruido
grises_filtrada = cv2.bilateralFilter(grises, 11, 17, 17)

# Se realiza un leve blur para resaltar bordes
blurred = cv2.GaussianBlur(grises_filtrada, (5, 5), 0)

# Se emplea el buscador de bordes CANNY
bordes = cv2.Canny(blurred, ParamCanny1, ParamCanny2)

# Se realiza un dilate para reforzar las líneas de Canny y homogeneizar bordes
kernel = np.ones((potDilate, potDilate), np.uint8)
dilatada = cv2.dilate(bordes, kernel, iterations=itDilate)

# Se realiza un erode para reforzar las líneas de Canny y homogeneizar bordes
kernel = np.ones((potErode, potDilate), np.uint8)
erosionada = cv2.erode(dilatada, kernel, iterations=itErode)

# Se buscan los contornos cerrados del rectangulo de la matricula en la imagen
keypoints = cv2.findContours(erosionada.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos = imutils.grab_contours(keypoints)
contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]

ubicacion = None
for contour in contornos:
    # Se simplifican los contornos cerrados detectados mediante la función approxPolyDP
    epsilon = paramEpsilon * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Se comprueba si alguno de los contornos detectados tiene 4 vértices
    if len(approx) == 4:
        P1, P2, P3, P4 = approx[0][0], approx[1][0], approx[2][0], approx[3][0]
        #print(P1, P2, P3, P4)

        # Se calculan las longitudes de cada recta
        dist1, dist2, dist3, dist4 = math.dist(P1, P2), math.dist(P2, P3), math.dist(P3, P4), math.dist(P4, P1)

        # Se define una variable rectasiguales que da una puntuacion de similitud entre rectas 2 a 2
        rectasiguales1 = abs(dist1 - dist3)
        rectasiguales2 = abs(dist2 - dist4)
        #print(rectasiguales1, rectasiguales2)

        # Se comprueba si las rectas paralelas son iguales, y si lo son se comprueba si son de unas dimensiones
        # minimas y arbitrarias, es probable que sea necesario cambiarlas también para cada sistema.
        if rectasiguales1 < UmbrMaxVer and rectasiguales2 < UmbrMaxHor and dist1 > DistMinVer and dist2 > DistMinHor:
            ubicacion = approx
            break

# Se visualizan los pasos mediante los siguientes comandos
mascara_matricula = np.zeros(imagen_afin.shape[:2], np.uint8)

# En caso de que no se realice una detección, se evitan las siguientes líneas de código (que hacen que el programa
# deje de funcionar)
if ubicacion is not None:
    imagen_mascara_final = cv2.drawContours(mascara_matricula, [ubicacion], 0, 255, -1)
    imagen_mascara_final = cv2.bitwise_and(imagen_afin, imagen_afin, mask=mascara_matricula)

    # Se recorta la matricula desde la imagen original
    (x, y) = np.where(mascara_matricula == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    imagen_final = imagen_afin[x1:x2+1, y1:y2+1]

#Se calcula el tiempo total elipsado
T2 = time.time()
print(T2-T1)

cv2.imshow('Deteccion', imagen)
cv2.waitKey(0)
cv2.imshow('Deteccion', imagen_afin)
cv2.waitKey(0)
cv2.imshow('Deteccion', mascara_preprocesado)
cv2.waitKey(0)
cv2.imshow('Deteccion', imagen_mascara)
cv2.waitKey(0)
cv2.imshow('Deteccion', grises)
cv2.waitKey(0)
cv2.imshow('Deteccion', bordes)
cv2.waitKey(0)
cv2.imshow('Deteccion', dilatada)
cv2.waitKey(0)
cv2.imshow('Deteccion', erosionada)
cv2.waitKey(0)
cv2.imshow('Deteccion', mascara_matricula)
cv2.waitKey(0)

if ubicacion is not None:
    cv2.imshow('Deteccion', imagen_mascara_final)
    cv2.waitKey(0)
    cv2.imshow('Deteccion', imagen_final)
    cv2.waitKey(0)

cv2.destroyAllWindows()