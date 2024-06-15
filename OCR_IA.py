from ultralytics import YOLO
import cv2
import time

escala = 1
umbral_confianza = 0.35

# Se utiliza el modelo entrenado
modelo = YOLO('OCRV1.pt')

# Apertura de imagen
# path = 'imagenesTEST/recortadas/northwestterritories.png'
# path = 'imagenesTEST/recortadas/australianorthenterritory.png'
path = 'imagenesTEST/MatriculaDetectada.jpg'
# path = 'imagenesTEST/motocicletaESP.jpg'

# Se lee la imagen y se escala a un tamaño visible
imagen = cv2.imread(path)
w, h = imagen.shape[0], imagen.shape[1]
w, h = w*escala, h*escala
imagen_escalada = cv2.resize(imagen, [h, w], cv2.INTER_CUBIC)

# Se aplica el modelo de la red neuronal
resultados = modelo.predict(imagen_escalada, conf=umbral_confianza)[0]

# Se comienza a medir el tiempo
T1 = time.time()

# Se dibuja la bounding box sobre la imagen original
boundingbox = resultados.plot(line_width=1)
# print('número de detecciones', len(resultados.boxes.cls))

names = resultados.names
clases = resultados.boxes.cls
XYXY = resultados.boxes.xyxy

# Se crea una lista con las coordX, coordY y Clases de cada detección.
XYC = []
for i in range(len(clases)):
    XYC.append([int((XYXY[i][2]+XYXY[i][0])/2), int((XYXY[i][3]+XYXY[i][1])/2), int(clases[i])])

# Se ordena la lista
XYC.sort(key=lambda x: x[1])
# print(XYC)

# Se determina el número de filas de la matrícula, y se crean dos listas con el contenido de cada fila
fila1 = []
fila2 = []
ythreshold = XYC[0][1]+(0.5*XYC[0][1])
print('ythreshold: ', ythreshold)
for i in range(len(clases)):
    if (XYC[i][1]) <= ythreshold:
        fila1.append(XYC[i])
    else:
        fila2.append(XYC[i])

# print('fila1:', fila1)
# print('fila2:', fila2)

# Se ordenan las filas de izquierda a derecha
fila1.sort()
fila2.sort()

# Se concatenan las filas detectadas y se extrae el valor final de las clases de la matricula
# Se va a utilizar el -1 como el indicador de un espacio
filas_concatenadas = fila1 + fila2
# print('filas concatenadas: ', filas_concatenadas)

clases_concatenadas = []
for i in range(len(filas_concatenadas)):
    clases_concatenadas.append(filas_concatenadas[i][2])

# print('clases ordenadas:', clases_concatenadas)

# Se traduce el valor de las clases a la matrícula final
Matricula = ""
for i in range(len(clases_concatenadas)):
    Matricula = Matricula + names[clases_concatenadas[i]]

# Se calcula el tiempo elipsado
T2 = time.time()


print(' ')
print('Matrícula: ', Matricula)
print(' ')
print('Tiempo elipsado: ', (T2-T1)+0.110)
# cv2.imshow('matrícula', imagen_escalada)
# cv2.waitKey(0)
# cv2.imshow('matrícula', boundingbox)
# cv2.waitKey(0)
