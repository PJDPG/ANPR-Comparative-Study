import cv2
import numpy as np
import os
import time

# VARIABLES
w1, h1 = 840, 200
threshold = 0.7
X_Thresh = 50
Y_Thresh = 100

# INICIALIZACION
# Se inicializan las variables de alto y ancho del glifo
h, w = 0, 0

# Se carga el path de la imagen a leer
path = 'imagenesTEST/matriculadetectada1.jpg'
# path = 'imagenesTEST/matriculadetectada2.jpg'

# Se cargan todos los glifos de la carpeta
template_list = os.listdir('Glifos2')
print("Número de glifos cargados: ", len(template_list))
Glifos = [cv2.imread(os.path.join('Glifos2/'+template_list[i])) for i in range(len(template_list))]

# Se carga además el diccionario de glifos para identificar el valor de cada detección
ArchivoGlifos = open("DiccionarioGlifos.txt", "r")
Indice_Glifos = ArchivoGlifos.read().split('\n')

# Se lee la imagen que contiene la matrícula de la que se va a extraer la información
imagen = cv2.imread(path)

# Se escala dicha imagen hasta el tamaño de los glifos, se conviete a grises y se binariza
# En el proceso se guarda una imagen_final para dibujar los resultados de las detecciones
imagen_escalada = cv2.resize(imagen, (w1, h1), interpolation=cv2.INTER_LINEAR)
imagen_final = cv2.resize(imagen, (w1, h1), interpolation=cv2.INTER_LINEAR)
imagen_gris = cv2.cvtColor(imagen_escalada, cv2.COLOR_RGB2GRAY)
ret, imagen_bin = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Se inicializa el array que contendrá los puntos detectados y la clase detectada
puntos_detectados = []

# Se realiza una búsqueda para cada glifo, denotado en "DiccionarioGlifos"
for i in range(len(Indice_Glifos)):
    # Se carga el glifo
    Glifo = Glifos[i]

    # Se convierte dicho glifo a grises y se binariza
    glifo_gris = cv2.cvtColor(Glifo, cv2.COLOR_RGB2GRAY)
    ret2, glifo_bin = cv2.threshold(glifo_gris, 150, 255, cv2.THRESH_BINARY)

    # Se realiza el reconocimiento de patrones
    deteccion = cv2.matchTemplate(imagen_bin, glifo_bin, cv2.TM_CCOEFF_NORMED)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(deteccion)

    # Se determina el tamaño de un glifo
    h, w = Glifo.shape[0], Glifo.shape[1]

    # Se buscan las detecciones de patrones que superen el umbral de confianza
    localizacion = np.where(deteccion >= threshold)

    # El siguiente paso es quitar puntos redundantes.
    # Como se conoce el tamaño de un glifo, se pueden borrar todos aquellos que se encuentren solapados
    # mediante el uso de un algoritmo Non Maximum Supression casero
    puntos_crudos = []
    for pt in zip(*localizacion[::-1]):
        puntos_crudos.append(pt)

    # Mientras que la lista de puntos crudos siga conteniendo puntos, se repite el bucle
    while len(puntos_crudos) != 0:
        # Se define un array con los puntos a borrar
        puntos_a_borrar = []
        # Se define el punto de referencia como el primer elemento de la lista, se asigna a la lista de detecciones
        # y se añade a la lista de borrar de los crudos
        punto_referencia = puntos_crudos[0]
        puntos_detectados.append((puntos_crudos[0][0], puntos_crudos[0][1], i))
        puntos_a_borrar.append(punto_referencia)

        # Se asignan a la lista de borrar los puntos que no superan el umbral de distancia
        for j in range(len(puntos_crudos)):
            if j != 0:
                if (abs(puntos_crudos[j][0] - punto_referencia[0]) < X_Thresh and
                        abs(puntos_crudos[j][1] - punto_referencia[1]) < Y_Thresh):
                    puntos_a_borrar.append(puntos_crudos[j])

        # Se borran los puntos asignados a la lista de borrar de la lista cruda
        for j in range(len(puntos_a_borrar)):
            puntos_crudos.remove(puntos_a_borrar[j])

        # Se repite el proceso hasta que la lista de crudos quede vacía

# Se organiza la lista de izquierda a derecha
puntos_detectados.sort()

# Se traducen los indices de las detecciones a los glifos correspondientes
Matricula = ''
for i in range(len(puntos_detectados)):
    Matricula += str(Indice_Glifos[puntos_detectados[i][2]])

# Se obtiene la matrícula final
print(" ")
print("Matrícula: ", Matricula)
print(" ")

for i in range(len(puntos_detectados)):
    cv2.rectangle(imagen_final, (puntos_detectados[i][0], puntos_detectados[i][1]),
                  (puntos_detectados[i][0] + w, puntos_detectados[i][1] + h), (255, 0, 0), 2)

cv2.imshow("matricula", imagen_final)
cv2.waitKey(0)
