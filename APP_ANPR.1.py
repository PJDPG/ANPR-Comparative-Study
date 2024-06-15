import customtkinter as ctk
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import json

#######################################
# VARIABLES #
#######################################
# Globales
global Matricula_entrada
global Matricula_salida
#######################################
umbral_confianza_det = 0.2
umbral_confianza_ocr = 0.25
algoritmoOCR = 1  # 1: 1 linea, 2: 2 lineas
y_thresh_valor = 0.5  # Para el algoritmo de dos lineas
#######################################
path1 = 'imagenesTEST/EntradaV0.4.mp4'
path2 = 'imagenesTEST/SalidaV0.5.mp4'
#######################################
# INICIALIZACIÓN DE VARIABLES SECUNDARIAS
#######################################
# Contadores para las matriculas
Contador_Matricula_Anterior_entrada = 0
Matricula_Anterior_entrada = ""
Contador_Matricula_Anterior_salida = 0
Matricula_Anterior_salida = ""

# Memorias de entrada / salida
matricula_reciente_entrada = ""
matricula_reciente_salida = ""
decision_entrada_pendiente = 0
decision_salida_pendiente = 0
barrera_entrada_arriba = 0
barrera_salida_arriba = 0

# Variables de los botones
var_permitir_acceso = 0
var_denegar_acceso = 0
var_permitir_salida = 0
var_denegar_salida = 0

# Memorias de tiempo
Timer_barrera_entrada = 0
Timer_barrera_salida = 0
Timer_thresh_barrera = 6  # Tiempo que tardan en cerrarse las barreras

# Una memoria para inicializar el registro
memoria_inicializar_variables = 0
#######################################


def cv2topil(imagenaconvetir):
    ##########################################################
    # Esta función convierte una imagen de formato CV2 a PIL #
    ##########################################################

    imagen_pil = cv2.cvtColor(imagenaconvetir, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(imagen_pil)
    return imagen_pil


def boton_decision(boton):
    ###############################################
    # Esta función se llama al presionar un botón #
    ###############################################
    # significado de boton:
    # 1: permitir acceso
    # 2: denegar acceso
    # 3: permitir salida
    # 4: denegar salida
    ######################

    global var_permitir_acceso
    global var_denegar_acceso
    global var_permitir_salida
    global var_denegar_salida

    if boton == 1:
        var_permitir_acceso = 1
    elif boton == 2:
        var_denegar_acceso = 1
    elif boton == 3:
        var_permitir_salida = 1
    elif boton == 4:
        var_denegar_salida = 1


def app():
    ####################
    # CÓDIGO DE LA APP #
    ####################

    global ventana
    global imagen_entrada
    global imagen_salida
    global boton_si_acceso
    global boton_no_acceso
    global boton_si_salida
    global boton_no_salida
    global label_registro
    global label_matricula_entrada
    global label_conocer_matricula_entrada
    global label_matricula_salida
    global label_conocer_matricula_salida
    global boton_estado_barrera_entrada
    global boton_estado_barrera_salida

    ventana = ctk.CTk()
    ventana.title("Aplicación ANPR | Pablo Javier de Paz                                                              "
                  "                                                                                                   "
                  "                          Universidad Europea de Madrid")
    ventana.geometry("1195x670")  # Imagen dividida en dos verticalmente: 300+70+300
    ventana.minsize(1195, 670)
    ventana.maxsize(1195, 670)
    ctk.set_appearance_mode("Light")
    ctk.set_default_color_theme("dark-blue")

    # Se lee una imagen de ejemplo
    cargando = cv2.imread("imagenes_app/loading.jpg")
    imagen_pil = cv2topil(cargando)
    imagen_entrada_tk = ctk.CTkImage(imagen_pil, size=(480, 270))

    # ****************************************************************************************************
    # WIDGETS #
    # ****************************************************************************************************
    ########################
    # Se crean los widgets #
    ########################
    # Se crean los cuadros, "frame"
    cuadro_entrada = ctk.CTkFrame(master=ventana, width=796, height=305, border_width=2)
    cuadro_salida = ctk.CTkFrame(master=ventana, width=796, height=305, border_width=2)
    cuadro_paramentrada = ctk.CTkFrame(master=cuadro_entrada, width=304, height=295, border_width=2)
    cuadro_paramsalida = ctk.CTkFrame(master=cuadro_salida, width=304, height=295, border_width=2)
    cuadro_registro = ctk.CTkScrollableFrame(master=ventana, width=300, height=570, border_width=2,
                                             label_text="Registro de entradas y salidas", label_font=("ARIAL", 20))

    # Labels de video
    imagen_entrada = ctk.CTkLabel(cuadro_entrada, text=' ', image=imagen_entrada_tk)
    imagen_salida = ctk.CTkLabel(cuadro_salida, text=' ', image=imagen_entrada_tk)

    # Labels de texto
    label_entrada = ctk.CTkLabel(cuadro_entrada, text="Cámara de acceso: Entrada",
                                 font=("ARIAL", 20), fg_color="transparent")
    label_salida = ctk.CTkLabel(cuadro_salida, text="Cámara de acceso: Salida",
                                font=("ARIAL", 20), fg_color="transparent")
    label_titulo_paramentrada = ctk.CTkLabel(cuadro_paramentrada, text="Matrícula en entrada:",
                                             font=("ARIAL", 20), fg_color="#b9c3c4", corner_radius=5)
    label_titulo_paramsalida = ctk.CTkLabel(cuadro_paramsalida, text="Matrícula en salida:",
                                            font=("ARIAL", 20), fg_color="#b9c3c4", corner_radius=5)
    label_registro = ctk.CTkLabel(cuadro_registro, text="Cargando registro...", font=("ARIAL", 17))
    label_matricula_entrada = ctk.CTkLabel(cuadro_paramentrada, text="-----------",
                                           font=("ARIAL", 30))
    label_conocer_matricula_entrada = ctk.CTkLabel(cuadro_paramentrada, text="- - - - - - - - - -")
    label_matricula_salida = ctk.CTkLabel(cuadro_paramsalida, text="-----------",
                                          font=("ARIAL", 30))
    label_conocer_matricula_salida = ctk.CTkLabel(cuadro_paramsalida, text="- - - - - - - - - -")
    label_estado_barrera_entrada = ctk.CTkLabel(cuadro_paramentrada, text="Estado de barrera entrada:",
                                                font=("ARIAL", 20), fg_color="#b9c3c4", corner_radius=5)
    label_estado_barrera_salida = ctk.CTkLabel(cuadro_paramsalida, text="Estado de barrera salida:",
                                               font=("ARIAL", 20), fg_color="#b9c3c4", corner_radius=5)

    # Botones
    boton_si_acceso = ctk.CTkButton(master=cuadro_paramentrada, width=98, height=80, fg_color="#767b82",
                                    border_width=2, hover_color="#767b82", text="Permitir \nacesso",
                                    text_color="#000000", font=("ARIAL", 17), state="disabled",
                                    command=lambda: boton_decision(1))
    boton_no_acceso = ctk.CTkButton(master=cuadro_paramentrada, width=199, height=80, fg_color="#767b82",
                                    border_width=2, hover_color="#767b82", text="Denegar acceso", text_color="#000000",
                                    font=("ARIAL", 17), state="disabled", command=lambda: boton_decision(2))
    boton_si_salida = ctk.CTkButton(master=cuadro_paramsalida, width=98, height=80, fg_color="#767b82",
                                    border_width=2, hover_color="#767b82", text="Permitir \nsalida",
                                    text_color="#000000", font=("ARIAL", 17), state="disabled",
                                    command=lambda: boton_decision(3))
    boton_no_salida = ctk.CTkButton(master=cuadro_paramsalida, width=199, height=80, fg_color="#767b82",
                                    border_width=2, hover_color="#767b82", text="Denegar salida", text_color="#000000",
                                    font=("ARIAL", 17), state="disabled", command=lambda: boton_decision(4))
    boton_estado_barrera_entrada = ctk.CTkButton(master=cuadro_paramentrada, width=299, height=60,
                                                 text="CERRADA", text_color="#000000",
                                                 font=("ARIAL", 17), fg_color="#0f76d1", hover_color="#0f76d1",
                                                 border_width=2)
    boton_estado_barrera_salida = ctk.CTkButton(master=cuadro_paramsalida, width=299, height=60,
                                                text="CERRADA", text_color="#000000",
                                                font=("ARIAL", 17), fg_color="#0f76d1", hover_color="#0f76d1",
                                                border_width=2)
    ##########################
    # Se colocan los widgets #
    ##########################
    # Cam entrada
    cuadro_entrada.place(x=20, y=5)
    label_entrada.place(x=10, y=4)
    imagen_entrada.place(x=1, y=33)
    cuadro_paramentrada.place(x=486, y=5)
    # Cuadro Param entrada
    label_titulo_paramentrada.place(x=8, y=8)
    label_matricula_entrada.place(x=10, y=40)
    label_conocer_matricula_entrada.place(x=10, y=67)
    boton_si_acceso.place(x=3, y=95)
    boton_no_acceso.place(x=102, y=95)
    label_estado_barrera_entrada.place(x=8, y=200)
    boton_estado_barrera_entrada.place(x=3, y=232)
    # Cuadro Param salida
    label_titulo_paramsalida.place(x=8, y=8)
    label_matricula_salida.place(x=10, y=40)
    label_conocer_matricula_salida.place(x=10, y=67)
    boton_si_salida.place(x=3, y=95)
    boton_no_salida.place(x=102, y=95)
    label_estado_barrera_salida.place(x=8, y=200)
    boton_estado_barrera_salida.place(x=3, y=232)
    # Cam salida
    cuadro_salida.place(x=20, y=335)
    label_salida.place(x=10, y=4)
    imagen_salida.place(x=1, y=33)
    cuadro_paramsalida.place(x=486, y=5)
    # Registro
    cuadro_registro.place(x=840, y=5)
    label_registro.pack(side="top", anchor="nw")
    # ****************************************************************************************************
    # Se llama a la función con el programa ANPR
    ventana.after(1500, anpr)

    # Se llama al loop de la función principal
    ventana.mainloop()


def anpr():
    def recortar_mat(resultadosdet_func):
        ########################################################################################
        # Esta función devuelve un recorte de la imagen original con la región de la matrícula #
        ########################################################################################
        # Se extraen de los resultados las coordenadas de la boundingbox de la detección,
        # en formato XY,XY (esquinas de la bounding box)
        XYXYDET = resultadosdet_func.boxes.xyxy
        if XYXYDET.numel():
            X1bb = int(XYXYDET[0, 0].item())
            Y1bb = int(XYXYDET[0, 1].item())
            X2bb = int(XYXYDET[0, 2].item())
            Y2bb = int(XYXYDET[0, 3].item())

            # Se crea una máscara vacía donde se cambia a negro todos los pixeles menos los de la boundingbox,
            # para ver con mas claridad la imagen
            mascara_matricula = np.zeros(resultadosdet_func.orig_img.shape[:2], np.uint8)
            mascara_matricula[Y1bb:Y2bb, X1bb:X2bb] = 255

            # Se recortan el resto de pixeles
            return resultadosdet_func.orig_img[Y1bb:Y2bb, X1bb:X2bb]

    def leer_mat(resultadosocr_func):
        ########################################################################
        # Esta funcion devuelve una cadena de texto con la matricula detectada #
        ########################################################################
        # Versión del algoritmo: una linea #
        ####################################

        # Se determina el diccionario de clases, los indices del diccionario y las coordenadas
        # de las bounding boxes
        names = resultadosocr_func.names
        clases = resultadosocr_func.boxes.cls
        XYXYOCR = resultadosocr_func.boxes.xyxy

        # Se crea una lista con las coordX, coordY y Clases de cada detección.
        XYC = []
        for i in range(len(clases)):
            XYC.append([int((XYXYOCR[i][2] + XYXYOCR[i][0]) / 2), int((XYXYOCR[i][3] + XYXYOCR[i][1]) / 2),
                        int(clases[i])])

        if algoritmoOCR == 1:
            # Se ordena la lista, de izquierda a derecha
            XYC.sort()

            clases_concatenadas = []
            for i in range(len(XYC)):
                clases_concatenadas.append(XYC[i][2])

        else:
            # Se ordena la lista
            XYC.sort(key=lambda x: x[1])

            # Se determina el número de filas de la matrícula, y se crean dos listas con el contenido de cada fila
            fila1, fila2 = [], []
            ythreshold = XYC[0][1] + (y_thresh_valor * XYC[0][1])
            print('ythreshold: ', ythreshold)
            for i in range(len(clases)):
                if (XYC[i][1]) <= ythreshold:
                    fila1.append(XYC[i])
                else:
                    fila2.append(XYC[i])

            # Se ordenan las filas de izquierda a derecha
            fila1.sort()
            fila2.sort()

            # Se concatenan las filas detectadas y se extrae el valor final de las clases de la matricula
            filas_concatenadas = fila1 + fila2
            # print('filas concatenadas: ', filas_concatenadas)

            clases_concatenadas = []
            for i in range(len(filas_concatenadas)):
                clases_concatenadas.append(filas_concatenadas[i][2])

        # Se traduce el valor de las clases a la matrícula final
        Matricula_func = ""
        for i in range(len(clases_concatenadas)):
            Matricula_func = Matricula_func + names[clases_concatenadas[i]]

        if 5 < len(Matricula_func) < 10:
            return Matricula_func
        else:
            return 0

    def mostrarvideos(fotograma_func, video):
        ####################################################################################
        # Esta función coloca la imagen recibida en el cuadro de video de salida o entrada #
        ####################################################################################
        # Video de Entrada: 1
        # Video de Salida: 2
        pil_fotograma_func = cv2topil(fotograma_func)
        ctk_fotograma_func = ctk.CTkImage(pil_fotograma_func, size=(480, 270))

        if video == 1:
            imagen_entrada.configure(image=ctk_fotograma_func)
            imagen_entrada.image = ctk_fotograma_func
        elif video == 2:
            imagen_salida.configure(image=ctk_fotograma_func)
            imagen_salida.image = ctk_fotograma_func

    def leerregistroaccesos():
        #############################################################################################
        # Devuelve una lista con el contenido del txt llamado registro, lee json y convierte a list #
        #############################################################################################
        registro_func = open("RegistroAccesos.txt", "r")
        registrojson_func = registro_func.read()
        if registrojson_func != "":
            registrolist_func = json.loads(registrojson_func)
            registro_func.close()
            return registrolist_func
        else:
            registrolist_vacio = []
            return registrolist_vacio

    def escribirregistroaccesos(registro):
        ########################################################################
        # Esta función sobreescribe en el registro accesos la lista "registro" #
        ########################################################################
        global matricula_reciente_entrada
        registro_func = open("RegistroAccesos.txt", "w")
        registronuevo_json = json.dumps(registro)
        registro_func.write(registronuevo_json)
        registro_func.close()
        matricula_reciente_entrada = Matricula_entrada

    def registrarentrada(matricula_func):
        #################################################################
        # Esta función registra una entrada en formato MAT/HORA_ENTRADA #
        #################################################################
        registro = leerregistroaccesos()
        ahora = datetime.now()
        ahora_str = ahora.strftime("%d/%m/%Y %H:%M:%S")
        estructura_mat = [matricula_func, ahora_str]
        registro.append(estructura_mat)
        escribirregistroaccesos(registro)
        toggle_barreras(1, -1)

    def registrarsalida(matricula_func):
        ###############################################################
        # Esta función registra una salida en formato MAT/HORA_SALIDA #
        ###############################################################
        registro = leerregistroaccesos()
        ahora = datetime.now()
        ahora_str = ahora.strftime("%d/%m/%Y %H:%M:%S")
        for i in range(len(registro)):
            if (matricula_func in registro[i]) and len(registro[i]) == 2:
                registro[i].append(ahora_str)
                break
        escribirregistroaccesos(registro)
        toggle_barreras(-1, 1)

    def toggle_preguntas(entrada, salida):
        ############################################################
        # Esta función enciende o apaga los botones en la interfaz #
        ############################################################
        # Encender pregunta Entrada: 1
        # Apagar pregunta Entrada: 0
        # Encender pregunta Salida: 1
        # Apagar pregunta Salida: 0
        ####################################
        if entrada == 1:
            boton_si_acceso.configure(fg_color="#5db82c", hover_color="#3e7a1d", state="normal")
            boton_no_acceso.configure(fg_color="#d12f1d", hover_color="#962215", state="normal")
        elif entrada == 0:
            boton_si_acceso.configure(fg_color="#767b82", hover_color="#767b82", state="disabled")
            boton_no_acceso.configure(fg_color="#767b82", hover_color="#767b82", state="disabled")
        elif salida == 1:
            boton_si_salida.configure(fg_color="#5db82c", hover_color="#3e7a1d", state="normal")
            boton_no_salida.configure(fg_color="#d12f1d", hover_color="#962215", state="normal")
        elif salida == 0:
            boton_si_salida.configure(fg_color="#767b82", hover_color="#767b82", state="disabled")
            boton_no_salida.configure(fg_color="#767b82", hover_color="#767b82", state="disabled")

    def toggle_barreras(entrada, salida):
        ##################################################################################################
        # Esta función abre o cierra las barreras, y borra la matrícula de la interfaz cuando las cierra #
        ##################################################################################################
        global Timer_barrera_entrada
        global Timer_barrera_salida
        global Timer_thresh_barrera
        global barrera_entrada_arriba
        global barrera_salida_arriba

        if entrada == 1:
            boton_estado_barrera_entrada.configure(text="ABIERTA", fg_color="#ebe834", hover_color="#ebe834")
            Timer_barrera_entrada = time.time()
            barrera_entrada_arriba = 1

        # Cerrar barrera entrada si el timer expira, además, quitar la matricula de la interfaz
        elif entrada == 0 and (time.time() - Timer_barrera_entrada) > Timer_thresh_barrera:
            boton_estado_barrera_entrada.configure(text="CERRADA", fg_color="#0f76d1", hover_color="#0f76d1")
            Timer_barrera_entrada = 0
            label_matricula_entrada.configure(text="-----------")
            label_conocer_matricula_entrada.configure(text="- - - - - - - - - -")
            barrera_entrada_arriba = 0

        elif salida == 1:
            boton_estado_barrera_salida.configure(text="ABIERTA", fg_color="#ebe834", hover_color="#ebe834")
            Timer_barrera_salida = time.time()
            barrera_salida_arriba = 1

        # Cerrar barrera salida si el timer expira, además, quitar la matricula de la interfaz
        elif salida == 0 and (time.time() - Timer_barrera_salida) > Timer_thresh_barrera:
            boton_estado_barrera_salida.configure(text="CERRADA", fg_color="#0f76d1", hover_color="#0f76d1")
            Timer_barrera_salida = 0
            label_matricula_salida.configure(text="-----------")
            label_conocer_matricula_salida.configure(text="- - - - - - - - - -")
            barrera_salida_arriba = 0

    def mostrarregistroaccesos():
        ##############################################################
        # Esta función muestra el registro de accesos en la interfaz #
        ##############################################################

        registro_func = leerregistroaccesos()
        registro_func.reverse()
        registro_a_mostrar = ""
        for i in range(len(registro_func)):
            registro_a_mostrar += "Matrícula: " + registro_func[i][0] + "\n"
            registro_a_mostrar += "Entrada: " + registro_func[i][1] + "\n"
            if len(registro_func[i]) == 3:
                registro_a_mostrar += "Salida: " + registro_func[i][2] + "\n"
            registro_a_mostrar += "\n"

        label_registro.configure(text=registro_a_mostrar)

    def anpr_entrada(fotograma_entrada):
        #############################################
        # FUNCIÓN ANPR PARA LA CÁMARA DE LA ENTRADA #
        #############################################

        global Matricula_entrada
        global Matricula_Anterior_entrada
        global Contador_Matricula_Anterior_entrada
        global matricula_reciente_entrada
        global decision_entrada_pendiente
        global var_permitir_acceso
        global var_denegar_acceso
        global Timer_barrera_entrada
        global barrera_entrada_arriba

        if decision_entrada_pendiente == 0 and barrera_entrada_arriba == 0:
            # Se aplica el modelo de la red neuronal
            resultadosDET = modeloDET.predict(fotograma_entrada, conf=umbral_confianza_det, max_det=1)[0]

            # Se comprueba si existe una detección de matricula
            if resultadosDET.boxes.xyxy.numel():
                # Se llama a la funcion recortar_mat() para recortar el resto de pixeles
                matricula_recortada = recortar_mat(resultadosDET)

                # Se aplica el modelo de la red neuronal
                resultadosOCR = modeloOCR.predict(matricula_recortada, conf=umbral_confianza_ocr, max_det=9)[0]

                # Se comprueba si existe al menos una detección de caracter
                if resultadosOCR.boxes.xyxy.numel():
                    # Se llama a la función que lee la matrícula
                    Matricula_entrada = leer_mat(resultadosOCR)

                    archivo_permitidas = open("Permitidas.txt", "r")
                    permitidas_list = archivo_permitidas.read().split('\n')
                    archivo_permitidas.close()

                    if Matricula_entrada != matricula_reciente_entrada:
                        if Matricula_entrada in permitidas_list:
                            print(' ')
                            print('Matrícula Permitida: ', Matricula_entrada)
                            print(' ')
                            # Se escribe en el registro la entrada permitida
                            registrarentrada(Matricula_entrada)
                            mostrarregistroaccesos()
                            matricula_reciente_entrada = Matricula_entrada
                            label_matricula_entrada.configure(text=Matricula_entrada)
                            label_conocer_matricula_entrada.configure(text="Matrícula registrada.  Permitiendo "
                                                                           "acceso...")

                        else:
                            # Si la matricula es la misma, se aumenta el contador
                            if (Matricula_Anterior_entrada == Matricula_entrada and Contador_Matricula_Anterior_entrada < 3
                                    and Matricula_entrada != 0 and Matricula_entrada != matricula_reciente_entrada):
                                Contador_Matricula_Anterior_entrada = Contador_Matricula_Anterior_entrada + 1

                            # Si la matrícula no es la misma que la anterior
                            elif Matricula_Anterior_entrada != Matricula_entrada:
                                Contador_Matricula_Anterior_entrada = 0

                            # Si la matrícula es la misma que la anterior y el contador es correcto
                            # se confirma que la matricula leida es desconocida
                            elif (Matricula_Anterior_entrada == Matricula_entrada and Contador_Matricula_Anterior_entrada >= 3
                                    and Matricula_entrada != matricula_reciente_entrada):
                                print(' ')
                                print('Matrícula Desconocida: ', Matricula_entrada)
                                print(' ')
                                label_matricula_entrada.configure(text=Matricula_entrada)
                                label_conocer_matricula_entrada.configure(text="Matrícula desconocida.")
                                # Se activa la función que permite elegir si dejar pasar
                                decision_entrada_pendiente = 1
                                # 1 significa encender los botones de pregunta entrada
                                toggle_preguntas(1, -1)

                        # Matricula anterior
                        Matricula_Anterior_entrada = Matricula_entrada

                    # Se dibuja la bounding box sobre las imagenes
                    boundingboxDET = resultadosDET.plot(line_width=1)

                    # Se convierte la imagen CV2 a PIL
                    mostrarvideos(boundingboxDET, 1)

            else:
                # Si no hay una detección en pantalla
                # se convierte el fotograma a formato PIL y se carga la imagen en la label de video
                mostrarvideos(fotograma_entrada, 1)

        elif decision_entrada_pendiente == 1:
            # Se muestra la imagen
            mostrarvideos(fotograma_entrada, 1)
            print("decidiendo entrada...")
            # Si se hace click en permitir acceso
            if var_permitir_acceso == 1:
                registrarentrada(Matricula_entrada)
                matricula_reciente_entrada = Matricula_entrada
                mostrarregistroaccesos()
                toggle_preguntas(0, -1)
                var_permitir_acceso = 0
                decision_entrada_pendiente = 0

            # Si se hace click en denegar acceso
            elif var_denegar_acceso == 1:
                toggle_preguntas(0, -1)
                var_denegar_acceso = 0
                decision_entrada_pendiente = 0
                label_matricula_entrada.configure(text="-----------")
                label_conocer_matricula_entrada.configure(text="- - - - - - - - - -")

        elif barrera_entrada_arriba == 1:
            # Se muestra la imagen
            mostrarvideos(fotograma_entrada, 1)
            print("permitiendo entrada...")

    def anpr_salida(fotograma_salida):
        ############################################
        # FUNCIÓN ANPR PARA LA CÁMARA DE LA SALIDA #
        ############################################

        global Matricula_salida
        global Matricula_Anterior_salida
        global Contador_Matricula_Anterior_salida
        global matricula_reciente_salida
        global decision_salida_pendiente
        global var_permitir_salida
        global var_denegar_salida
        global barrera_salida_arriba

        if decision_salida_pendiente == 0 and barrera_salida_arriba == 0:
            # Se aplica el modelo de la red neuronal
            resultadosDET = modeloDET.predict(fotograma_salida, conf=umbral_confianza_det, max_det=1)[0]

            # Se comprueba si existe una detección de matricula
            if resultadosDET.boxes.xyxy.numel():
                # Se llama a la funcion recortar_mat() para recortar el resto de pixeles
                matricula_recortada = recortar_mat(resultadosDET)

                # Se aplica el modelo de la red neuronal
                resultadosOCR = modeloOCR.predict(matricula_recortada, conf=umbral_confianza_ocr, max_det=9)[0]

                # Se comprueba si existe al menos una detección de caracter
                if resultadosOCR.boxes.xyxy.numel():
                    # Se llama a la función que lee la matrícula
                    Matricula_salida = leer_mat(resultadosOCR)

                    archivo_permitidas = open("Permitidas.txt", "r")
                    permitidas_list = archivo_permitidas.read().split('\n')
                    archivo_permitidas.close()

                    if Matricula_salida != matricula_reciente_salida:
                        if Matricula_salida in permitidas_list:
                            print(' ')
                            print('Matrícula Permitida: ', Matricula_salida)
                            print(' ')
                            # Se escribe en el registro la entrada permitida
                            registrarsalida(Matricula_salida)
                            mostrarregistroaccesos()
                            matricula_reciente_salida = Matricula_salida
                            label_matricula_salida.configure(text=Matricula_salida)
                            label_conocer_matricula_salida.configure(text="Matrícula registrada.  Permitiendo "
                                                                          "salida...")

                        else:
                            # Si la matricula es la misma, se aumenta el contador
                            if (Matricula_Anterior_salida == Matricula_salida and Contador_Matricula_Anterior_salida < 3
                                    and Matricula_salida != 0 and Matricula_salida != matricula_reciente_salida):
                                Contador_Matricula_Anterior_salida = Contador_Matricula_Anterior_salida + 1

                            # Si la matrícula no es la misma que la anterior
                            elif Matricula_Anterior_salida != Matricula_salida:
                                Contador_Matricula_Anterior_salida = 0

                            # Si la matrícula es la misma que la anterior y el contador es correcto
                            # se confirma que la matricula leida es desconocida
                            elif (Matricula_Anterior_salida == Matricula_salida and Contador_Matricula_Anterior_salida >= 3
                                    and Matricula_salida != matricula_reciente_salida):
                                print(' ')
                                print('Matrícula Desconocida: ', Matricula_salida)
                                print(' ')
                                label_matricula_salida.configure(text=Matricula_salida)
                                label_conocer_matricula_salida.configure(text="Matrícula desconocida.")
                                # Se activa la función que permite elegir si dejar pasar
                                decision_salida_pendiente = 1
                                # 1 significa encender los botones de pregunta entrada
                                toggle_preguntas(-1, 1)

                        # Matricula anterior
                        Matricula_Anterior_salida = Matricula_salida

                    # Se dibuja la bounding box sobre las imagenes
                    boundingboxDET = resultadosDET.plot(line_width=1)

                    # Se convierte la imagen CV2 a PIL
                    mostrarvideos(boundingboxDET, 2)
            else:
                # Si no hay una detección en pantalla
                # se convierte el fotograma a formato PIL y se carga la imagen en la label de video
                mostrarvideos(fotograma_salida, 2)

        elif decision_salida_pendiente == 1:
            # Se muestra la imagen
            mostrarvideos(fotograma_salida, 2)
            print("decidiendo salida...")
            # Si se hace click en permitir acceso
            if var_permitir_salida == 1:
                registrarsalida(Matricula_salida)
                matricula_reciente_salida = Matricula_salida
                mostrarregistroaccesos()
                toggle_preguntas(-1, 0)
                var_permitir_salida = 0
                decision_salida_pendiente = 0

            # Si se hace click en denegar acceso
            elif var_denegar_salida == 1:
                toggle_preguntas(-1, 0)
                var_denegar_salida = 0
                decision_salida_pendiente = 0
                label_matricula_salida.configure(text="-----------")
                label_conocer_matricula_salida.configure(text="- - - - - - - - - -")

        elif barrera_salida_arriba == 1:
            # Se muestra la imagen
            mostrarvideos(fotograma_salida, 2)
            print("permitiendo salida...")

    ######################
    # MAIN DE AMBOS ANPR #
    ######################
    global memoria_inicializar_variables
    global Timer_barrera_entrada
    global Timer_barrera_salida

    # Se llama a las funciones que muestran por primera vez las variables
    if memoria_inicializar_variables == 0:
        mostrarregistroaccesos()
        memoria_inicializar_variables = 1

    # Se llama a las funciones de cerrar barreras
    if Timer_barrera_entrada != 0:
        toggle_barreras(0, -1)

    if Timer_barrera_salida != 0:
        toggle_barreras(-1, 0)

    # Se obtiene el fotograma actual del video
    ret, fotograma1 = video_entrada.read()
    if ret:
        anpr_entrada(fotograma1)

    ret, fotograma2 = video_salida.read()
    if ret:
        anpr_salida(fotograma2)
    ventana.after(33, anpr)


#####################################
# INICIALIZACIÓN DE MODELOS/ VÍDEOS #
#####################################

# Se cargan los modelos a emplear
modeloDET = YOLO('modelos/DetMatriculasV1.pt')
modeloOCR = YOLO('modelos/OCRV1.pt')

# Apertura de video
video_entrada = cv2.VideoCapture(path1)
video_salida = cv2.VideoCapture(path2)

# Lanzamiento del programa
app()
