#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function, division
import rospy
import numpy as np
import numpy
import tf
import math
import cv2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from numpy import linalg
from tf import transformations
from tf import TransformerROS
import tf2_ros
from geometry_msgs.msg import Twist, Vector3, Pose, Vector3Stamped
import cv2.aruco as aruco

from nav_msgs.msg import Odometry
from std_msgs.msg import Header

print("EXECUTE ANTES da 1.a vez: ")
print("wget https://github.com/Insper/robot21.1/raw/main/projeto/ros_projeto/scripts/MobileNetSSD_deploy.caffemodel")
print("PARA TER OS PESOS DA REDE NEURAL")

import visao_module

bridge = CvBridge()

cv_image = None
media = []
centro = []
atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos
centro_pista = [0,0]
centro_ciano = [0,0]
centro_verde = [0,0]
centro_laranja = [0,0]
posicao_geral = [0,0]
posicao_aruco_100 = [0,0]
posicao_aruco_200 = [0,0]
distancia = 0
area = 0.0 # Variavel com a area do maior contorno
area_aruco_50 = 0.0
area_aruco_100 = 0.0
area_aruco_150 = 0.0
area_aruco_200 = 0.0
maior_area_laranja = 0.0
maior_area_ciano = 0.0
maior_area_verde = 0.0

aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters  = aruco.DetectorParameters_create()
parameters.minDistanceToBorder = 0
parameters.adaptiveThreshWinSizeMax = 1000

# Só usar se os relógios ROS da Raspberry e do Linux desktop estiverem sincronizados. 
# Descarta imagens que chegam atrasadas demais
check_delay = False 

resultados = [] # Criacao de uma variavel global para guardar os resultados vistos

x = 0
y = 0
z = 0 
id = 0

frame = "camera_link"
#frame = "head_camera"  # DESCOMENTE para usar com webcam USB via roslaunch tag_tracking usbcam

angle_z = None

tfl = 0

tf_buffer = tf2_ros.Buffer()

viuCreeper = False
creeperLaranja = False
creeperCiano = False
creeperVerde = False

lista = ('blue', 11, 'cat')

cor = lista[0]
id_aruco = lista[1]
estacao = lista[2]

def scaneou(dado):
    global distancia
    ranges = np.array(dado.ranges).round(decimals=2)
    distancia = ranges[0]

def recebe_odometria(data):
    global x
    global y
    global posicao_geral

    x = data.pose.pose.position.x
    y = data.pose.pose.position.y

    #quat = data.pose.pose.orientation
    posicao_geral = [x, y]

def filtra_amarelo(img_in):
    global centro_pista
    img = img_in.copy()
    
    #Filtra amarelo
    hvs1 = np.array([22, 50, 50],dtype=np.uint8)
    hsv2 = np.array([36, 255, 255],dtype=np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hvs1, hsv2)

    #Calcula centro de massa
    try:
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centro_pista = []
        centro_pista.append(int(cX))
        centro_pista.append(int(cY))
        viuCreeper = False
    except:
        pass

def filtra_verde(img_in):
    global creeperVerde
    global centro_verde
    global viuCreeper
    global maior_area_verde
    img = img_in.copy()
    
    #Filtra verde
    hvs1 = np.array([45, 50, 50],dtype=np.uint8)
    hsv2 = np.array([80, 255, 255],dtype=np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hvs1, hsv2)
    contornos, arvore = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    maior_area_verde = 0.0
    for c in contornos:
        area = cv2.contourArea(c)
        if area > maior_area_verde:
            maior_area_verde = area
    #Calcula centro de massa
    try:
        if maior_area_verde > 1000.0:    
            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centro_verde = []
            centro_verde.append(int(cX))
            centro_verde.append(int(cY))
            creeperVerde = True
            viuCreeper = True
    except:
        creeperVerde = False
        viuCreeper = False

def filtra_laranja(img_in):
    global creeperLaranja
    global centro_laranja
    global viuCreeper
    global maior_area_laranja
    img = img_in.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #Filtra laranja
    hvs1 = np.array([0, 50, 150],dtype=np.uint8)
    hsv2 = np.array([6, 255, 255],dtype=np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hvs1, hsv2)

    contornos, arvore = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    maior_area_laranja = 0.0
    for c in contornos:
        area = cv2.contourArea(c)
        if area > maior_area_laranja:
            maior_area_laranja = area

    #Calcula centro de massa and 
    try:
        if maior_area_laranja > 1900.0:
            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centro_laranja = []
            centro_laranja.append(int(cX))
            centro_laranja.append(int(cY))
            creeperLaranja = True
            viuCreeper = True
            #cv2.imshow("vermelho", mask)
    except:
        creeperLaranja = False
        viuCreeper = False

def filtra_ciano(img_in):
    global creeperCiano
    global centro_ciano
    global viuCreeper
    global maior_area_ciano
    img = img_in.copy()
    
    #Filtra ciano
    hvs1 = np.array([80, 50, 50],dtype=np.uint8)
    hsv2 = np.array([115, 255, 255],dtype=np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hvs1, hsv2)

    contornos, arvore = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    maior_area_ciano = 0.0
    for c in contornos:
        area = cv2.contourArea(c)
        if area > maior_area_ciano:
            maior_area_ciano = area   
    #Calcula centro de massa
    try:
        if maior_area_ciano > 1900.0:    
            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centro_ciano = []
            centro_ciano.append(int(cX))
            centro_ciano.append(int(cY))
            viuCreeper = True
            creeperCiano = True
    except:
        viuCreeper = False
        creeperCiano = False

#A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
    global area_aruco_50, area_aruco_100, area_aruco_150, area_aruco_200
    global cv_image
    global media
    global centro
    global resultados

    now = rospy.get_rostime()
    imgtime = imagem.header.stamp
    lag = now-imgtime # calcula o lag
    delay = lag.nsecs
    # print("delay ", "{:.3f}".format(delay/1.0E9))
    if delay > atraso and check_delay==True:
        # Esta logica do delay so" precisa ser usada com robo real e rede wifi 
        # serve para descartar imagens antigas
        print("Descartando por causa do delay do frame:", delay)
        return 
    try:
        temp_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
        # Note que os resultados já são guardados automaticamente na variável
        # chamada resultados

        centro, saida_net, resultados =  visao_module.processa(temp_image)        
        
        for r in resultados:
            # print(r) - print feito para documentar e entender
            # o resultado            
            pass

        # Desnecessário - Hough e MobileNet já abrem janelas
        cv_image = saida_net.copy()
        cv2.imshow("cv_image", cv_image)
        filtra_amarelo(cv_image)
        
        if not viuCreeper:            
            if cor == "green":
                filtra_verde(cv_image)
            if cor == "orange":
                filtra_laranja(cv_image)
            if cor == "blue":
                filtra_ciano(cv_image)
        
        if creeperLaranja:
            filtra_laranja(cv_image)
        
        if creeperVerde:
            filtra_verde(cv_image)
                    
        if creeperCiano:
            filtra_ciano(cv_image)

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        try:
            for i in range(len(ids)):
                if ids[i][0] == 50:
                    for c in corners[i]: 
                        for canto in c:
                            area_aruco_50 = (c[1][0]-c[0][0])**2
            
                if ids[i][0] == 100:
                    for c in corners[i]: 
                        area_aruco_100 = (c[1][0]-c[0][0])**2
        
                if ids[i][0] == 150:
                    for c in corners[i]: 
                        for canto in c:
                            area_aruco_150 = (c[1][0]-c[0][0])**2
    
                if ids[i][0] == 200:
                    for c in corners[i]: 
                        for canto in c:
                            area_aruco_200 = (c[1][0]-c[0][0])**2
        except:
            pass
        cv2.waitKey(1)
    except CvBridgeError as e:
        print("ex", e)
    
if __name__=="__main__":
    rospy.init_node("cor")

    topico_imagem = "/camera/image/compressed"

    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)
    recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)    
    ref_odometria = rospy.Subscriber("/odom", Odometry, recebe_odometria)

    print("Usando ", topico_imagem)

    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

    tfl = tf2_ros.TransformListener(tf_buffer) #conversao do sistema de coordenadas 
    tolerancia = 25

    zero = Twist(Vector3(0,0,0), Vector3(0,0,0))
    esq = Twist(Vector3(0.1,0,0), Vector3(0,0,0.2))
    dire = Twist(Vector3(0.1,0,0), Vector3(0,0,-0.2))    
    frente = Twist(Vector3(0.3,0,0), Vector3(0,0, 0.1))
    andaBifurcacao = Twist(Vector3(0.5,0,0), Vector3(0,0,-0.05))
    gira30graus = Twist(Vector3(0.0,0,0), Vector3(0,0,30*math.pi/180))
    gira45graus = Twist(Vector3(0.0,0,0), Vector3(0,0,45*math.pi/180))
    gira70graus = Twist(Vector3(0.0,0,0), Vector3(0,0,70*math.pi/180))
    gira180graus = Twist(Vector3(0.0,0,0), Vector3(0,0,math.pi))

    centro_tela  = 320
    margem_tela = 12

    #Flags
    anda = True
    pistaInteira = True 
    passou_aruco_100 = False
    passou_aruco_200 = False
    pegaCreeper = False 
    

    try:
        while not rospy.is_shutdown():
            #for r in resultados:
                #print(r)
            if pistaInteira:
                if passou_aruco_100:
                    if (posicao_geral[0] - 0.7 <= posicao_aruco_100[0] <= posicao_geral[0] +  0.7):
                        if (posicao_geral[1] - 0.3 <= posicao_aruco_100[1] <= posicao_geral[1] +  0.3):   
                            velocidade_saida.publish(zero)
                            rospy.sleep(2)
                            velocidade_saida.publish(gira45graus)
                            rospy.sleep(1)
                            passou_aruco_100 = [0,0]
                            passou_aruco_100 = False

                if passou_aruco_200:
                    if (posicao_geral[0] - 1 <= posicao_aruco_200[0] <= posicao_geral[0] +  1):
                        print("primeiroif")
                        if (posicao_geral[1] - 0.5 <= posicao_aruco_200[1] <= posicao_geral[1] +  0.5):  
                            print("segundoif") 
                            velocidade_saida.publish(zero)
                            rospy.sleep(2)
                            velocidade_saida.publish(gira45graus)
                            rospy.sleep(1)
                            passou_aruco_200 = [0,0]
                            passou_aruco_200 = False

                if area_aruco_50 > 26000 or area_aruco_150 > 26000:
                    velocidade_saida.publish(zero)
                    rospy.sleep(2)
                    velocidade_saida.publish(gira180graus)
                    rospy.sleep(1)
                    area_aruco_50 = 0
                    area_aruco_150 = 0

                if area_aruco_100 > 15000:
                    posicao_aruco_100 = posicao_geral
                    velocidade_saida.publish(zero)
                    rospy.sleep(2)
                    velocidade_saida.publish(gira30graus)
                    rospy.sleep(1)
                    velocidade_saida.publish(andaBifurcacao)
                    rospy.sleep(2)
                    area_aruco_100 = 0
                    passou_aruco_100 = True     
               
                if area_aruco_200 > 15500:
                    velocidade_saida.publish(zero)
                    rospy.sleep(0.5),
                    velocidade_saida.publish(frente)
                    rospy.sleep(2)
                    posicao_aruco_200 = posicao_geral
                    velocidade_saida.publish(gira70graus)
                    rospy.sleep(0.7)
                    velocidade_saida.publish(andaBifurcacao)
                    rospy.sleep(3)
                    area_aruco_200 = 0
                    passou_aruco_200 = True
        
                if anda:
                    if centro_pista[0] <  centro_tela - margem_tela: 
                        velocidade_saida.publish(esq)
                        rospy.sleep(0.001)

                    elif centro_pista[0] >  centro_tela + margem_tela: 
                        velocidade_saida.publish(dire)
                        rospy.sleep(0.001)
                    else: 
                        velocidade_saida.publish(frente)
                        rospy.sleep(0.001)
            
            if pegaCreeper:
                if viuCreeper:
                    pistaInteira = False
                   
                    if creeperCiano:
                        if centro_ciano[0] <  centro_tela - margem_tela: 
                            velocidade_saida.publish(esq)
                            rospy.sleep(0.0001)

                        elif centro_ciano[0] >  centro_tela + margem_tela: 
                            velocidade_saida.publish(dire)
                            rospy.sleep(0.0001)
                        else: 
                            velocidade_saida.publish(frente)
                            rospy.sleep(0.0001)
                    
                    if creeperVerde:
                        if centro_verde[0] <  centro_tela - margem_tela: 
                            velocidade_saida.publish(esq)
                            rospy.sleep(0.0001)

                        elif centro_verde[0] >  centro_tela + margem_tela: 
                            velocidade_saida.publish(dire)
                            rospy.sleep(0.0001)
                        else: 
                            velocidade_saida.publish(frente)
                            rospy.sleep(0.0001)

                    if creeperLaranja:
                        if centro_laranja[0] <  centro_tela - margem_tela: 
                            velocidade_saida.publish(esq)
                            rospy.sleep(0.0001)

                        elif centro_laranja[0] >  centro_tela + margem_tela: 
                            velocidade_saida.publish(dire)
                            rospy.sleep(0.0001)
                        else: 
                            velocidade_saida.publish(frente)
                            rospy.sleep(0.0001)
                    
                if distancia < 0.23:
                    if distancia != 0.0:
                        print("oi")
                        velocidade_saida.publish(gira180graus)
                        rospy.sleep(1)
                        pegaCreeper = False
                        pistaInteira = True
    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")