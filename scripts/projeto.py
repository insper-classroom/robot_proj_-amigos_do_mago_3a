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
from sensor_msgs.msg import Image, CompressedImage
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
centro_list = [0,0]
posicao_geral = [0,0]
posicao_aruco_100 = [0,0]
posicao_aruco_200 = [0,0]

area = 0.0 # Variavel com a area do maior contorno
area_aruco_50 = 0.0
area_aruco_100 = 0.0
area_aruco_150 = 0.0
area_aruco_200 = 0.0

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

def recebe_odometria(data):
    global x
    global y
    global posicao_geral

    x = data.pose.pose.position.x
    y = data.pose.pose.position.y

    quat = data.pose.pose.orientation
    posicao_geral = [quat.x, quat.y]

def filtra_amarelo(img_in):
    global centro_list
    img = img_in.copy()
    
    #Filtra amarelo
    hvs1 = np.array([22, 50, 50],dtype=np.uint8)
    hsv2 = np.array([36, 255, 255],dtype=np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hvs1, hsv2)

    #Calcula centro de massa
    M = cv2.moments(mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centro_list = []
    centro_list.append(int(cX))
    centro_list.append(int(cY))

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
        # Esta logica do delay so' precisa ser usada com robo real e rede wifi 
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
        print('ex', e)
    
if __name__=="__main__":
    rospy.init_node("cor")

    topico_imagem = "/camera/image/compressed"

    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)
    ref_odometria = rospy.Subscriber("/odom", Odometry, recebe_odometria)

    print("Usando ", topico_imagem)

    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

    tfl = tf2_ros.TransformListener(tf_buffer) #conversao do sistema de coordenadas 
    tolerancia = 25

    zero = Twist(Vector3(0,0,0), Vector3(0,0,0))
    esq = Twist(Vector3(0.1,0,0), Vector3(0,0,0.2))
    dire = Twist(Vector3(0.1,0,0), Vector3(0,0,-0.2))    
    frente = Twist(Vector3(0.5,0,0), Vector3(0,0, 0.15))
    gira30graus = Twist(Vector3(0.0,0,0), Vector3(0,0,30*math.pi/180))
    gira180graus = Twist(Vector3(0.0,0,0), Vector3(0,0,math.pi))

    centro_tela  = 320
    margem_tela = 12

    #Flags
    anda = True
    pistaInteira = True
    passou_aruco_100 = False
    passou_aruco_200 = False

    #Contadores
    cont_100 = 0
    cont_200 = 0

    try:
        while not rospy.is_shutdown():
            #for r in resultados:
                #print(r)
            if pistaInteira:
                if area_aruco_50 > 26000 or area_aruco_150 > 26000:
                    velocidade_saida.publish(zero)
                    rospy.sleep(2)
                    velocidade_saida.publish(gira180graus)
                    rospy.sleep(1)
                    area_aruco_50 = 0
                    area_aruco_150 = 0

                if area_aruco_100 > 15000:
                    velocidade_saida.publish(zero)
                    rospy.sleep(2)
                    velocidade_saida.publish(gira30graus)
                    rospy.sleep(1)
                    area_aruco_100 = 0
                    #posicao_aruco_100 = posicao_geral
                    #passou_aruco_100 = True     
                    #cont_100 += 1
               
                if area_aruco_200 > 15000:
                    velocidade_saida.publish(zero)
                    rospy.sleep(2),
                    velocidade_saida.publish(frente)
                    rospy.sleep(2)
                    velocidade_saida.publish(gira30graus)
                    rospy.sleep(1)
                    area_aruco_200 = 0
                    #posicao_aruco_200 = posicao_geral
                    #passou_aruco_200 = True
                    #cont_200 += 1

                #if passou_aruco_100:
                #    if posicao_geral[0] - 0.00001 <= posicao_aruco_100[0] <= posicao_geral[0] +  0.00001 and posicao_geral[0] - 0.00001 <= posicao_aruco_100[0] <= posicao_geral[0] +  0.00001:   
                #        print('100')
                #        velocidade_saida.publish(zero)
                #        rospy.sleep(2)
                #        velocidade_saida.publish(gira30graus)
                #        rospy.sleep(1)
                #    passou_aruco_100 = False

            #    if passou_aruco_200 and cont_200 >= 2:
            #        if posicao_geral[0] - 0.0009 <= posicao_aruco_200[0] <= posicao_geral[0]+  0.0009 and posicao_geral[1] - 0.00001 <= posicao_aruco_200[1] <= posicao_geral[1] + 0.00001:   
            #            print('200')
            #            velocidade_saida.publish(zero)
            #            rospy.sleep(2)
            #            velocidade_saida.publish(gira30graus)
            #            rospy.sleep(1)
            #        passou_aruco_200 = False
        #    #
        #    #print('0', posicao_geral[0], posicao_aruco_200[0], cont_200)
            if anda:
                if centro_list[0] <  centro_tela - margem_tela: 
                    velocidade_saida.publish(esq)
                    rospy.sleep(0.1)

                elif centro_list[0] >  centro_tela + margem_tela: 
                    velocidade_saida.publish(dire)
                    rospy.sleep(0.1)
                else: 
                    velocidade_saida.publish(frente)
                    rospy.sleep(0.1)
 

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")