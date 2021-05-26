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
from std_msgs.msg import Float64

print("EXECUTE ANTES da 1.a vez: ")
print("wget https://github.com/Insper/robot21.1/raw/main/projeto/ros_projeto/scripts/MobileNetSSD_deploy.caffemodel")
print("PARA TER OS PESOS DA REDE NEURAL")

import visao_module

bridge = CvBridge()

'''Variaveis Globais'''
cv_image = None
media = []
centro = []
atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos
centro_pista = [0,0]
centro_ciano = [0,0]
centro_verde = [0,0]
centro_laranja = [0,0]
centro_creeper = [0,0]
posicao_geral = [0,0]
posicao_aruco_100 = [math.inf, math.inf]
posicao_aruco_200 = [math.inf, math.inf]
distancia = 0
area = 0.0 # Variavel com a area do maior contorno
area_aruco_50 = 0.0
area_aruco_100 = 0.0
area_aruco_150 = 0.0
area_aruco_200 = 0.0
maior_area_laranja = 0.0
maior_area_ciano = 0.0
maior_area_verde = 0.0

'''Codigos do Aruco'''
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

'''Flags'''
viuCreeper = False
creeperLaranja = False
creeperCiano = False
creeperVerde = False


'''Objetivo'''
lista = ('orange', 11, 'car')

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

    posicao_geral = [x, y]

def filtra_amarelo(img_in):
    '''
    Filtra o amarelo da pista e registra seu centro de massa
    '''
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
    '''
    Filtra o verde do creeper e registra seu centro de massa
    '''
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
        if maior_area_verde > 400.0:    
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
    '''
    Filtra o laranja do creeper e registra seu centro de massa
    '''
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
        if maior_area_laranja > 1000.0:
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
    '''
    Filtra o ciano do creeper e registra seu centro de massa
    '''
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
        if maior_area_ciano > 800.0:    
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

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        try:
            if not viuCreeper:            
                if cor == "green":
                    for i in range(len(ids)):
                        if ids[i][0] == id_aruco:
                            filtra_verde(cv_image)
                if cor == "orange":
                    for i in range(len(ids)):
                        if ids[i][0] == id_aruco:
                            filtra_laranja(cv_image)
                if cor == "blue":
                    for i in range(len(ids)):
                        if ids[i][0] == id_aruco:
                            
                            filtra_ciano(cv_image)
        except:
            pass

        if creeperLaranja:
            filtra_laranja(cv_image)
        
        if creeperVerde:
            filtra_verde(cv_image)
                    
        if creeperCiano:
            filtra_ciano(cv_image)

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
    #rospy.init_node("garra")

    topico_imagem = "/camera/image/compressed"

    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)
    recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)    
    ref_odometria = rospy.Subscriber("/odom", Odometry, recebe_odometria)

    print("Usando ", topico_imagem)

    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
    
    ombro = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
    garra = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)


    tfl = tf2_ros.TransformListener(tf_buffer) #conversao do sistema de coordenadas 
    tolerancia = 25
    
    v_lento = 0.2
    v_rapido = 0.45
    w_lento = 0.17
    w_rapido = 0.40

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
    pegaCreeper = True 


    INICIAL= -1
    AVANCA = 0
    ALINHAPISTA = 1
    FIM = 2
    BIFURCACAO = 3
    BIFURCACAO2 = 4
    FIMDEPISTA = 5
    BIFURCACAOVOLTA = 6
    ALINHACREEPER = 7
    PEGACREEPER = 8
    
    
    tempo_aruco_100 = 0
    tempo_aruco_200 = 0

    state = INICIAL

    def inicial():
        '''
        Guarda o tempo de quando o robo passa pelos arucos das bifurcações
        '''
        global tempo_aruco_100, tempo_aruco_200
        tempo_aruco_100 = rospy.Time.now()
        tempo_aruco_200 = rospy.Time.now()
        

    def avanca():
        '''
        Anda pra frente
        '''
        vel = frente 
        velocidade_saida.publish(vel) 

    def alinhapista():
        '''
        Alinha o robo na pista
        '''
        delta_x = centro_tela - centro_pista[0]
        max_delta = 150.0
        w = (delta_x/max_delta)*w_rapido
        vel = Twist(Vector3(v_lento,0,0), Vector3(0,0,w)) 
        velocidade_saida.publish(vel)        

    def fim(): 
        '''
        Zera a velocidade do robo
        '''
        vel = zero        
        velocidade_saida.publish(vel)

    def bifurcacao():
        '''
        Gira 30 graus e zera a area do Aruco quando chega na bifurcação "da perna"
        '''
        global area_aruco_100
        vel = gira30graus        
        velocidade_saida.publish(vel)
        rospy.sleep(1)
        area_aruco_100 = 0 

    def bifurcacao2():
        '''
        Gira 70 graus e zera a area do Aruco quando chega na bifurcação "da cabeça"
        '''
        global area_aruco_200
        vel = gira70graus
        velocidade_saida.publish(vel)
        rospy.sleep(1)
        area_aruco_200 = 0 

    def bifurcacaovolta():
        '''
        Gira 30 graus quando passa pela segunda vez por uma bifurcação
        '''
        global posicao_aruco_100, posicao_aruco_200
        vel = gira30graus        
        velocidade_saida.publish(vel)
        rospy.sleep(1)
        posicao_aruco_100 = [math.inf, math.inf]
        posicao_aruco_200 = [math.inf, math.inf]
   
    def fimdepista():
        '''
        Gira 180 graus quando chega ao fim de uma das "pernas" da pista
        '''
        global area_aruco_50, area_aruco_150
        vel = gira180graus
        velocidade_saida.publish(vel)
        rospy.sleep(1)
        area_aruco_50 = 0
        area_aruco_150 = 0 
    
    def alinhacreeper():
        '''
        Alinha o robo ao creeper definido acima
        '''
        if creeperCiano:
            centro_creeper = centro_ciano
        if creeperLaranja:
            centro_creeper = centro_laranja
        if creeperVerde:
            centro_creeper = centro_verde
        ombro.publish(-0.6) ## para cima
        garra.publish(-1.0) ## Aberto
        delta_x = centro_tela - centro_creeper[0]
        max_delta = 150.0
        w = (delta_x/max_delta)*w_rapido
        vel = Twist(Vector3(v_lento,0,0), Vector3(0,0,w)) 
        velocidade_saida.publish(vel)  

    def pegacreeper():
        '''
        Pega o creeper e volta pra pista
        '''
        global pegaCreeper
        global pistaInteira
        garra.publish(0.0)  ## Fechado
        velocidade_saida.publish(gira180graus)
        rospy.sleep(1)
        pegaCreeper = False
        pistaInteira = True 

   
    def controle():
        '''
        Controla qual estado executar
        '''
        global state
        global posicao_aruco_100, posicao_aruco_200    
        global area_aruco_100, area_aruco_200, area_aruco_50, area_aruco_150
        global tempo_aruco_100, tempo_aruco_200
        global pistaInteira
        

        if pistaInteira:
            if centro_tela - margem_tela < centro_pista[0] < centro_tela + margem_tela:
                state = AVANCA
            else:
                state = ALINHAPISTA

        if area_aruco_50 > 26000 or area_aruco_150 > 26000:
            state = FIMDEPISTA

        if area_aruco_100 > 14000:
            posicao_aruco_100 = posicao_geral
            tempo_aruco_100 = rospy.Time.now()
            state = BIFURCACAO

        if area_aruco_200 > 15500:
            posicao_aruco_200 = posicao_geral
            tempo_aruco_200 = rospy.Time.now()
            state = BIFURCACAO2

        if rospy.Time.now() - tempo_aruco_100 >= rospy.Duration.from_sec(5.0):
            if (posicao_geral[0] - 0.7 <= posicao_aruco_100[0] <= posicao_geral[0] +  0.7):
                if (posicao_geral[1] - 0.3 <= posicao_aruco_100[1] <= posicao_geral[1] +  0.3):
                    state = BIFURCACAOVOLTA


        if rospy.Time.now() - tempo_aruco_200 >= rospy.Duration.from_sec(5.0):
            if (posicao_geral[0] - 1 <= posicao_aruco_200[0] <= posicao_geral[0] +  1):
                if (posicao_geral[1] - 0.5 <= posicao_aruco_200[1] <= posicao_geral[1] +  0.5):
                    state = BIFURCACAOVOLTA    

        if pegaCreeper:
            if viuCreeper:
                pistaInteira = False
                if centro_tela - margem_tela < centro_creeper[0] < centro_tela + margem_tela:
                    state = AVANCA
                else:
                    state = ALINHACREEPER            
                if distancia < 0.22:
                    if distancia != 0.0:
                        state = pegacreeper
    acoes = {
        INICIAL:inicial, 
        AVANCA: avanca, 
        ALINHAPISTA: alinhapista, 
        FIM:fim,
        BIFURCACAO: bifurcacao, 
        BIFURCACAO2: bifurcacao2, 
        FIMDEPISTA:fimdepista, 
        BIFURCACAOVOLTA:bifurcacaovolta, 
        ALINHACREEPER:alinhacreeper, 
        PEGACREEPER:pegacreeper}
        
    r = rospy.Rate(100) 

    try:
        while not rospy.is_shutdown():
            print("Estado: ", state)
            acoes[state]()
            controle()            
            r.sleep()

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")