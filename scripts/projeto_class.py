#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function, division

from numpy.core.numeric import False_
import rospy
import sys
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

class projeto:
    
    def __init__(self):
        '''Variaveis Globais'''
        self.cv_image = None
        self.media = []
        self.centro = []
        self.atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos
        self.centro_pista = [0,0]
        self.centro_ciano = [0,0]
        self.centro_verde = [0,0]
        self.centro_laranja = [0,0]
        self.centro_creeper = [0,0]
        self.posicao_geral = [0,0]
        self.posicao_aruco_100 = [math.inf, math.inf]
        self.posicao_aruco_200 = [math.inf, math.inf]
        self.distancia = 0
        self.area = 0.0 # Variavel com a area do maior contorno
        self.area_aruco_50 = 0.0
        self.area_aruco_100 = 0.0
        self.area_aruco_150 = 0.0
        self.area_aruco_200 = 0.0
        self.maior_area_laranja = 0.0
        self.maior_area_ciano = 0.0
        self.maior_area_verde = 0.0
        self.mx = 0.0

        '''Codigos do Aruco'''
        self.aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters  = aruco.DetectorParameters_create()
        self.parameters.minDistanceToBorder = 0
        self.parameters.adaptiveThreshWinSizeMax = 1000

        # Só usar se os relógios ROS da Raspberry e do Linux desktop estiverem sincronizados. 
        # Descarta imagens que chegam atrasadas demais
        self.check_delay = False 
        
        self.resultados = [] # Criacao de uma variavel global para guardar os resultados vistos
        
        self.x = 0
        self.y = 0
        self.z = 0 
        self.id = 0
        
        self.frame = "camera_link"
        #frame = "head_camera"  # DESCOMENTE para usar com webcam USB via roslaunch tag_tracking usbcam
        
        self.angle_z = None
        
        self.tfl = 0
        
        self.tf_buffer = tf2_ros.Buffer()
        
        '''Flags'''
        self.viuCreeper = False
        self.creeperLaranja = False
        self.creeperCiano = False
        self.creeperVerde = False
        

        '''Objetivo'''
        #self.goal = ("blue", 12, "dog")
        #self.goal = ("green", 23, "horse")
        self.goal = ("orange", 11, "cow")

        self.cor = self.goal[0]
        self.id_aruco = self.goal[1]
        self.estacao = self.goal[2]

        rospy.init_node("cor")
        #rospy.init_node("garra")
    
        self.topico_imagem = "/camera/image/compressed"
    
        recebedor = rospy.Subscriber(self.topico_imagem, CompressedImage, self.roda_todo_frame, queue_size=4, buff_size = 2**24)
        recebe_scan = rospy.Subscriber("/scan", LaserScan, self.scaneou)    
        ref_odometria = rospy.Subscriber("/odom", Odometry, self.recebe_odometria)
    
        print("Usando ", self.topico_imagem)
    
        self.velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

        self.ombro = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
        self.garra = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)

    
        self.tfl = tf2_ros.TransformListener(self.tf_buffer) #conversao do sistema de coordenadas 
        self.tolerancia = 25
        
        self.v_lento = 0.2
        self.v_rapido = 0.45
        self.w_lento = 0.17
        self.w_rapido = 0.40
    
        self.zero = Twist(Vector3(0,0,0), Vector3(0,0,0))
        self.esq = Twist(Vector3(0.1,0,0), Vector3(0,0,0.2))
        self.dire = Twist(Vector3(0.1,0,0), Vector3(0,0,-0.2))    
        self.frente = Twist(Vector3(0.3,0,0), Vector3(0,0, 0.1))
        self.andaBifurcacao = Twist(Vector3(0.5,0,0), Vector3(0,0,-0.05))
        self.gira30graus = Twist(Vector3(0.0,0,0), Vector3(0,0,30*math.pi/180))
        self.gira45graus = Twist(Vector3(0.0,0,0), Vector3(0,0,45*math.pi/180))
        self.gira50graus = Twist(Vector3(0.0,0,0), Vector3(0,0,-45*math.pi/180))
        self.gira70graus = Twist(Vector3(0.0,0,0), Vector3(0,0,70*math.pi/180))
        self.gira71graus = Twist(Vector3(0.0,0,0), Vector3(0,0,-71*math.pi/180))
        self.gira180graus = Twist(Vector3(0.0,0,0), Vector3(0,0,math.pi))
    
        self.centro_tela  = 320
        self.margem_tela = 12
    
        #Flags
        self.anda = True
        self.pistaInteira = True 
        self.passou_aruco_100 = False
        self.passou_aruco_200 = False
        self.pegaCreeper = True 
        self.pegouCreeper = False
        self.viuEstacao = False
    
        self.INICIAL= -1
        self.AVANCA = 0
        self.ALINHAPISTA = 1
        self.FIM = 2
        self.BIFURCACAO = 3
        self.BIFURCACAO2 = 4
        self.FIMDEPISTA = 5
        self.BIFURCACAOVOLTA = 6
        self.ALINHACREEPER = 7
        self.PEGACREEPER = 8
        self.ALINHAESTACAO = 9
        self.SOLTACREEPER = 10
        
    
        self.tempo_aruco_100 = 0
        self.tempo_aruco_200 = 0

        self.state = self.INICIAL

        self.acoes = {
            self.INICIAL: self.inicial, 
            self.AVANCA: self.avanca, 
            self.ALINHAPISTA: self.alinhapista, 
            self.FIM: self.fim,
            self.BIFURCACAO: self.bifurcacao, 
            self.BIFURCACAO2: self.bifurcacao2, 
            self.FIMDEPISTA: self.fimdepista, 
            self.BIFURCACAOVOLTA: self.bifurcacaovolta, 
            self.ALINHACREEPER: self.alinhacreeper, 
            self.PEGACREEPER: self.pegacreeper,
            self.ALINHAESTACAO: self.alinhaEstacao,
            self.SOLTACREEPER: self.soltaCreeper
            }
        
        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            #print("Estado: ", self.state)
            self.acoes[self.state]()
            self.controle()            
            r.sleep()

    def scaneou(self, dado):
        global distancia
        self.ranges = np.array(dado.ranges).round(decimals=2)
        self.distancia = self.ranges[0]

    def recebe_odometria(self, data):
        global x
        global y
        global posicao_geral

        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y

        self.posicao_geral = [self.x, self.y]

    def filtra_amarelo(self, img_in):
        '''
        Filtra o amarelo da pista e registra seu centro de massa
        '''
        global centro_pista
        self.img = img_in.copy()

        #Filtra amarelo
        hvs1 = np.array([22, 50, 50],dtype=np.uint8)
        hsv2 = np.array([36, 255, 255],dtype=np.uint8)

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hvs1, hsv2)

        #Calcula centro de massa
        try:
            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            self.centro_pista = []
            self.centro_pista.append(int(cX))
            self.centro_pista.append(int(cY))
            self.viuCreeper = False
        except:
            pass

    def filtra_verde(self, img_in):
        '''
        Filtra o verde do creeper e registra seu centro de massa
        '''
        global creeperVerde
        global centro_verde
        global viuCreeper
        global maior_area_verde
        self.img = img_in.copy()

        #Filtra verde
        hvs1 = np.array([45, 50, 50],dtype=np.uint8)
        hsv2 = np.array([80, 255, 255],dtype=np.uint8)

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hvs1, hsv2)
        contornos, arvore = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        self.maior_area_verde = 0.0
        for c in contornos:
            area = cv2.contourArea(c)
            if area > self.maior_area_verde:
                self.maior_area_verde = area
        #Calcula centro de massa
        try:
            if self.maior_area_verde > 400.0:    
                M = cv2.moments(mask)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                self.centro_verde = []
                self.centro_verde.append(int(cX))
                self.centro_verde.append(int(cY))
                self.creeperVerde = True
                self.viuCreeper = True
        except:
            self.creeperVerde = False
            self.viuCreeper = False

    def filtra_laranja(self, img_in):
        '''
        Filtra o laranja do creeper e registra seu centro de massa
        '''
        global creeperLaranja
        global centro_laranja
        global viuCreeper
        global maior_area_laranja
        self.img = img_in.copy()

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        #Filtra laranja
        hvs1 = np.array([0, 50, 150],dtype=np.uint8)
        hsv2 = np.array([6, 255, 255],dtype=np.uint8)

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hvs1, hsv2)

        contornos, arvore = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        self.maior_area_laranja = 0.0
        for c in contornos:
            area = cv2.contourArea(c)
            if area > self.maior_area_laranja:
                self.maior_area_laranja = area

        #Calcula centro de massa and 
        try:
            if self.maior_area_laranja > 1000.0:
                M = cv2.moments(mask)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                self.centro_laranja = []
                self.centro_laranja.append(int(cX))
                self.centro_laranja.append(int(cY))
                self.creeperLaranja = True
                self.viuCreeper = True
                #cv2.imshow("vermelho", mask)
        except:
            self.creeperLaranja = False
            self.viuCreeper = False

    def filtra_ciano(self, img_in):
        '''
        Filtra o ciano do creeper e registra seu centro de massa
        '''
        global creeperCiano
        global centro_ciano
        global viuCreeper
        global maior_area_ciano
        self.img = img_in.copy()

        #Filtra ciano
        hvs1 = np.array([80, 50, 50],dtype=np.uint8)
        hsv2 = np.array([115, 255, 255],dtype=np.uint8)

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hvs1, hsv2)

        contornos, arvore = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        self.maior_area_ciano = 0.0
        for c in contornos:
            area = cv2.contourArea(c)
            if area > self.maior_area_ciano:
                self.maior_area_ciano = area   
        #Calcula centro de massa
        try:
            if self.maior_area_ciano > 800.0:    
                M = cv2.moments(mask)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                self.centro_ciano = []
                self.centro_ciano.append(int(cX))
                self.centro_ciano.append(int(cY))
                self.viuCreeper = True
                self.creeperCiano = True
        except:
            self.viuCreeper = False
            self.creeperCiano = False

    #A função a seguir é chamada sempre que chega um novo frame
    def roda_todo_frame(self, imagem):
        global area_aruco_50, area_aruco_100, area_aruco_150, area_aruco_200
        global cv_image
        global media
        global centro
        global resultados
        global viuEstacao

        now = rospy.get_rostime()
        imgtime = imagem.header.stamp
        lag = now-imgtime # calcula o lag
        delay = lag.nsecs
        # print("delay ", "{:.3f}".format(delay/1.0E9))
        if delay > self.atraso and self.check_delay==True:
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
                if r[0] == self.estacao:
                    if self.pegouCreeper:
                        if r[0] == 'horse' and r[1] > 55.0:
                            self.viuEstacao = True 
                        elif r[1] > 90.0:
                            self.viuEstacao = True 

            # Desnecessário - Hough e MobileNet já abrem janelas
            cv_image = saida_net.copy()
            cv2.imshow("cv_image", cv_image)
            self.filtra_amarelo(cv_image)

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            try:
                if not self.viuCreeper:            
                    if self.cor == "green":
                        for i in range(len(ids)):
                            if ids[i][0] == self.id_aruco:
                                self.filtra_verde(cv_image)
                    if self.cor == "orange":
                        for i in range(len(ids)):
                            if ids[i][0] == self.id_aruco:
                                self.filtra_laranja(cv_image)
                    if self.cor == "blue":
                        for i in range(len(ids)):
                            if ids[i][0] == self.id_aruco:
                                self.filtra_ciano(cv_image)
            except:
                pass

            if self.creeperLaranja:
                self.filtra_laranja(cv_image)

            if self.creeperVerde:
                self.filtra_verde(cv_image)

            if self.creeperCiano:
                self.filtra_ciano(cv_image)

            try:
                for i in range(len(ids)):
                    if ids[i][0] == 50:
                        for c in corners[i]: 
                            for canto in c:
                                self.area_aruco_50 = (c[1][0]-c[0][0])**2

                    if ids[i][0] == 100:
                        for c in corners[i]: 
                            self.area_aruco_100 = (c[1][0]-c[0][0])**2

                    if ids[i][0] == 150:
                        for c in corners[i]: 
                            for canto in c:
                                self.area_aruco_150 = (c[1][0]-c[0][0])**2

                    if ids[i][0] == 200:
                        for c in corners[i]: 
                            for canto in c:
                                self.area_aruco_200 = (c[1][0]-c[0][0])**2
            except:
                pass
            

            cv2.waitKey(1)
        except CvBridgeError as e:
            print("ex", e)


    def inicial(self):
        '''
        Guarda o tempo de quando o robo passa pelos arucos das bifurcações
        '''
        global tempo_aruco_100, tempo_aruco_200
        self.tempo_aruco_100 = rospy.Time.now()
        self.tempo_aruco_200 = rospy.Time.now()
        

    def avanca(self):
        '''
        Anda pra frente
        '''
        self.vel = self.frente 
        self.velocidade_saida.publish(self.vel) 

    def alinhapista(self):
        '''
        Alinha o robo na pista
        '''
        self.delta_x = self.centro_tela - self.centro_pista[0]
        self.max_delta = 150.0
        self.w = (self.delta_x/self.max_delta)*self.w_rapido
        self.vel = Twist(Vector3(self.v_lento,0,0), Vector3(0,0,self.w)) 
        self.velocidade_saida.publish(self.vel)        

    def fim(self): 
        '''
        Zera a velocidade do robo
        '''
        self.vel = self.zero        
        self.velocidade_saida.publish(self.vel)

    def bifurcacao(self):
        '''
        Gira 30 graus e zera a area do Aruco quando chega na bifurcação "da perna"
        '''
        #global area_aruco_100
        self.vel = self.gira30graus        
        self.velocidade_saida.publish(self.vel)
        comeca_girar = rospy.Time.now()
        while rospy.Time.now() - comeca_girar <= rospy.Duration.from_sec(1.0):
            pass
        self.area_aruco_100 = 0 

    def bifurcacao2(self):
        '''
        Gira 70 graus e zera a area do Aruco quando chega na bifurcação "da cabeça"
        '''
        global area_aruco_200
        self.vel = self.gira71graus
        self.velocidade_saida.publish(self.vel)
        comeca_girar = rospy.Time.now()
        while rospy.Time.now() - comeca_girar <= rospy.Duration.from_sec(1.0):
            pass
        self.area_aruco_200 = 0 

    def bifurcacaovolta(self):
        '''
        Gira 30 graus quando passa pela segunda vez por uma bifurcação
        '''
        global posicao_aruco_100, posicao_aruco_200
        self.vel = self.gira30graus        
        self.velocidade_saida.publish(self.vel)
        comeca_girar = rospy.Time.now()
        while rospy.Time.now() - comeca_girar <= rospy.Duration.from_sec(1.0):
            pass
        self.posicao_aruco_100 = [math.inf, math.inf]
        self.posicao_aruco_200 = [math.inf, math.inf]
   
    def fimdepista(self):
        '''
        Gira 180 graus quando chega ao fim de uma das "pernas" da pista
        '''
        global area_aruco_50, area_aruco_150
        self.vel = self.gira180graus
        self.velocidade_saida.publish(self.vel)
        comeca_girar = rospy.Time.now()
        while rospy.Time.now() - comeca_girar <= rospy.Duration.from_sec(1.0):
            pass
        self.area_aruco_50 = 0
        self.area_aruco_150 = 0 
    
    def alinhacreeper(self):
        '''
        Alinha o robo ao creeper definido acima
        '''
        if self.creeperCiano:
            centro_creeper = self.centro_ciano
        if self.creeperLaranja:
            centro_creeper = self.centro_laranja
        if self.creeperVerde:
            centro_creeper = self.centro_verde
        self.ombro.publish(-0.6) ## para cima
        self.garra.publish(-1.0) ## Aberto
        self.delta_x = self.centro_tela - centro_creeper[0]
        self.max_delta = 150.0
        self.w = (self.delta_x/self.max_delta)*self.w_rapido
        self.vel = Twist(Vector3(self.v_lento,0,0), Vector3(0,0,self.w)) 
        self.velocidade_saida.publish(self.vel)  

    def pegacreeper(self):
        '''
        Pega o creeper e volta pra pista
        '''
        global pegaCreeper
        global pistaInteira
        global pegouCreeper 
        self.garra.publish(0.0)  ## Fechado
        self.ombro.publish(2.0) ## para cima
        if self.cor == 'green':
            self.velocidade_saida.publish(self.gira180graus)
        else:
            self.velocidade_saida.publish(self.gira45graus)
        comeca_girar = rospy.Time.now()
        while rospy.Time.now() - comeca_girar <= rospy.Duration.from_sec(1.0):
            pass
        self.pegaCreeper = False
        self.pistaInteira = True 
        self.pegouCreeper = True

    def alinhaEstacao(self):
        global state
        global mx
        for r in resultados:
            mc = 20
            print(self.mx)
            if r[0] == self.estacao:
                self.mx = (r[2][0] + r[3][0])/ 2
                self.mx = int(self.mx)
            self.delta_x = self.centro_tela - self.mx
            self.max_delta = 150.0
            self.w = (self.delta_x/self.max_delta)*self.w_rapido
            self.vel = Twist(Vector3(self.v_lento,0,0), Vector3(0,0,self.w)) 
            self.velocidade_saida.publish(self.vel)  
    
    def soltaCreeper(self):
        self.ombro.publish(-1.0)
        self.garra.publish(-1.0) 
        self.velocidade_saida.publish(self.zero)        
        comeca_girar = rospy.Time.now()
        while rospy.Time.now() - comeca_girar <= rospy.Duration.from_sec(2.0):
            pass
        self.velocidade_saida.publish(self.gira180graus)
        comeca_girar = rospy.Time.now()
        while rospy.Time.now() - comeca_girar <= rospy.Duration.from_sec(1.0):
            pass
        self.pistaInteira = True 
        self.pegouCreeper = False 
        self.viuEstacao =  False
        pass

    def controle(self):
        '''
        Controla qual estado executar
        '''
        global state
        global posicao_aruco_100, posicao_aruco_200    
        global area_aruco_100, area_aruco_200, area_aruco_50, area_aruco_150
        global tempo_aruco_100, tempo_aruco_200
        global pistaInteira
        global mx
        

        if self.pistaInteira:
            if self.centro_tela - self.margem_tela < self.centro_pista[0] < self.centro_tela + self.margem_tela:
                self.state = self.AVANCA
            else:
                self.state = self.ALINHAPISTA

        if self.area_aruco_50 > 26000 or self.area_aruco_150 > 26000:
            self.state = self.FIMDEPISTA

        if self.area_aruco_100 > 14000:
            self.posicao_aruco_100 = self.posicao_geral
            self.tempo_aruco_100 = rospy.Time.now()
            self.state = self.BIFURCACAO

        if self.area_aruco_200 > 19000:
            self.posicao_aruco_200 = self.posicao_geral
            self.tempo_aruco_200 = rospy.Time.now()
            self.state = self.BIFURCACAO2

        if rospy.Time.now() - self.tempo_aruco_100 >= rospy.Duration.from_sec(5.0):
            if (self.posicao_geral[0] - 0.7 <= self.posicao_aruco_100[0] <= self.posicao_geral[0] +  0.7):
                if (self.posicao_geral[1] - 0.3 <= self.posicao_aruco_100[1] <= self.posicao_geral[1] +  0.3):
                    self.state = self.BIFURCACAOVOLTA


        if rospy.Time.now() - self.tempo_aruco_200 >= rospy.Duration.from_sec(5.0):
            if (self.posicao_geral[0] - 1 <= self.posicao_aruco_200[0] <= self.posicao_geral[0] +  1):
                if (self.posicao_geral[1] - 0.5 <= self.posicao_aruco_200[1] <= self.posicao_geral[1] +  0.5):
                    self.state = self.BIFURCACAOVOLTA    

        if self.pegaCreeper:
            if self.viuCreeper:
                self.pistaInteira = False
                if self.centro_tela - self.margem_tela < self.centro_creeper[0] < self.centro_tela + self.margem_tela:
                    self.state = self.AVANCA
                else:
                    self.state = self.ALINHACREEPER            
                if self.distancia < 0.22:
                    if self.distancia != 0.0:
                        self.state = self.PEGACREEPER
        
        if self.pegouCreeper:
            print('pegou creeper')
            if self.viuEstacao:
                print('viu estacao')
                self.pistaInteira = False
                self.state = self.ALINHAESTACAO           
                if self.distancia < 0.40:
                    if self.distancia != 0.0:
                        self.state = self.SOLTACREEPER


if __name__ == '__main__':
    ic = projeto()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")