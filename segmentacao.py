import scipy
import matplotlib
from scipy import ndimage, signal
import skimage.morphology as sm
from scipy import signal
import numpy as np
from skimage import io
import math


def aplicacao_mascara(imagem, mascara):
    imagem_retina = imagem * mascara
    
    return imagem_retina



def padding(imagem, comprimento):
    centro = int(comprimento/2) #dividir as margens por igual, vai ser um ímpar, logo buscamos só o inteiro
    imagem_padding = np.zeros((imagem.shape[0]+2*centro,imagem.shape[1]+2*centro))
    imagem_padding[centro:-centro, centro:-centro] = imagem #padding com a imagem anterior
    
    return imagem_padding



def inversao(imagem):
    nova_imagem = imagem.copy()
    nova_imagem = 255 - imagem #0 passa a 255 e vice-versa
    
    return nova_imagem



def processamento_bordas(imagem, mascara, comprimento):
    imagem_processada = imagem.copy()
    centro = int(comprimento/2) #centrar o kernel
    mascara_padding = padding(mascara, comprimento)
    for x in range (centro, imagem.shape[0]-centro): #correr apenas a imagem sem o padding, por causa do kernel
        for y in range (centro, imagem.shape[1]-centro):
            if (mascara_padding[x,y] == 0 ): #apenas interessam os pontos exteriores à máscara
                kernel = imagem[x-centro:x+centro+1,y-centro:y+centro+1] #vai buscar os pontos vizinhos
                for i in range(-centro,centro+1): #corre todos os pontos do kernel criado
                    for j in range(-centro,centro+1):
                        if (mascara_padding[x+i,y+j] == 0): #apenas nos interessa dentro os pontos da retina
                            kernel[i+centro,j+centro] = 0 #se for fora, será igual a 0
                contagem = 0
                somatorio = 0
                for i in range(kernel.shape[0]):
                    for j in range(kernel.shape[1]):
                        if (kernel[i,j] != 0):
                            contagem = contagem + 1
                            somatorio = somatorio + kernel[i,j]
                if (contagem != 0):
                    imagem_processada[x,y] = somatorio / contagem #correlação ponto a ponto
                    
    return imagem_processada



def correlacao(imagem, kernel, mode):
    nova_imagem = imagem.copy() #cópia da imagem orignal de modo a não a alterar/danificar
    imagem_correlacionada = signal.correlate2d(nova_imagem, kernel, mode)
    
    return (imagem_correlacionada)



def criacao_kernels(comprimento):
    kernel = np.zeros((comprimento,comprimento,12)) #criação de um kernel com dimensões dadas pelo utilizador
    centro = int(comprimento/2) #definição do ponto central do kernel
    for i in range(0,90,15): #apenas é necessário correr para um quadrante
        radianos = i*math.pi/180 #conversão de graus para radianos
        for x in range (-centro,centro+1): #0 é no ponto central do kernel, inversão porque causa do sistema de coordenadas estar também invertido
            for y in range (centro,-centro-1,-1): #0 é no ponto central do kernel
                if (i == 0):
                    kernel[centro,:,0] = 1 #kernel a 0 graus
                    kernel[:,centro,6] = 1 #kernel a 90 graus
                if (y == round(math.tan(radianos)*x)):
                    if (i == 15):
                        kernel[-y+centro, x+centro, 1] = 1 #kernel a 15 graus
                        kernel[y+centro, x+centro, 11] = 1 #kernel a 165 graus
                        kernel[x+centro, -y+centro, 5] = 1 #kernel a 75 graus
                        kernel[x+centro, y+centro, 7] = 1 #kernel a 105 graus
                    elif (i == 30):
                        kernel[-y+centro, x+centro, 2] = 1 #kernel a 30 graus
                        kernel[y+centro, x+centro, 10] = 1 #kernel a 150 graus
                        kernel[x+centro, -y+centro, 4] = 1 #kernel a 60 graus
                        kernel[x+centro, y+centro, 8] = 1 #kernel a 120 graus
                    elif (i == 45):
                        kernel[-y+centro, x+centro, 3] = 1 #kernel a 45 graus
                        kernel[y+centro, x+centro, 9] = 1 #kernel a 135 graus  
                        
    return kernel/comprimento #por causa da correlação



def calculo_imagem_S(imagem, comprimento): #S(i,j)
    imagem_L = np.ones((imagem.shape)) * -1e12 #L(i,j) garantir que as entradas são extremamente grandes e negativas
    mapa = np.zeros((imagem.shape)) 
    kernels = criacao_kernels(comprimento) #função acima criada
    kernel_ones = np.ones((comprimento,comprimento)) / (comprimento*comprimento) #N(i,j) equivale à aplicaçao de um filtro média 
    imagem_N = correlacao(imagem, kernel_ones, 'same')
    for i in range(12): #para percorrer os 12 kernels e convoluir um a um com a imagem
        imagem_correlacionada = correlacao(imagem, kernels[:,:,i], 'same') #correlação da imagem com um kernel
        for x in range (imagem_correlacionada.shape[0]):
            for y in range (imagem_correlacionada.shape[1]):
                if (imagem_correlacionada[x,y] > imagem_L[x,y]): #seleção do ponto máximo
                    imagem_L[x,y] = imagem_correlacionada[x,y] #L(i,j)
                    mapa[x,y] = i * 15 #no mapa vai ficar apontado o ângulo que deu o valor máximo
                    
    imagem_S = imagem_L - imagem_N #S(i,j) = L(i,j) - N(i,j)
    return imagem_S, mapa



def criacao_kernels_ortogonais(comprimento):
    kernel = np.zeros((comprimento,comprimento,12)) #criação de um kernel com dimensões dadas pelo utilizador
    centro = int(comprimento/2) #definição do ponto central do kernel
    for i in range(0,90,45): #apenas é necessário correr para um quadrante, conseguimos defenir todos
        radianos = i*math.pi/180 #conversão de graus para radianos
        for x in range (-centro,centro+1): #0 é no ponto central do kernel, inversão porque causa do sistema de coordenadas estar também invertido
            for y in range (centro,-centro-1,-1): #0 é no ponto central do kernel
                if (i == 0):
                    #kernel para os ângulos 165, 0, 15
                    kernel[centro,centro-1:centro+2, 0] = 1
                    #kernel para os ângulos 75, 90, 105
                    kernel[centro-1:centro+2,centro, 2] = 1 
                if (y == round(math.tan(radianos)*x)):
                    if (i == 45):
                        #kernel para os ângulos 120, 135, 150
                        kernel[centro, centro, 1] = 1
                        kernel[centro-1, centro+1, 1] = 1
                        kernel[centro+1, centro-1, 1] = 1
                        #kernel para os ângulos 30, 45, 60
                        kernel[centro, centro, 3] = 1 
                        kernel[centro-1, centro-1, 3] = 1
                        kernel[centro+1, centro+1, 3] = 1
                        
    return kernel/3 #por causa da correlação



def calculo_imagem_S0(imagem, mapa, comprimento): #S0(i,j)
    kernels = criacao_kernels_ortogonais(comprimento) #função acima criada
    imagem_L0 = np.zeros((imagem.shape)) #L0(i,j)
    kernel = np.ones((comprimento,comprimento)) / (comprimento*comprimento) #N(i,j) equivale à aplicaçao de um filtro média 
    imagem_N = correlacao(imagem, kernel, 'same')
    centro = int(comprimento/2)
    for x in range (centro, imagem.shape[0]-centro):
        for y in range (centro, imagem.shape[1]-centro):
            parte_imagem = imagem[x-centro:x+centro+1,y-centro:y+centro+1]
            if (mapa[x,y] == 0 or mapa[x,y] == 15 or mapa[x,y] == 165):
                imagem_L0[x,y] = correlacao(parte_imagem, kernels[:,:,0], 'valid') #L0(i,j) => kernel para os ângulos 165, 0, 15
            elif (mapa[x,y] == 120 or mapa[x,y] == 135 or mapa[x,y] == 150):
                imagem_L0[x,y] = correlacao(parte_imagem, kernels[:,:,1], 'valid') #L0(i,j) => kernel para os ângulos 120, 135, 150
            elif (mapa[x,y] == 75 or mapa[x,y] == 90 or mapa[x,y] == 105): 
                imagem_L0[x,y] = correlacao(parte_imagem, kernels[:,:,2], 'valid') #L0(i,j) => kernel para os ângulos 75, 90, 105
            elif (mapa[x,y] == 30 or mapa[x,y] == 45 or mapa[x,y] == 60):
                imagem_L0[x,y] = correlacao(parte_imagem, kernels[:,:,3], 'valid') #L0(i,j) => kernel para os ângulos 30, 45, 60
    imagem_S0 = imagem_L0 - imagem_N #S0(i,j) = L0(i,j) - N(i,j)
    
    return imagem_S0



def binarizacao(imagem, threshold):
    imagem_binaria = np.zeros((imagem.shape))
    for x in range (imagem.shape[0]):
        for y in range (imagem.shape[1]):
            if (imagem[x,y] > threshold):
                imagem_binaria[x,y] = 1
                
    return imagem_binaria



def unpadding(imagem, comprimento):
    centro = int(comprimento/2)
    imagem_unpadding = ((imagem.shape[0]-centro*2, imagem.shape[1]-centro*2))
    imagem_unpadding = imagem[centro:-centro, centro:-centro]
    
    return imagem_unpadding



def calculo_metricas(imagem, mascara, mascara_vasos):
    verdadeiros_positivos = 0
    verdadeiros_negativos = 0
    falsos_positivos = 0
    falsos_negativos = 0
    for x in range (imagem.shape[0]):
        for y in range (imagem.shape[1]): 
            if (mascara[x,y] == 1):
                if (imagem[x,y] == 1 and mascara_vasos[x,y] == 1): #verdadeiros positivos => imagem calculada = 1 e mascara da DRIVE = 1
                    verdadeiros_positivos = verdadeiros_positivos + 1
                elif (imagem[x,y] == 0 and mascara_vasos[x,y] == 0): #verdadeiros negativos => imagem calculada = 0 e mascara da DRIVE = 0
                    verdadeiros_negativos = verdadeiros_negativos + 1
                elif (imagem[x,y] == 1 and mascara_vasos[x,y] == 0): #falsos positivos => imagem calculada = 1 e mascara da DRIVE = 0
                    falsos_positivos = falsos_positivos + 1
                elif (imagem[x,y] == 0 and mascara_vasos[x,y] == 1): #falsos negativos => imagem calculada = 0 e mascara da DRIVE = 1
                    falsos_negativos = falsos_negativos + 1
    sensibilidade = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos) * 100
    especificidade = verdadeiros_negativos / (verdadeiros_negativos + falsos_positivos) * 100
    exatidao = (verdadeiros_positivos + verdadeiros_negativos) / (verdadeiros_positivos + verdadeiros_negativos + falsos_positivos + falsos_negativos) * 100  
    
    return sensibilidade, especificidade, exatidao



def leitura_imagens(test_training):
    lista_imagens = []
    lista_mascaras = []
    lista_mascaras_vasos = []
    lista_final = []
    if (test_training == 'test'): #imagens 1 a 20
        inicio_contagem = 1
    elif (test_training == 'training'): #imagens 21 a 40
        inicio_contagem = 21
    else:
        print('ERRO! Nenhuma pasta encontra')
        
    for i in range(inicio_contagem, inicio_contagem+20):
        if (i < 10):
            numero = str(i)
            imagem = scipy.misc.imread('DRIVE/'+ test_training + '/images/0' + numero + '_test.tif')
            lista_imagens.append(imagem) #adicionar imagens, uma a uma
            mascara = scipy.misc.imread('DRIVE/'+ test_training + '/mask/mask0' + numero + '.png')
            lista_mascaras.append(mascara) #adicionar máscaras, uma a uma
            mascara_vasos = scipy.misc.imread('DRIVE/'+ test_training + '/1st_manual/0' + numero + '_manual1.gif')     
            lista_mascaras_vasos.append(mascara_vasos) #adicionar máscaras dos vasos, uma a uma
        elif (i > 9 and i < 21):
            numero = str(i)
            imagem = scipy.misc.imread('DRIVE/'+ test_training + '/images/' + numero + '_test.tif')
            lista_imagens.append(imagem) #adicionar imagens, uma a uma
            mascara = scipy.misc.imread('DRIVE/'+ test_training + '/mask/mask' + numero + '.png')
            lista_mascaras.append(mascara) #adicionar máscaras, uma a uma
            mascara_vasos = scipy.misc.imread('DRIVE/'+ test_training + '/1st_manual/' + numero + '_manual1.gif')
            lista_mascaras_vasos.append(mascara_vasos) #adicionar máscaras dos vasos, uma a uma
        elif (i > 20):
            numero = str(i)
            imagem = scipy.misc.imread('DRIVE/'+ test_training + '/images/' + numero + '_training.tif')
            lista_imagens.append(imagem) #adicionar imagens, uma a uma
            mascara = scipy.misc.imread('DRIVE/'+ test_training + '/mask/mask' + numero + '.png')
            lista_mascaras.append(mascara) #adicionar máscaras, uma a uma
            mascara_vasos = scipy.misc.imread('DRIVE/'+ test_training + '/1st_manual/' + numero + '_manual1.gif')
            lista_mascaras_vasos.append(mascara_vasos) #adicionar máscaras dos vasos, uma a uma
    lista_final.append(lista_imagens) #adicionar todas as imagens
    lista_final.append(lista_mascaras) #adicionar todas as máscaras
    lista_final.append(lista_mascaras_vasos) #adicionar todas as máscaras dos vasos
    
    return lista_final



def segmentacao(imagem, mascara, mascara_vasos, tamanho):
    
    #seleção da imagem no canal verde, melhor para detetar vasos sanguíneos 
    matplotlib.pyplot.gray()
    imagem_green = imagem[:,:,1]
    
    #aplicação da máscara criada no TPC5 para eliminar pixels indesejados exteriores à retina
    imagem_green_sem_bordas = aplicacao_mascara(imagem_green, mascara)
    
    #padding da imagem, tamanho extra equivale a uma margem, será igual ao inteiro da metade do dimensão do kernel 
    imagem_padding = padding(imagem_green_sem_bordas, tamanho)
    
    #inversão da imagem, porque os vasos estão mais escuros que o background
    imagem_padding_invertida = inversao(imagem_padding)
    imagem_copia_padding_invertida = imagem_padding_invertida.copy()
    imagem_unpadding_invertida = unpadding(imagem_copia_padding_invertida, tamanho) #para apresentar no relatório como no artigo
    
    #processamento das bordas, de modo a minimizar o efeito do círculo logo a seguir fora da FOV
    imagem_processada = processamento_bordas(imagem_padding_invertida, mascara, tamanho)

    #padding da máscara
    mascara_padding = padding(mascara, tamanho) 
    
    #cálculo e tratamento da imagem S(i,j)
    #aplicação da máscara e inversão da imagem S(i,j), para tornar os vasos mais escuros que o background e eliminar pontos exteriores à retina
    [imagem_S, mapa] = calculo_imagem_S(imagem_processada, tamanho)
    imagem_S_sem_bordas = aplicacao_mascara(imagem_S, mascara_padding) #aplicação da máscara to TPC5
    imagem_S_invertida = inversao(imagem_S_sem_bordas)
    imagem_S_invertida_unpadding = unpadding(imagem_S_invertida, tamanho) #para apresentar no relatório como no artigo
    imagem_S_final = unpadding(imagem_S_sem_bordas, tamanho) #imagem S(i,j) sem ser invertida e com unpadding
 
    #cálculo e tratamento da imagem S0(i,j)
    #aplicação da máscara e inversão da imagem S0(i,j), para tornar os vasos mais escuros que o background e eliminar pontos exteriores à retina
    imagem_S0 = calculo_imagem_S0(imagem_processada, mapa, tamanho)  
    imagem_S0_sem_bordas = aplicacao_mascara(imagem_S0, mascara_padding) #aplicação da máscara to TPC5
    imagem_S0_invertida = inversao(imagem_S0_sem_bordas)
    imagem_S0_invertida_unpadding = unpadding(imagem_S0_invertida, tamanho) #para apresentar no relatório como no artigo
    imagem_S0_final = unpadding(imagem_S0_sem_bordas, tamanho) #imagem S0(i,j) sem ser invertida e com unpadding
    
    #binarização da imagem => criação da imagem binária
    threshold = 5.5 #definir o melhor threshold
    imagem_binaria = binarizacao(imagem_S_final, threshold)
    
    #inversão da imagem binária para retornar como no artigo, de modo a facilitar a visuzaliação dos vasos
    imagem_segmentada = inversao(imagem_binaria)
    
    #cálculos das métricas
    [sensibilidade, especificidade, exatidao] = calculo_metricas(imagem_binaria, mascara, mascara_vasos)
    
    return imagem_unpadding_invertida, imagem_S_invertida_unpadding, imagem_S0_invertida_unpadding, imagem_segmentada, sensibilidade, especificidade, exatidao



def saida_imagens(test_training, imagem_invertida, imagem_S, imagem_S0, imagem_segmentada, i):
    if (i < 10):
        numero = str(i)
        scipy.misc.toimage(imagem_invertida).save('Imagens segmentadas/'+ test_training + '/imagens_invertidas/0' + numero + '_imagem_invertida.png')
        scipy.misc.toimage(imagem_S).save('Imagens segmentadas/'+ test_training + '/imagens_S/0' + numero + '_imagem_S.png')
        scipy.misc.toimage(imagem_S0).save('Imagens segmentadas/'+ test_training + '/imagens_S0/0' + numero + '_imagem_S0.png')
        scipy.misc.toimage(imagem_segmentada).save('Imagens segmentadas/'+ test_training + '/imagens_segmentadas/0' + numero + '_imagem_segmentada.png')
    else:
        numero = str(i)
        scipy.misc.toimage(imagem_invertida).save('Imagens segmentadas/'+ test_training + '/imagens_invertidas/' + numero + '_imagem_invertida.png')
        scipy.misc.toimage(imagem_S).save('Imagens segmentadas/'+ test_training + '/imagens_S/' + numero + '_imagem_S.png')
        scipy.misc.toimage(imagem_S0).save('Imagens segmentadas/'+ test_training + '/imagens_S0/' + numero + '_imagem_S0.png')
        scipy.misc.toimage(imagem_segmentada).save('Imagens segmentadas/'+ test_training + '/imagens_segmentadas/' + numero + '_imagem_segmentada.png')
            
    return



def geral():
    test_training = input('Insira o nome da pasta (test ou training) que deseja segmentar: ')
    tamanho = int(input('Insira o tamanho do kernel desejado (deverá ser ímpar): '))
    print() #parágrafo
    lista_final = leitura_imagens(test_training)
    ficheiro = open('Imagens segmentadas/' + test_training + '/Métricas ' + test_training + '.txt','w') #cria um ficheiro para guardar as métricas cálculadas
    ficheiro.write('IMAGEM SENSIBILIDADE ESPECIFICIDADE EXATIDÃO \n')
    
    for i in range(len(lista_final[0])):
        if (test_training == 'test'):
            numero = i + 1
            print('Imagem ', numero)
        elif (test_training == 'training'):
            numero = i + 21
            print('Imagem ', numero)
        [imagem_unpadding_invertida, imagem_S_invertida_unpadding, imagem_S0_invertida_unpadding, imagem_segmentada, sensibilidade, especificidade, exatidao] = segmentacao(lista_final[0][i], lista_final[1][i]/255, lista_final[2][i]/255, tamanho)
        saida_imagens(test_training, imagem_unpadding_invertida, imagem_S_invertida_unpadding, imagem_S0_invertida_unpadding, imagem_segmentada, numero)
        ficheiro.write('Imagem ' + str(numero) + '\t' + str(sensibilidade) + '\t' + str(especificidade) + '\t' + str(exatidao) + '\t' + '\n')
        print("Imagem segmentada!")
        print() #parágrafo
        
    ficheiro.close() #obrigatório
        
    return



if __name__ == '__main__':
	geral()
	print() #parágrafo
	print('Segmentação concluída!')