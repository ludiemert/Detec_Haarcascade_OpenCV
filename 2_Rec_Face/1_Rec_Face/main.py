# pip install dlib opencv-python


import cv2
import dlib

# Carrega o detector de rosto do dlib
detector = dlib.get_frontal_face_detector()

# Lê a imagem (coloque a imagem no mesmo diretório ou especifique o caminho completo)
imagem = cv2.imread("sua_imagem.jpg")

# Converte a imagem para escala de cinza (melhor para detecção)
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detecta rostos
rostos = detector(imagem_cinza)

# Desenha retângulos ao redor dos rostos detectados
for rosto in rostos:
    x, y, l, a = rosto.left(), rosto.top(), rosto.width(), rosto.height()
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)

# Exibe a imagem com as detecções
cv2.imshow("Rostos Detectados", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
