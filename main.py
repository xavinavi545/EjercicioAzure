import os
import cv2
import requests
import json

# Configuración de la API de Azure
ruta = 'C:\\Users\\Xavi\\Desktop\\TallerDev'

# Cargar credenciales desde key.json
try:
    credenciales = json.load(open(os.path.join(ruta, "key.json")))
    CLAVE = credenciales['KEY']
    ENDPOINT = credenciales['ENDPOINT'].rstrip('/')  
except FileNotFoundError:
    print("Error: El archivo key.json no se encontró. Verifica la ruta.")
    exit()

# URL de la API de detección facial
url_api_face = f"{ENDPOINT}/face/v1.0/detect"

print("URL construida correctamente:", url_api_face)

cabeceras = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': CLAVE
}

parametros = {
    'detectionModel': 'detection_01',
    'returnFaceId': 'true',
    'returnFaceRectangle': 'true'
}

# Código para activar la cámara
def activar_camara():
    camara = cv2.VideoCapture(0)
    print("Cámara activada. Presiona ESC para salir.")

    while True:
        ret, frame = camara.read()
        if not ret:
            print("Error al acceder a la cámara.")
            break

        # Convertir la imagen a bytes
        imagen = cv2.imencode('.jpg', frame)[1].tobytes()

        try:
            # Llamada a la API
            respuesta = requests.post(url_api_face, params=parametros, headers=cabeceras, data=imagen)
            respuesta.raise_for_status()
            rostros = respuesta.json()

            # Dibujar rectángulos en los rostros detectados
            for rostro in rostros:
                rect = rostro['faceRectangle']
                x, y, w, h = rect['left'], rect['top'], rect['width'], rect['height']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Rostro Detectado", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except requests.exceptions.HTTPError as error:
            print("Error en la detección:", error)

        # Mostrar la imagen
        cv2.imshow('Reconocimiento Facial', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Tecla ESC para salir
            print("Saliendo del programa...")
            break

    camara.release()
    cv2.destroyAllWindows()

# Ejecución principal
if __name__ == "__main__":
    activar_camara()
