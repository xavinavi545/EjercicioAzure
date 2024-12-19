import cv2
import requests
import time
import json
import glob
import sys
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

# Configuración de la API de Azure
ruta = 'C:\\Users\\Xavi\\Desktop\\TallerDev'

# Cargar credenciales desde key.json
credenciales = json.load(open("C:\\Users\\Xavi\\Desktop\\TallerDev\\key.json"))
CLAVE = credenciales['KEY']
ENDPOINT = credenciales['ENDPOINT']

# URL de la API de detección facial
url_api_face = f"{ENDPOINT}/face/v1.0/detect"
cabeceras = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': CLAVE
}
parametros = {
    'detectionModel': 'detection_01',
    'returnFaceId': 'true',
    'returnFaceRectangle': 'true'  # Eliminamos atributos no soportados
}

# Listas globales
GRUPOS = []
PERSONAS = []
ID = []

# Cliente de Azure Face
cliente_face = FaceClient(ENDPOINT, CognitiveServicesCredentials(CLAVE))


# Funciones principales
def crear_grupo(nombre_grupo):
    nombre_grupo = nombre_grupo.lower().replace(" ", "_")  # Forzar formato válido
    try:
        cliente_face.person_group.create(person_group_id=nombre_grupo, name=nombre_grupo)
        print(f"Se creó el grupo: {nombre_grupo}")
    except Exception as e:
        print(f"Error al crear el grupo '{nombre_grupo}': {e}")
        sys.exit()


def crear_persona(nombre_persona, grupo):
    try:
        persona = cliente_face.person_group_person.create(grupo, nombre_persona)
        print(f'ID de la persona {nombre_persona}: {persona.person_id}')
        ID.append(persona.person_id)

        # Buscar imágenes que empiecen con el nombre de la persona
        fotos_persona = [archivo for archivo in glob.glob('*.jpg') if archivo.startswith(nombre_persona)]
        for imagen in fotos_persona:
            with open(imagen, 'rb') as archivo_imagen:
                cliente_face.person_group_person.add_face_from_stream(grupo, persona.person_id, archivo_imagen)
                print(f'Foto agregada: {imagen}')
                time.sleep(1)
    except Exception as e:
        print(f"Error al agregar la persona '{nombre_persona}': {e}")


def entrenar_grupo(grupo):
    try:
        print(f'Iniciando entrenamiento del grupo: {grupo}')
        cliente_face.person_group.train(grupo)
        while True:
            estado = cliente_face.person_group.get_training_status(grupo)
            print(f"Estado del entrenamiento: {estado.status}")
            if estado.status == 'succeeded':
                break
            elif estado.status == 'failed':
                cliente_face.person_group.delete(person_group_id=grupo)
                sys.exit('El entrenamiento falló.')
            time.sleep(3)
    except Exception as e:
        print(f"Error durante el entrenamiento del grupo '{grupo}': {e}")


def iniciar_reconocimiento():
    # Configurar grupo y personas
    grupo = input('Escribe el nombre del grupo: ').lower()
    crear_grupo(grupo)

    nombre = None
    while nombre != 'fin':
        nombre = input(f"Escribe el nombre de una persona (o 'fin' para terminar): ").lower()
        if nombre != 'fin':
            PERSONAS.append(nombre)
            crear_persona(nombre, grupo)

    entrenar_grupo(grupo)

    # Iniciar la cámara
    camara = cv2.VideoCapture(0)
    while True:
        ret, frame = camara.read()
        if not ret:
            print("Error al acceder a la cámara.")
            break

        # Convertir imagen a bytes
        imagen = cv2.imencode('.jpg', frame)[1].tobytes()

        try:
            # Llamada a la API de detección facial
            respuesta = requests.post(url_api_face, params=parametros, headers=cabeceras, data=imagen)
            rostros = respuesta.json()

            for rostro in rostros:
                rect = rostro['faceRectangle']
                x, y, w, h = rect['left'], rect['top'], rect['width'], rect['height']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Reconocimiento Facial', frame)

        except Exception as e:
            print(f"Error en la detección: {e}")

        if cv2.waitKey(1) & 0xFF == 27:  # Tecla ESC para salir
            print("Saliendo del programa...")
            break

    camara.release()
    cv2.destroyAllWindows()


def finalizar_reconocimiento(grupo):
    try:
        cliente_face.person_group.delete(person_group_id=grupo)
        print(f"El grupo '{grupo}' ha sido eliminado correctamente.")
    except Exception as e:
        print(f"Error al eliminar el grupo '{grupo}': {e}")


# Ejecución del programa
if __name__ == "__main__":
    iniciar_reconocimiento()
    finalizar_reconocimiento(GRUPOS[0])
