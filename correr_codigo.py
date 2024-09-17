from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
model = YOLO('best.pt')

# Mostrar la información del modelo
model.info()

# Imprimir las clases del modelo
print("Clases del modelo:", model.names)

# ----- Prueba 1: Imagen Estática -----
# Cargar una imagen con patente para probar el modelo
img = cv2.imread("20200712_105815.jpg")
img = cv2.imread("400.jpg")# Asegúrate de que la ruta sea correcta
if img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta.")
else:
    # Redimensionar la imagen a 640x640 (si es necesario)
    img_resized = cv2.resize(img, (640, 640))

    # Realizar detección sobre la imagen
    results = model(img_resized, conf=0.1)  # Umbral de confianza bajo para capturar más detecciones
    results[0].plot()  # Mostrar resultados visualmente
    
    # Mostrar detalles del resultado en consola
    print("Resultados detección imagen estática:")
    
    # Verifica si hay detecciones
    if len(results[0].boxes) > 0:
        # Itera sobre las detecciones
        for box in results[0].boxes:
            # Imprime la información de la caja y la clase detectada
            print(f"Coordenadas de la caja: {box.xyxy}, Confianza: {box.conf}, Clase: {model.names[int(box.cls)]}")
    else:
        print("No se detectaron objetos.")


# ----- Prueba 3: Ajuste de Confianza -----
img = cv2.imread("20200712_105815.jpg")
img = cv2.imread("400.jpg")
if img is not None:
    # Probar con diferentes niveles de confianza
    for conf in [0.5, 0.25, 0.1]:
        print(f"Prueba con confianza {conf}:")
        results = model(img, conf=conf)
        results[0].plot()
        
        # Mostrar detalles del resultado en consola
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                print(f"Coordenadas de la caja: {box.xyxy}, Confianza: {box.conf}, Clase: {model.names[int(box.cls)]}")
        else:
            print("No se detectaron objetos con este nivel de confianza.")
else:
    print("Error: No se pudo cargar la imagen para la prueba de confianza.")
