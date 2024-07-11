import re
from datetime import datetime, timedelta
import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import legacy as legacy_optimizers
import numpy as np
import random
import pickle
import json



# Diccionario de meses en español
meses_espanol = {
    1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril', 5: 'mayo', 6: 'junio',
    7: 'julio', 8: 'agosto', 9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
}

stemmer = LancasterStemmer()

# Cargar datos y modelo del chatbot
with open('/content/intents.json ') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    # Preprocesamiento
    words, labels, docs_x, docs_y = [], [], [], []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training, output = [], []
    out_empty = [0] * len(labels)

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            bag.append(1) if w in wrds else bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Construir el modelo con Keras
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))

# Usar el optimizador SGD legacy
sgd_legacy = legacy_optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd_legacy, metrics=['accuracy'])

try:
    model.load_weights("model.h5")
except:
    model.fit(training, output, epochs=200, batch_size=5, verbose=1)
    model.save("model.h5")

def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def chatbot_response(msg):
    results = model.predict(np.array([bag_of_words(msg, words)]))[0]
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in data['intents']:
        if tg['tag'] == tag:
            responses = tg['responses']
    return random.choice(responses)

# Funciones específicas del asistente universitario
df_notas = pd.DataFrame(columns=['Materia', 'Porcentaje_evaluacion', 'Notas', 'Tipo'])
df_eventos = pd.DataFrame(columns=['Descripcion', 'Fecha'])

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Inicialización de variables
df_eventos = None

# Función para formatear la fecha con nombre de mes en español
def formatear_fecha(fecha):
    nombre_mes = meses_espanol[fecha.strftime('%B')]
    return f"{fecha.day} de {nombre_mes}"

def programar_evento(comando):
    global df_eventos  # Declarar df_eventos como global para poder modificarla
    
    keywords = {
        'tarea': ['tarea', 'trabajo'],
        'parcial': ['parcial', 'semestral'],
        'exposicion': ['exposición', 'exposicion'],
        'investigacion': ['investigación', 'investigacion'],
        'quiz': ['quiz'],
        'taller': ['taller']
    }

    tipo_evento = None
    materia = None
    fecha = None

    # Buscar tipo de evento y normalizar palabras
    for tipo, palabras_clave in keywords.items():
        for palabra in palabras_clave:
            if palabra in comando.lower():
                tipo_evento = tipo.capitalize()
                comando = comando.replace(palabra, tipo, 1)  # Reemplazar palabra clave por tipo de evento
                break
        if tipo_evento:
            break

    if not tipo_evento:
        return "Lo siento, no entendí el tipo de evento que deseas programar."

    # Buscar la fecha en el comando
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }

    fecha_encontrada = False
    for mes, numero_mes in meses.items():
        if mes in comando.lower():
            dia = re.search(r'\b\d{1,2}\b', comando)
            if dia:
                fecha = datetime(datetime.now().year, numero_mes, int(dia.group()))
                fecha_encontrada = True
            break

    if not fecha_encontrada:
        return "No encontré la fecha en tu comando. Por favor, especifícala claramente."

    # Buscar la materia (opcional)
    match = re.search(r'de\s+(\w+(?:\s+\w+)*)\s+el', comando.lower())
    if match:
        materia = match.group(1).capitalize()

    if not materia:
        return f"¿Podrías indicarme la materia para el evento de tipo {tipo_evento} el {fecha.day} de {meses_espanol[fecha.month]}?"

    # Programar el evento en el calendario o base de datos
    nuevo_evento = pd.DataFrame({'Tipo': [tipo_evento], 'Materia': [materia], 'Fecha': [fecha]})
    if df_eventos is None:
        df_eventos = nuevo_evento  # Si df_eventos es None, inicialízalo con el nuevo evento
    else:
        df_eventos = pd.concat([df_eventos, nuevo_evento], ignore_index=True)

    # Programar sesiones de estudio si es una tarea
    if tipo_evento == 'Tarea':
        # No programar sesiones de estudio muy cerca de la fecha de entrega
        if (fecha - datetime.now()).days > 2:
            sesion_1 = fecha - timedelta(days=2)
            sesion_2 = fecha - timedelta(days=1)
            print(f"Se han programado sesiones de estudio para la tarea de {materia} el {fecha.day} de {meses_espanol[fecha.month]}:")
            print(f"- Sesión 1: {sesion_1.day} de {meses_espanol[sesion_1.month]}")
            print(f"- Sesión 2: {sesion_2.day} de {meses_espanol[sesion_2.month]}")

    # Programar sesiones de estudio si es un parcial, ejercicio o semestral
    elif tipo_evento in ['Parcial', 'Ejercicio', 'Semestral']:
        sesion_1 = fecha - timedelta(days=7)
        sesion_2 = fecha - timedelta(days=3)
        print(f"Se han programado sesiones de estudio para el {tipo_evento.lower()} de {materia} el {fecha.day} de {meses_espanol[fecha.month]}:")
        print(f"- Sesión 1: {sesion_1.day} de {meses_espanol[sesion_1.month]}")
        print(f"- Sesión 2: {sesion_2.day} de {meses_espanol[sesion_2.month]}")

        agregar_sesiones = input("¿Deseas agregar más sesiones de estudio para este evento? (Sí/No): ")
        if agregar_sesiones.lower() == 'sí' or agregar_sesiones.lower() == 'si':
            cantidad_sesiones = int(input("¿Cuántas sesiones adicionales deseas programar?: "))
            for i in range(cantidad_sesiones):
                sesion = fecha - timedelta(days=3 - i*2)
                print(f"- Sesión {i + 3}: {sesion.day} de {meses_espanol[sesion.month]}")
        else:
            print("Entendido, no se agregarán más sesiones de estudio.")

    return f"Evento de tipo {tipo_evento} de {materia} programado el {fecha.day} de {meses_espanol[fecha.month]}."

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Función para consultar eventos programados
def consultar_eventos(comando):
    fechas = re.findall(r'\b\d{1,2}\b', comando)
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }

    if fechas:
        dia = int(fechas[0])
        mes = None
        for nombre_mes, numero_mes in meses.items():
            if nombre_mes in comando:
                mes = numero_mes
                break
        if not mes:
            return "No encontré el mes en tu comando. Por favor, especifícalo claramente."
        
        # Filtrar eventos por fecha específica
        eventos_fecha = df_eventos[(df_eventos['Fecha'].dt.day == dia) & (df_eventos['Fecha'].dt.month == mes)]
        
        if eventos_fecha.empty:
            return f"No tienes eventos programados para el {dia} de {nombre_mes.capitalize()}."
        else:
            return f"Para el {dia} de {nombre_mes.capitalize()} tienes los siguientes eventos programados:\n{eventos_fecha['Descripcion'].tolist()}."

    # Buscar eventos para esta semana
    if 'esta semana' in comando:
        # Obtener fecha de inicio y fin de la semana actual
        hoy = datetime.now()
        inicio_semana = hoy - timedelta(days=hoy.weekday())
        fin_semana = inicio_semana + timedelta(days=6)

        # Filtrar eventos por semana actual
        eventos_semana = df_eventos[(df_eventos['Fecha'] >= inicio_semana) & (df_eventos['Fecha'] <= fin_semana)]
        
        if eventos_semana.empty:
            return "No tienes eventos programados para esta semana."
        else:
            return f"Para esta semana tienes los siguientes eventos programados:\n{eventos_semana['Descripcion'].tolist()}."

    # Buscar eventos por materia específica
    if 'de ' in comando and ('tengo algo de ' in comando or 'hay algo de ' in comando):
        materia = re.search(r'de\s+(\w+(?:\s+\w+)*)\s+', comando.lower())
        if materia:
            materia = materia.group(1)
            eventos_materia = df_eventos[df_eventos['Materia'].str.lower() == materia.lower()]
            if eventos_materia.empty:
                return f"No tienes eventos programados para la materia {materia.capitalize()}."
            else:
                return f"Tienes los siguientes eventos programados para la materia {materia.capitalize()}:\n{eventos_materia['Descripcion'].tolist()}."

    return "No entendí la consulta. ¿Podrías intentarlo de nuevo?"

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def reprogramar_evento(comando):
    # Implementar lógica para reprogramar eventos aquí
    return "Función en desarrollo."
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def comando_desconocido():
    return "Lo siento, no entendí ese comando. ¿Podrías intentar otra vez?"
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def tutorial_ingreso():
    tutorial = """
    Para ingresar un evento, puedes decir cosas como:
    - Tengo tarea de EDO el 15 de julio.
    - Cambio de fecha de parcial de EDO del 15 de julio a el 20 de julio.
    - ¿Qué tengo para esta semana de EDO?

    Para ingresar tus notas, puedes decir algo como:
    - Saqué 85 en el parcial 1 de EDO.
    - Modifica los porcentajes de evaluación para EDO.

    ¿Cómo más puedo ayudarte?
    """
    return tutorial
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Interacción con el chatbot
def interactuar_con_chatbot():
    global df_eventos
    print("Bienvenido al Asistente de Gestión Universitaria. ¿En qué puedo ayudarte hoy?")
    while True:
        entrada = input("Ingresa un comando (o 'salir' para terminar): ").lower()

        if entrada == 'salir':
            print("Gracias por usar el Asistente de Gestión Universitaria. ¡Hasta luego!")
            break

        if 'tengo algo para' in entrada or 'que tengo' in entrada or 'hay algo' in entrada:
            respuesta = consultar_eventos(entrada)
        elif 'cambio de fecha' in entrada or 'reprograma' in entrada or 'elimina' in entrada or 'quita' in entrada:
            respuesta = reprogramar_evento(entrada)
        elif any(keyword in entrada for keyword in ['tarea', 'parcial', 'exposicion', 'investigacion', 'quiz', 'taller']):
            respuesta = programar_evento(entrada)
        elif 'tutorial' in entrada or 'explícame' in entrada or 'cómo ingreso' in entrada:
            respuesta = tutorial_ingreso()
        else:
            respuesta = comando_desconocido()

        print("Chatbot:", respuesta)

interactuar_con_chatbot()