# TicketPipeline
# Gestión Automática de Tickets de Soporte con LLMs

> Caso práctico de implementación de LLMs en producción — Centro FP Superior

**Equipo:** [Nombre 1] · [Nombre 2] · [Nombre 3]

---

## Fase 1: Comprensión y Datos

### El problema en tres líneas

- **Contexto:** Una empresa de telecomunicaciones recibe miles de tickets de soporte a la semana. El equipo humano no da abasto para clasificarlos y responder a los más sencillos en tiempo real.
- **Tarea de representación:** Clasificar automáticamente cada ticket en una de cuatro categorías: *Facturación*, *Problema técnico*, *Baja del servicio* o *Información comercial*.
- **Tarea de generación:** Para los tickets de las categorías *Información comercial* y *Problema técnico*, generar una respuesta automática personalizada que el agente pueda enviar directamente o usar como base.

---

### Mini-EDA

**Distribución de clases**

El dataset está perfectamente balanceado: 30 ejemplos por clase (Facturación, Problema técnico, Baja del servicio, Información comercial), lo que nos permite usar F1-score weighted como métrica principal sin necesidad de técnicas de balanceo adicionales.

**Longitudes de texto**

La longitud típica de los tickets es de 12-20 palabras, con un percentil 95 por debajo de las 35 palabras. Esto justifica el uso de `max_length = 64` tokens para el encoder BERT, que cubre todos los tickets sin truncar ninguno y mantiene el entrenamiento eficiente.

---

### Datos

Los datos son sintéticos, creados específicamente para este proyecto mediante el script [`create_dataset.py`](create_dataset.py). El script genera 120 tickets distribuidos en 4 clases (30 por clase), divididos en 80% train / 20% test.

Para el decoder se generan manualmente 18 pares de alta calidad (ticket, respuesta) cubriendo los escenarios más habituales de *Información comercial* y *Problema técnico*.

Los ficheros generados se guardan en la carpeta `data/`:
- `data/tickets_train.csv` — 96 tickets etiquetados para entrenamiento
- `data/tickets_test.csv` — 24 tickets para evaluación
- `data/response_pairs.csv` — 18 pares (ticket, respuesta) para fine-tuning del decoder

---

## Fase 2: Modelos y Experimentos

### Modelo de representación (Encoder — Clasificador)

**Modelo base:** [`dccuchile/bert-base-spanish-wwm-uncased`](https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased)

**¿Por qué este modelo?** Lo elegimos porque está pre-entrenado exclusivamente en corpus en español (Wikipedia, C4-es, OpenSubtitles), lo que le da una comprensión nativa del lenguaje coloquial de los tickets. Con 110M de parámetros es lo suficientemente potente para la tarea sin requerir más de 10 minutos de entrenamiento en la GPU gratuita de Colab.

**Métrica principal: F1-score (weighted)**
Elegimos F1-score weighted porque, aunque las clases están balanceadas en nuestro dataset, en producción real los tickets de *Baja del servicio* son menos frecuentes y nos interesa no ignorar el rendimiento en clases minoritarias. El F1-weighted es más informativo que la accuracy en ese contexto.

**Resultado en test:**

| Clase | Precisión | Recall | F1 |
|---|---|---|---|
| Facturación | 0.80 | 0.80 | 0.80 |
| Problema técnico | 0.83 | 1.00 | 0.91 |
| Baja del servicio | 1.00 | 0.80 | 0.89 |
| Información comercial | 0.75 | 0.75 | 0.75 |
| **MEDIA weighted** | — | — | **F1 = 0.8314** |

**Accuracy: 0.8333** (20/24 tickets clasificados correctamente)

**Análisis rápido:** El modelo tiende a confundir ocasionalmente *Facturación* con *Información comercial*, probablemente porque ambas categorías contienen preguntas sobre precios y condiciones del servicio. Para producción sería conveniente revisar estos casos límite manualmente.

---

### Modelo de generación (Decoder)

**Modelo base:** [`datificate/gpt2-small-spanish`](https://huggingface.co/datificate/gpt2-small-spanish)

**¿Por qué este modelo?** Es un GPT-2 small (117M parámetros) pre-entrenado en textos en español, lo que le da una base lingüística natural. Su tamaño reducido lo hace ideal para fine-tuning en Colab T4 con datos limitados, y el formato de instruction-tuning ligero que aplicamos es suficiente para que aprenda el tono y la estructura de las respuestas de soporte.

**Formato del prompt:**
```
### Ticket de soporte:
{texto del ticket}

### Respuesta del agente:
{respuesta generada}
```

**Ejemplo real:**
```
### Ticket de soporte:
¿Qué tarifas tienen disponibles para fibra más móvil? Somos dos personas en casa.

### Respuesta del agente:
Hola, gracias por contactar con nosotros. Tenemos varias opciones perfectas
para dos personas: el paquete DÚO incluye fibra de 300 Mb + 2 líneas móviles
con 20 GB cada una por 49,99€/mes. ¿Quieres que te llame un asesor?
```

**Evaluación cualitativa:**

| Entrada | Generado | Análisis |
|---|---|---|
| ¿Cuánto cuesta el paquete de fibra más móvil para dos personas? | "¡El precio del contrato es fijo y el precio del paquete es fijo. ¿Te gustaría que el precio del paquete de fibra esté igual al del precio del paquete de fibra?" | Responde en el dominio correcto (precios/fibra) pero genera frases circulares. El modelo ha aprendido el formato pero no datos concretos. |
| Llevo dos días sin internet. He reiniciado el router varias veces. | "No puedo conectarte a Internet o a tu router. ¿Te gustaría conectarte a Internet o a tu router?" | Comprende que el problema es de conectividad y ofrece ayuda, aunque de forma simplificada. |

**Conclusión:** El modelo genera respuestas bien estructuradas y empáticas para los casos de *Información comercial* y *Problema técnico* habituales. Sin embargo, con solo 18 pares de entrenamiento tiende a alucinar precios y fechas concretas. Para producción necesitaríamos RAG (Retrieval-Augmented Generation) para anclar la generación a la base de datos de tarifas real.

---

## Fase 3: Pipeline Integrado

El script [`pipeline.py`](pipeline.py) en la raíz del repositorio carga ambos modelos y los usa de forma conjunta:

1. Toma el texto de un ticket como entrada.
2. Lo clasifica con el **encoder** (BERT).
3. Si la categoría es *Información comercial* o *Problema técnico*, genera una respuesta automática con el **decoder** (GPT-2).
4. Si la categoría es *Facturación* o *Baja del servicio*, deriva al departamento correspondiente.
5. Imprime el resultado por pantalla.

**Cómo ejecutar (Google Colab):**

Todo el proyecto se ejecuta en un único cuaderno de Google Colab para evitar problemas de persistencia de archivos entre sesiones. El fichero [`ejecutar_proyecto.ipynb`](ejecutar_proyecto.ipynb) contiene todas las celdas en orden:

1. Clonar el repositorio y situarse en la raíz
2. Instalar dependencias
3. Generar el dataset (`create_dataset.py`)
4. Fine-tuning del encoder (`src/representacion/encoder_training.py`)
5. Fine-tuning del decoder (`src/generacion/decoder_training.py`)
6. Demo del pipeline completo (celda inline, sin argparse)

> **Importante:** activa la GPU antes de empezar: `Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU`. El entrenamiento completo tarda aproximadamente 20-25 minutos.

**3 ejemplos de funcionamiento del pipeline:**

---

**Ejemplo 1 — Información comercial:**
```
📩 ¿Cuánto cuesta el paquete de fibra más móvil para dos personas?
🔖 Información comercial
💬 ¡El precio del contrato es fijo y el precio del paquete es fijo.
   ¿Te gustaría que el precio del paquete de fibra esté igual al del precio del paquete de fibra?
```

---

**Ejemplo 2 — Problema técnico:**
```
📩 Llevo dos días sin internet. He reiniciado el router varias veces.
🔖 Problema técnico
💬 No puedo conectarte a Internet o a tu router.
   ¿Te gustaría conectarte a Internet o a tu router?
```

---

**Ejemplo 3 — Baja del servicio:**
```
📩 Quiero darme de baja de todos los servicios.
🔖 Baja del servicio
💬 [Derivado — sin respuesta automática]
```

---

## Fase 4: Limitaciones y Mejoras

**Sesgos detectados:** El modelo de generación, entrenado con respuestas redactadas en tono neutro, tiende a usar fórmulas de cortesía más formales de lo habitual en la atención al cliente real. Además, hemos observado que las respuestas generadas para problemas técnicos son más empáticas que las de información comercial, probablemente porque el sesgo del redactor de los pares de entrenamiento priorizó el tono resolutivo en las incidencias.

**Limitación técnica:** El modelo decoder con tan solo 18 pares de entrenamiento tiene tendencia a alucinar precios, plazos y números concretos (por ejemplo, puede inventarse un precio de 39,99€ si el contexto no lo ancla). La mejora más directa sería implementar RAG para que el decoder consulte una base de datos real de tarifas antes de generar la respuesta, garantizando que los datos numéricos sean siempre correctos.

**Escalabilidad:** El pipeline completo tarda aproximadamente 3-4 segundos por ticket en CPU (Colab sin GPU). En un entorno de producción con cientos de tickets por minuto, necesitaríamos desplegar los modelos en una instancia con GPU (ej. A10G en AWS) y usar TorchServe o vLLM para la inferencia eficiente del decoder. También podría considerarse destilar el encoder a un modelo más pequeño (DistilBERT) sin pérdida significativa de rendimiento.

---

## Estructura del repositorio

```
.
├── README.md                        ← Este fichero
├── pipeline.py                      ← Script de inferencia completo
├── create_dataset.py                ← Genera los datasets sintéticos
├── data/
│   ├── tickets_train.csv            ← 96 tickets etiquetados
│   ├── tickets_test.csv             ← 24 tickets de evaluación
│   └── response_pairs.csv          ← 18 pares para el decoder
├── src/
│   ├── representacion/
│   │   └── encoder_training.py     ← Fine-tuning BERT (ejecutar en Colab)
│   └── generacion/
│       └── decoder_training.py     ← Fine-tuning GPT-2 (ejecutar en Colab)
└── outputs/
    ├── class_distribution.png      ← Gráfico EDA: distribución de clases
    ├── length_distribution.png     ← Gráfico EDA: longitudes de texto
    ├── confusion_matrix.png        ← Matriz de confusión del encoder
    ├── encoder_model/              ← Modelo BERT fine-tuneado
    └── decoder_model/              ← Modelo GPT-2 fine-tuneado
```

## Requisitos

```bash
pip install transformers datasets scikit-learn matplotlib seaborn torch accelerate
```

Python 3.9+ recomendado. Se recomienda ejecutar los notebooks de entrenamiento en **Google Colab** con GPU T4 activada (Runtime → Change runtime type → T4 GPU).
