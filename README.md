# entrenamiento_lana

    Necesitamos obtener un conjunto de datos de entrenamiento que contenga frases de ejemplo con etiquetas que indiquen si son machistas o no. Un formato común para almacenar estos datos es un archivo CSV con dos columnas: una para la frase y otra para la etiqueta (por ejemplo, "1" para machista y "0" para no machista).

    Usaremos pandas para leer el archivo CSV y scikit-learn para entrenar un modelo de clasificación.

    Leemos el archivo CSV utilizando pandas y dividimos los datos en dos conjuntos: uno para entrenar el modelo y otro para evaluar su rendimiento.

    Procesamos las frases para convertirlas en vectores de características que el modelo de aprendizaje automático pueda utilizarusando el método CountVectorizer de scikit-learn, que convierte las frases en matrices de contabilización de términos.

    Entrenamos un modelo de clasificación utilizando el conjunto de entrenamiento.

    Una vez que el modelo ha sido entrenado, podemos utilizar el conjunto de evaluación para evaluar su rendimiento y calcular medidas como la precisión y el recall.

    Finalmente hay que guardar el modelo entrenado para su uso posterior en la detección de machismos en nuevas frases.


## Instalación del proyecto
    
    En un entorno virtual de python
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```

    Instalamos las librerías, no se incluye requirements.txt en el repositorio porque son librerías que evolucionan muy rápido y conviene tener la última
    ```
    pip install pandas
    pip install scikit-learn
    pip install flask
    ```

    Guardar el csv en la raiz del proyecto como `machismo_data.csv` que debe tener dos columnas: una para la frase y otra para la etiqueta (por ejemplo, "1" para machista y "0" para no machista). Luego, divide los datos en un conjunto de entrenamiento y un conjunto de evaluación, procesa las frases para convertirlas en vectores de características, entrena un modelo de regresión logística y evalúa su rendimiento utilizando el conjunto de evaluación. Finalmente, guarda el modelo entrenado para su uso posterior.

    Ejecutar:

    `python train.py`


  ## Uso del modelo entrenado

Una vez el modelo está entrenado y guardado con pickle, puedes cargarlo usando.

`python predict.py`

Esto lanzara un servidor flask que usa el endpoint `{{url}}/detect_machismo` con el verbo POST en el cual irá incluida la frase a analizar.

Retornando por ejemplo:
```
{"machista": 0.7}
```
