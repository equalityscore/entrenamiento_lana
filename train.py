import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Leemos el archivo CSV y dividimos los datos en conjuntos de entrenamiento y evaluación
df = pd.read_csv("machismo_data.csv")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Procesamos las frases para convertirlas en vectores de características utilizando CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Entrenamos un modelo de regresión logística utilizando el conjunto de entrenamiento
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluamos el rendimiento del modelo utilizando el conjunto de evaluación
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Guardamos el modelo entrenado para su uso posterior
with open("machismo_detector.pkl", "wb") as f:
    pickle.dump(model, f)

