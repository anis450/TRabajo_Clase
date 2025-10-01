# TRabajo_Clase
# Análisis de Datos de Diamantes con Regresión Lineal

##  Descripción
Este proyecto implementa un análisis de datos sobre un conjunto de diamantes con el objetivo de estudiar la relación entre el **peso en quilates (carat)** y el **precio**.  
Se utiliza **Python en Google Colab** junto con librerías de análisis de datos y machine learning como `pandas`, `matplotlib`, `scikit-learn` y `numpy`.

El enfoque principal consiste en aplicar **regresión lineal** para comprobar si existe una relación lineal entre estas dos variables y evaluar el desempeño del modelo.

---

##  Metodología

1. **Carga de Datos**  
   Se parte de un archivo `diamonds.csv` con información de diamantes.  
   Ejemplo de columnas:
   - `carat` → peso del diamante en quilates.
   - `price` → precio en dólares.

   ```python
   import pandas as pd

   data = pd.read_csv("diamonds.csv")
   print(data.head())
   ```

2. **Preprocesamiento**  
   - Selección de variables: `carat` y `price`.
   - Eliminación de valores nulos (si los hubiera).
   - División en datos de entrenamiento y prueba.

   ```python
   from sklearn.model_selection import train_test_split

   X = data[['carat']]
   y = data['price']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

3. **Modelo de Regresión Lineal**  
   ```python
   from sklearn.linear_model import LinearRegression

   model = LinearRegression()
   model.fit(X_train, y_train)
   ```

4. **Predicción y Evaluación**  
   - Predicciones sobre el conjunto de prueba.  
   - Cálculo de métricas: R², MSE.

   ```python
   from sklearn.metrics import mean_squared_error, r2_score

   y_pred = model.predict(X_test)
   print("R²:", r2_score(y_test, y_pred))
   print("MSE:", mean_squared_error(y_test, y_pred))
   ```

5. **Visualización**  
   Gráfico de dispersión y recta de regresión.

   ```python
   import matplotlib.pyplot as plt

   plt.scatter(X_test, y_test, color="black")
   plt.scatter(X_test, y_pred, color="red")
   plt.plot(X_test, y_pred, color="green")
   plt.xlabel("Carat")
   plt.ylabel("Price")
   plt.title("Regresión Lineal: Carat vs Price")
   plt.show()
   ```

---

##  Resultados y Discusión

La gráfica obtenida muestra lo siguiente:

- Existe una **relación positiva fuerte** entre el peso del diamante (carat) y su precio.  
- El modelo de regresión lineal logra capturar esta tendencia, ajustando una recta que representa adecuadamente el crecimiento del precio con respecto al tamaño del diamante.  
- Sin embargo, al observar los puntos dispersos (en negro) frente a las predicciones (en rojo), se aprecia que **no todos los precios siguen perfectamente una relación lineal**. Esto sugiere que otros factores (como corte, color o claridad) influyen en el precio.

 **Interpretación:**  
Aunque la regresión lineal simple es útil para entender la correlación entre carat y price, para lograr un modelo más preciso se debería aplicar una **regresión múltiple** que considere todas las variables del dataset.

---

##  Conclusiones

- La variable **carat** tiene una gran influencia en el precio de los diamantes.  
- La **regresión lineal simple** confirma esta correlación con un modelo interpretable.  
- Aun así, el precio de un diamante no depende solo del peso, sino también de características adicionales.  
- Este análisis inicial es un buen punto de partida, pero se recomienda profundizar con modelos multivariados y no lineales.

---

##  Bibliografía

- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O’Reilly Media.  
- Scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)  
- Pandas Documentation: [https://pandas.pydata.org/](https://pandas.pydata.org/)  
- Matplotlib Documentation: [https://matplotlib.org/](https://matplotlib.org/)
