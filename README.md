## **Proyecto de Optimización de Estrategias de Fidelización con Machine Learning**

En este proyecto, el objetivo fue desarrollar un modelo predictivo que ayudara a identificar y 
comprender mejor a los clientes con alto riesgo de deserción (churn) en una empresa, con la intención 
de diseñar estrategias de fidelización efectivas y optimizar la retención. Aprovechando un conjunto de 
datos ficticio que simula comportamientos y características relevantes de clientes, se buscó descubrir 
patrones que servirían para proponer soluciones al churn. 

Para alcanzar estos objetivos, se emplearon técnicas avanzadas de Machine Learning, como el análisis 
de segmentación con K-Means y la regresión logística para predicciones de churn. La metodología 
también incluyó un análisis exploratorio detallado (EDA) para visualizar la relación entre distintas 
variables clave y la tasa de deserción, apoyándose en librerías como pandas, scikit-learn, matplotlib y 
seaborn para el procesamiento de datos, modelado y visualización de resultados. 

Este proyecto refleja mis habilidades en el análisis de datos y el diseño de modelos predictivos, así 
como mi capacidad para aplicar conocimientos de Machine Learning en la solución de problemas de 
negocios reales. La combinación de análisis exploratorio, segmentación y modelado predictivo permite 
crear un enfoque integral, aplicable en el desarrollo de estrategias de retención personalizadas y en la 
toma de decisiones informadas basadas en datos. 



## **Contaré como trabaje el Proyecto paso a paso** 

### **Paso 1: Exploración Inicial de Datos**
Se inició el análisis con una muestra de las primeras cinco filas del dataset mediante sample(5) y una revisión rápida de la estructura 
y el tamaño con shape e info. Esto ayudó a conocer el tipo de datos y detectar valores faltantes o inconsistencias iniciales. 

### **Paso 2: Análisis Univariado** 
Realicé un análisis univariado para comprender la distribución de las variables clave: Edad, Gasto Mensual, Antigüedad, y 
Satisfacción del Cliente. Se usaron histogramas para cada variable y un gráfico de torta para la variable Churn, lo que brindó un 
panorama general de las características de los clientes. 

### **Paso 3: Análisis Bivariado**
El análisis bivariado exploró relaciones entre pares de variables usando boxplots, incluyendo: Satisfacción del Cliente por Gasto 
Mensual, Gasto Mensual por Edad, y Gasto Mensual por Antigüedad. 

### **Paso 4: Preprocesamiento de Datos y Método del Codo**
Los datos se estandarizaron mediante StandardScaler para asegurar uniformidad en la escala de las variables. Luego, utilicé el 
método del codo para determinar el número óptimo de clústeres, evaluando la inercia para elegir el valor que mejor segmentara a 
los clientes. 

### **Paso 5: Segmentación de Clientes con K-Means**
Con el número óptimo de clústeres determinado, segmenté los datos en tres grupos usando K-Means. Esta segmentación permitió 
identificar perfiles de clientes con características similares, esenciales para diseñar estrategias de retención específicas. 

### **Paso 6: Preprocesamiento Adicional para Regresión Logística**
Preprocesé los datos para la etapa de modelado predictivo, con el fin de preparar las variables adecuadamente para la regresión 
logística. Esta preparación aseguraba la consistencia en las predicciones de probabilidad de churn. 

### **Paso 7: Modelo de Regresión Logística y Evaluación**
Entrené una regresión logística y validé el modelo mediante una matriz de confusión, evaluando la precisión en la predicción de 
churn. Este paso confirmó la efectividad del modelo en diferenciar entre clientes propensos al abandono. 

### **Paso 8: Predicción de Churn para Nuevos Clientes**
Finalmente, se probó el modelo al ingresar los datos de un cliente nuevo, prediciendo si abandonaría. Esta predicción sirvió para 
demostrar la aplicabilidad del modelo en escenarios prácticos de retención de clientes.
