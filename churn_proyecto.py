import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# Cargar el dataset
dataset = pd.read_csv("churn_dataset.csv")

# Configuración de los estilos de los gráficos
sns.set(style="whitegrid")

# Crear una figura con subgráficos para los histogramas
fig, axs = plt.subplots(2, 2, figsize=(8, 6))

# Histograma de Edad
sns.histplot(dataset['Age'], bins=20, kde=True, ax=axs[0, 0], color=sns.color_palette("viridis")[0])
axs[0, 0].set_title('Histograma de Edad')
axs[0, 0].set_xlabel('Edad')
axs[0, 0].set_ylabel('Frecuencia')

# Histograma de Gasto Mensual
sns.histplot(dataset['MonthlySpend'], bins=20, kde=True, ax=axs[0, 1], color=sns.color_palette("viridis")[1])
axs[0, 1].set_title('Histograma de Gasto Mensual')
axs[0, 1].set_xlabel('Gasto Mensual')
axs[0, 1].set_ylabel('Frecuencia')

# Histograma de Antigüedad
sns.histplot(dataset['Tenure'], bins=20, kde=True, ax=axs[1, 0], color=sns.color_palette("viridis")[2])
axs[1, 0].set_title('Histograma de Antigüedad')
axs[1, 0].set_xlabel('Antigüedad (meses)')
axs[1, 0].set_ylabel('Frecuencia')

# Histograma de Satisfacción del Cliente
sns.histplot(dataset['CustomerSatisfactionScore'], bins=10, kde=True, ax=axs[1, 1], color=sns.color_palette("viridis")[3])
axs[1, 1].set_title('Histograma de Satisfacción del Cliente')
axs[1, 1].set_xlabel('Puntuación de Satisfacción')
axs[1, 1].set_ylabel('Frecuencia')

# Ajustar el layout
plt.tight_layout()
plt.show()

# Gráfico de torta para la variable Churn
plt.figure(figsize=(8, 6))
churn_counts = dataset['Churn'].value_counts()
plt.pie(churn_counts, labels=['No Churn (0)', 'Churn (1)'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", 2))
plt.title('Distribución de Churn')
plt.axis('equal')  # Para que el gráfico sea un círculo
plt.show()



#----------ANALISIS BIVARIADO: BOXPLOT ------------#


# Crear categorías para Gasto Mensual
dataset['SpendCategory'] = pd.cut(dataset['MonthlySpend'], bins=[0, 50, 100, 150, 200, 250], 
                                    labels=['Bajo', 'Medio-Bajo', 'Medio', 'Medio-Alto', 'Alto'])

# Crear categorías para Edad
dataset['AgeCategory'] = pd.cut(dataset['Age'], bins=[18, 25, 35, 45, 55, 65], 
                                  labels=['18-24', '25-34', '35-44', '45-54', '55-64'])

# Crear categorías para Antigüedad
dataset['TenureCategory'] = pd.cut(dataset['Tenure'], bins=[0, 6, 12, 18, 24, 36], 
                                     labels=['0-6 meses', '6-12 meses', '12-18 meses', '18-24 meses', '24-36 meses'])

# Configurar la figura con subgráficos
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Boxplot para Gasto Mensual vs Satisfacción del Cliente
sns.boxplot(x='SpendCategory', y='CustomerSatisfactionScore', data=dataset, palette='viridis', ax=axs[0])
axs[0].set_title('Satisfacción del Cliente por Gasto Mensual')
axs[0].set_xlabel('Categoría de Gasto Mensual')
axs[0].set_ylabel('Satisfacción del Cliente')

# Boxplot para Edad vs Gasto Mensual
sns.boxplot(x='AgeCategory', y='MonthlySpend', data=dataset, palette='viridis', ax=axs[1])
axs[1].set_title('Gasto Mensual por Edad')
axs[1].set_xlabel('Categoría de Edad')
axs[1].set_ylabel('Gasto Mensual')

# Boxplot para Antigüedad vs Gasto Mensual
sns.boxplot(x='TenureCategory', y='MonthlySpend', data=dataset, palette='viridis', ax=axs[2])
axs[2].set_title('Gasto Mensual por Antigüedad')
axs[2].set_xlabel('Categoría de Antigüedad')
axs[2].set_ylabel('Gasto Mensual')

# Ajustar la disposición
plt.tight_layout()
plt.show()


#-------Análisis de segmentos con el modelo K-means------------#

# Seleccionar características para K-Means
X = dataset[['MonthlySpend', 'Tenure', 'CustomerSatisfactionScore']]

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinar el número óptimo de clústeres
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Gráfico del método del codo
plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Método del codo')
plt.xlabel('Número de clústeres')
plt.ylabel('WCSS')
plt.show()

# Aplicar K-Means con el número óptimo de clústeres (por ejemplo, 3)
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
dataset['Cluster'] = kmeans.fit_predict(X_scaled)

# Seleccionar una muestra del 10% de los datos para graficar
sample_data = dataset.sample(frac=0.1, random_state=42)

# Visualización de los clústeres con la muestra reducida
plt.figure(figsize=(6, 5))
sns.scatterplot(data=sample_data, x='MonthlySpend', y='CustomerSatisfactionScore', hue='Cluster', palette='viridis')
plt.title('Clusters de Clientes (Muestra Reducida)')
plt.show()



#-----------Prediccion con Regresion Logistica---------#


# Supongamos que el dataset ya está cargado en la variable "dataset"
dataset= pd.read_csv("churn_dataset.csv")

# Crear categorías para Gasto Mensual, Edad y Antigüedad
dataset['SpendCategory'] = pd.cut(dataset['MonthlySpend'], bins=[0, 50, 100, 150, 200, 250], labels=['Bajo', 'Medio-Bajo', 'Medio', 'Medio-Alto', 'Alto'])
dataset['AgeCategory'] = pd.cut(dataset['Age'], bins=[18, 25, 35, 45, 55, 65], labels=['18-24', '25-34', '35-44', '45-54', '55-64'])
dataset['TenureCategory'] = pd.cut(dataset['Tenure'], bins=[0, 6, 12, 18, 24, 36], labels=['0-6 meses', '6-12 meses', '12-18 meses', '18-24 meses', '24-36 meses'])

# Convertir categorías en variables dummy
dataset = pd.get_dummies(dataset, columns=['SpendCategory', 'AgeCategory', 'TenureCategory'], drop_first=True)

# Preparar conjuntos de datos para el modelo
X = dataset.drop(['Churn', 'CustomerID', 'CustomerSatisfactionScore'], axis=1)
y = dataset['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos y entrenar el modelo de regresión logística
scaler = StandardScaler()
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
model = LogisticRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Resultados del modelo
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Graficar la matriz de confusión
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='viridis', cbar=False,
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.show()

# Graficar la función sigmoide de probabilidad de churn respecto a Gasto Mensual
spend_range = np.linspace(0, 250, 300).reshape(-1, 1)
spend_range_scaled = scaler.transform(np.concatenate([spend_range, np.zeros((spend_range.shape[0], X.shape[1] - 1))], axis=1))
plt.plot(spend_range, model.predict_proba(spend_range_scaled)[:, 1], color='blue', label='Probabilidad de Churn')
plt.axhline(0.5, color='red', linestyle='--', label='Umbral de Churn (0.5)')
plt.title('Función Sigmoide de la Probabilidad de Churn')
plt.xlabel('Gasto Mensual')
plt.ylabel('Probabilidad de Churn')
plt.legend()
plt.show()

# Función de predicción de churn para un nuevo cliente
def predict_churn(new_customer):
    new_customer_df = pd.DataFrame([new_customer]).reindex(columns=X.columns, fill_value=0)
    return "Sí" if model.predict(scaler.transform(new_customer_df))[0] == 1 else "No"

# Ejemplo de uso
nuevo_cliente = {'Age': 30,
                  'MonthlySpend': 120,
                  'Tenure': 12,
                  'DiscountsReceived': 1,
                  'SpendCategory': 'Medio',
                  'AgeCategory': '25-34',
                  'TenureCategory': '6-12 meses'}

print(f"¿El nuevo cliente abandonará? {predict_churn(nuevo_cliente)}")
