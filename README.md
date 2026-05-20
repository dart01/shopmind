# 🛍️ ShopMind — Motor de Recomendaciones para E-commerce

ShopMind nació de una pregunta simple: ¿cómo sabe Amazon qué productos mostrarte 
justo cuando los necesitas? Este proyecto es mi intento de responder esa pregunta 
construyendo un motor de recomendaciones desde cero, con datos reales y dos 
enfoques completamente diferentes que se complementan entre sí.

---

## ¿Qué problema resuelve?

Cuando entras a una tienda online con miles de productos, encontrar lo que buscas 
es difícil. Pero el problema más grande no es ese — es descubrir productos que 
te encantarían y que nunca hubieras buscado por tu cuenta. ShopMind ataca ese 
problema aprendiendo de los patrones de comportamiento de los usuarios y del 
contenido de los productos para hacer recomendaciones que realmente tienen sentido.

---

## Demo en vivo

👉 [Probar ShopMind] (https://huggingface.co/spaces/dart01/shopmind) 
---

## Cómo funciona

Decidí implementar dos enfoques distintos porque cada uno resuelve un problema 
diferente, y compararlos fue una de las partes más interesantes del proyecto.

### Collaborative Filtering con ALS

La idea detrás de este modelo es sencilla: si dos usuarios compraron productos 
similares en el pasado, probablemente les gusten cosas parecidas en el futuro. 
El algoritmo ALS (Alternating Least Squares) toma una matriz enorme de 
52,204 usuarios × 57,289 productos y la factoriza en vectores de 50 dimensiones 
que capturan los gustos de cada usuario y las características de cada producto.

El gran reto aquí fue la dispersión de los datos — solo el 0.013% de las 
combinaciones posibles usuario-producto tienen un rating. En la vida real esto 
siempre es así, y ALS está diseñado específicamente para trabajar bien en 
estas condiciones.

### Embeddings Semánticos

El segundo modelo funciona de manera completamente diferente. En lugar de mirar 
el comportamiento de los usuarios, analiza el contenido de los productos. 
Usando el modelo all-MiniLM-L6-v2 de Hugging Face, convertí las descripciones 
de 57,289 productos en vectores de 384 dimensiones. Dos productos con 
descripciones similares quedan cerca en ese espacio vectorial, lo que permite 
recomendar por similitud de contenido usando cosine similarity.

Este enfoque es especialmente útil para el problema del arranque en frío — 
cuando un usuario nuevo no tiene historial, todavía podemos recomendarle 
productos similares a los que está viendo.

---

## Los datos

Trabajé con el dataset de Amazon Beauty Reviews disponible en Kaggle, que 
contiene más de 2 millones de reseñas reales de usuarios comprando productos 
de belleza. Después de un proceso de filtrado para quedarnos solo con usuarios 
y productos con al menos 5 interacciones, el dataset quedó en 394,908 reseñas 
con 52,204 usuarios activos y 57,289 productos.

El rating promedio es de 4.15 sobre 5, lo que refleja el sesgo positivo típico 
de las reseñas en e-commerce — la gente tiende a reseñar más cuando está 
satisfecha.

---

## Estructura del proyecto

```text
shopmind/
├── data/
│   ├── raw/                  ← datos originales de Kaggle
│   └── processed/            ← datos limpios, embeddings y modelos
├── notebooks/
│   ├── 01_eda.ipynb          ← análisis exploratorio de datos
│   ├── 02_collaborative_filtering.ipynb
│   └── 03_embeddings.ipynb
├── src/
│   └── data_loader.py        ← pipeline de descarga via API de Kaggle
├── app/
│   └── app.py                ← demo interactiva en Streamlit
└── README.md
```

---

## Tecnologías usadas

| Capa | Herramientas |
|---|---|
| Datos | pandas, numpy, scipy |
| Collaborative Filtering | implicit (ALS) |
| Embeddings | sentence-transformers, scikit-learn |
| App | Streamlit |
| Fuente de datos | API de Kaggle + requests |

---

## Correr el proyecto localmente

```bash
# Clonar el repositorio
git clone https://huggingface.co/spaces/dart01/shopmind
cd shopmind

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Agregar credenciales de Kaggle
# Crear kaggle.json con tu usuario y token

# Descargar el dataset
python src/data_loader.py

# Correr la app
streamlit run app/app.py
```

---

## Lo que aprendí

Más allá del código, este proyecto me enseñó que los datos del mundo real 
son complicados. La dispersión extrema de la matriz usuario-producto fue el 
mayor obstáculo técnico, y entender por qué ALS funciona bien en esas 
condiciones me hizo comprender mucho mejor cómo piensan los algoritmos de 
recomendación en producción.

También aprendí que ningún modelo es suficiente por sí solo — el collaborative 
filtering falla con usuarios nuevos, y los embeddings semánticos no capturan 
preferencias personales. La combinación de los dos es lo que hace un sistema 
robusto.

---

## Autor

**Diego Riaño - ingeniero mecatronico**  
https://www.linkedin.com/in/diegoandres001/· https://github.com/dart01 · https://www.kaggle.com/diegoandresriao
