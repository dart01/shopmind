import json
import os
import zipfile

import requests

with open("kaggle.json") as f:
    creds = json.load(f)

KAGGLE_USER = creds["username"]
KAGGLE_TOKEN = creds["key"]

print(f"Usuario: {KAGGLE_USER}")
print(f"Token: {KAGGLE_TOKEN[:6]}...")  # solo muestra los primeros 6 caracteres


def descargar_dataset(owner, dataset, destino="data/raw"):
    url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset}"
    print(f"Conectando a: {url}")

    response = requests.get(
        url, auth=(KAGGLE_USER, KAGGLE_TOKEN), stream=True, timeout=60
    )

    print(f"Status code: {response.status_code}")

    if response.status_code != 200:
        print(f"Error: {response.text}")
        return

    os.makedirs(destino, exist_ok=True)
    zip_path = f"{destino}/{dataset}.zip"

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Descomprimiendo...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(destino)

    os.remove(zip_path)
    print(f"✓ Listo en {destino}/")


if __name__ == "__main__":
    descargar_dataset("webdevbadger", "amazon-beauty-products")
