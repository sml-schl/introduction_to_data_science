# 🧠 Introduction to Data Science – Setup Guide

Willkommen zur Vorlesung **Introduction to Data Science** von **Samuel Schlenker & Joel Weiss**.
Dieses Dokument führt dich Schritt für Schritt durch die Installation und das Setup deiner Entwicklungsumgebung.

---

## 🪟 Windows Setup

### 1. Installiere WSL

Installiere **Ubuntu für WSL** über den folgenden Link:
🔗 [https://ubuntu.com/desktop/wsl](https://ubuntu.com/desktop/wsl)

Teste anschließend die Installation in PowerShell:

```powershell
wsl.exe –version
```

---

### 2. Installiere Podman

Lade **Podman Desktop** über den folgenden Link herunter:
🔗 [https://podman-desktop.io/downloads](https://podman-desktop.io/downloads)

Teste die Installation in PowerShell:

```powershell
podman –version
```

---

## 🍎 macOS Setup

### 1. Installiere Homebrew

Homebrew ist der Paketmanager für macOS.
Installiere ihn im Terminal mit folgendem Befehl:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Teste die Installation mit:

```bash
brew –version
```

---

### 2. Installiere Podman

Installiere Podman und Podman Desktop über Homebrew:

```bash
brew install podman
brew install --cask podman-desktop
```

Teste anschließend die Installation:

```bash
podman –version
```

---

## 🧰 Gemeinsame Schritte (Windows & macOS)

### 1. Erstelle deine virtuelle Podman-Maschine

```bash
podman machine init
podman machine start
```

---

### 2. Deploye deinen ersten Container

```bash
podman run hello-world
```

---

### 3. Lade ein Image herunter

```bash
podman pull busybox
```

Prüfe deine Images:

```bash
podman images
```

---

### 4. Starte weitere Container

```bash
podman run busybox
podman run busybox echo “hello from busybox”
podman run –it busybox sh
```

---

### 5. Zeige deine Container an

```bash
podman ps
podman ps –a
```

---

### 6. Bereinige deine Umgebung

```bash
podman rm ID1 ID2
```

Oder lösche alle gestoppten Container:

```bash
podman system prune
```

Lösche ungenutzte Images:

```bash
podman image prune -a
```

---

## 📓 Jupyter Setup

Sobald Podman eingerichtet ist, kannst du Jupyter direkt in einem Container starten.

---

### 1. Öffne dein Terminal

Erstelle einen neuen Ordner, z. B.:

```bash
mkdir vorlesung
cd vorlesung
```

---

### 2. Starte einen Jupyter Container

#### macOS:

```bash
podman run -it --rm -p 8888:8888 \
-v "$(pwd)":/home/jovyan/work \
docker.io/jupyter/base-notebook:latest
```

Command for the Image with data and installed requirements
```bash
podman run -p 8888:8888 -v "$(pwd)":/home/jovyan/work docker.io/jocowhite/intro-ds:latest
```


#### Windows (PowerShell):

```powershell
podman run -it --rm -p 8888:8888 -v "${PWD}:/home/jovyan/work" docker.io/jupyter/base-notebook:latest
```

Command for the Image with data and installed requirements
```bash
podman run -p 8888:8888 -v "${PWD}:/home/jovyan/work" docker.io/jocowhite/intro-ds:latest
```


---

### 3. Öffne deine Umgebung im Browser

Kopiere den Link aus dem Terminal, der mit
`http://127.0.0.1:8888` beginnt.

> ⚠️ **Achtung:**
> Ändere den Port von **8888** auf **88**
> (z. B. `http://127.0.0.1:88/lab`)
> und prüfe, dass **kein www** davor steht.

Falls du aufgefordert wirst, einen Token einzugeben,
kopiere alles **nach dem "=" Zeichen** aus dem Terminal-Link.

---

## 📘 Notebook herunterladen und starten

Lade das Notebook für den ersten Praxisteil herunter:
🔗 [day1_hello_data_science.ipynb](https://github.com/sml-schl/introduction_to_data_science/blob/9e96e707c5f093d492f18fe2830008da73d6de83/hands-on-lab/day1_hello_data_science.ipynb)

Speichere die Datei in deinem Arbeitsordner (`vorlesung`),
öffne sie im **Jupyter Notebook** (im Browser) und führe **alle Zellen** nacheinander aus.

---

## 💻 Installation von Visual Studio Code

### 1. Installiere VS Code

Lade VS Code über den folgenden Link herunter:
🔗 [https://code.visualstudio.com/download](https://code.visualstudio.com/download)

#### macOS (über Homebrew):

```bash
brew install --cask visual-studio-code
```

---

### 2. Verbinde VS Code mit deinem Jupyter-Server

1. Öffne ein Notebook in VS Code.
2. Klicke oben auf **Select Kernel**.
3. Wähle **Install/Enable suggested extensions**.
4. Warte, bis die Erweiterungen installiert sind.
5. Klicke auf **Existing Jupyter Server…**.
6. Füge die URL aus deinem Terminal ein, z. B.:

Beispiellink:
```
http://127.0.0.1:8888/lab?token=eae0d08f5a3e0d4083d3e5d8b54c071daea0604a4036365a
```