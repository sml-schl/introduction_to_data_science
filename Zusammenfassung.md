Zusammenfassung der Vorlesung "Introduction to Data Science"
Tag 2: Einführung in Data Science (04.11.2025)
Lernziele:

Definition und Umfang von Data Science als interdisziplinäres Feld verstehen
Die drei Säulen von Data Science identifizieren: Domänenwissen, Statistik/Mathematik und Informatik
Unterscheidung zwischen AI, Machine Learning, Deep Learning und Generative AI
Big Data Charakteristiken verstehen (Volume, Velocity, Variety, Veracity, Value)
Datenquellen identifizieren (Open-Source, privat, kommerziell)
Prinzipien effektiver Datenvisualisierung anwenden
Schlechte oder irreführende Visualisierungen erkennen
Unterschied zwischen Korrelation und Kausalität erkennen

Inhalte:

Data Science Grundlagen: Definition als interdisziplinäres Feld aus Domänenexpertise, Statistik/Mathematik und Informatik
Anwendungsfälle: Qualitätskontrolle, Predictive Maintenance, Betrugserkennung, autonomes Fahren, Empfehlungssysteme
AI-Hierarchie: AI → Machine Learning → Deep Learning → LLM/GenAI
Big Data: Die 5 V's (Volume, Velocity, Variety, Veracity, Value)
Datenquellen: Open-Source Daten (Kaggle, WHO, etc.), private und kommerzielle Daten
Data Storytelling und Visualisierung:

Verschiedene Darstellungsmöglichkeiten (Länge, Breite, Orientierung, Größe, Form, Position, Farbe)
Schlechte/irreführende Visualisierungen erkennen
Diagrammtypen: Liniendiagramme, Balkendiagramme, Scatterplots, Heatmaps, etc.


Chancen und Risiken: Datenschutz, Manipulation, ethische Überlegungen


Tag 4: Datenaufbereitung & Feature Engineering (06.11.2025)
Lernziele:

Unterscheidung zwischen Objekten, Daten, Datenbanken und Informationen
Eigenschaften qualitativ hochwertiger Datensätze verstehen
Zwischen strukturierten und unstrukturierten Datenformaten unterscheiden
Dimensionale Datenstrukturen anwenden (Dimensionen vs. Fakten)
Datenoperationen durchführen: Slicing, Dicing, Roll-Up, Drill-Down
Bedeutung der Datenaufbereitung verstehen (45-80% der Arbeitszeit)
Fehlende Daten identifizieren und behandeln (MCAR, MAR, MNAR)
Imputation-Techniken anwenden
Ausreißer erkennen und behandeln (IQR, Z-Score, Isolation Forest)
Normalisierungstechniken anwenden (Min-Max, Z-Score)
Kategoriale Kodierung durchführen (One-Hot Encoding)
Feature Selection und Extraction durchführen

Inhalte:

Wissenstreppe nach Prof. Klaus North: Daten → Information → Wissen → Fähigkeiten/Handlungen
Eigenschaften von Datensätzen:

Volumen, Historie, Konsistenz, Reinheit, Detailgrad, Klarheit, transparente Herkunft
Dimensionale Struktur (Dimensionen und Fakten)


Datenoperationen: Slicing (Selektion), Dicing (Projektion), Roll-Up (Aggregation), Drill-Down (Disaggregation)
Strukturierte vs. unstrukturierte Daten
Data Preparation (45% der Zeit):

Datensammlung, Integration, Transformation


Data Cleansing:

Fehlende Werte: MCAR, MAR, MNAR und Imputation-Strategien
Ausreißer: Erkennung mit IQR, Z-Score, Isolation Forest
Normalisierung: Min-Max (0-1 Skalierung), Z-Score (Standardisierung)
One-Hot Encoding: Umwandlung kategorialer Variablen


Feature Engineering: Feature Selection und Extraction zur Modellverbesserung


Tag 6: Einführung in maschinelles Lernen (11.11.2025)
Lernziele:

Machine Learning als Teilbereich von AI und Optimierungsaufgabe verstehen
Zwischen Supervised, Unsupervised und Reinforcement Learning unterscheiden
Train-Test Split und Cross-Validation anwenden
Klassifikationsaufgaben und -algorithmen verstehen (KNN, Decision Trees, SVM)
Regressionsaufgaben verstehen
Unsupervised Learning Techniken anwenden (K-means Clustering)
Grundarchitektur neuronaler Netze verstehen
Wichtige Hyperparameter identifizieren
Rolle von Aktivierungsfunktionen verstehen
Modelltraining und Loss-Funktionen verstehen

Inhalte:

Machine Learning Paradigmen:

Supervised Learning: Lernen mit gelabelten Daten

Klassifikation (z.B. Spam-Erkennung, Bilderkennung)
Regression (z.B. Preisvorhersage, Wettervorhersage)
Algorithmen: KNN, Decision Trees, SVM, Random Forest


Unsupervised Learning: Lernen ohne Labels

Clustering (K-means)
Pattern Discovery


Reinforcement Learning: Lernen durch Interaktion und Belohnungen


Train-Test Split und Cross-Validation
Neuronale Netze:

Aufbau: Input Layer, Hidden Layers, Output Layer
Künstliche Neuronen und Gewichtungen
Aktivierungsfunktionen: Sigmoid, Tanh, ReLU


Hyperparameter:

Train/Test Ratio, Batch Size, Epochs, Learning Rate
Loss/Cost Functions
Anzahl Hidden Layers


Modelltraining: Iterativer Prozess der Gewichtsanpassung


Tag 8: Modellbewertung & Prüfungsleistung (11.11.2025)
Lernziele:

Bias und Variance in ML-Modellen verstehen
Overfitting und Underfitting erkennen
Performance-Metriken für Regression anwenden (MSE, RMSE, MAE, R²)
Performance-Metriken für Klassifikation anwenden (Precision, Recall, Accuracy, F1-Score)
Confusion Matrix interpretieren
ROC/AUC-Kurven verstehen
Type 1 und Type 2 Fehler unterscheiden
Deployment-Überlegungen verstehen
Ethische Überlegungen und Risiken in AI-Systemen erkennen
EU AI Act verstehen
Trustworthy AI Prinzipien verstehen
Explainable AI (XAI) verstehen

Inhalte:

Bias und Variance:

High Variance (Overfitting): Modell zu flexibel, "memoriert" Trainingsdaten
High Bias (Underfitting): Modell zu einfach/inflexibel


Performance-Metriken für Regression:

MSE (Mean Squared Error)
RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
R² (Bestimmtheitsmaß)


Performance-Metriken für Klassifikation:

Confusion Matrix (TP, FP, TN, FN)
Precision, Recall, Accuracy, F1-Score
ROC/AUC-Kurven
Type 1 Fehler (False Positive) vs. Type 2 Fehler (False Negative)


Deployment: Überlegungen zur Produktivschaltung von ML-Modellen
Ethische Überlegungen:

Unfaire AI-Entscheidungen
Autonome Systeme und Unfälle
Datenschutz und Vertraulichkeit


EU AI Act:

Risikobasierter Ansatz (Unacceptable, High, Limited, Minimal Risk)
Anforderungen und Strafen (bis zu €35 Mio / 7% Jahresumsatz)
Timeline (2024-2027)


Trustworthy AI:

Explainability & Comprehensibility
Fairness & Inclusiveness
Privacy
Robustness & Performance
Accountability
Security


Explainable AI (XAI): "Wie" ist genauso wichtig wie "Was" - Robustheit und Erklärbarkeit von AI-Systemen