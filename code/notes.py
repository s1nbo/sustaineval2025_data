'''
=============================================================================
Title:        Exploratory Data Analysis (EDA) and Model Training for Sustainability Classification
Description:  This script loads, analyzes, visualizes, tokenizes, and fine-tunes a model
              for Task A (Content Classification) of the SustainEval Challenge.
Task Link:    https://sustaineval.github.io/

Idee:
    *Daten vorbereiten:
        - Reduziere labels um -1, da Model Startlabel bei 0 erwartet
        - Erstelle Spalten für label namen (auch auf deutsch), super labels, super label namen (auch auf deutsch)
        - Bereite context Spalte als string auf falls als Liste gespeichert
        - Erstelle Spalten für word_count, letter_count

    *Analysieren: 
        - erstelle Plots um Beispiele pro Label zu validieren (ausgewogene Beispiele in Training und Development)
        - Entscheidung zur Featureauswahl:  'year' gleichverteilt über alle labels -> kein Mehrwert für Model. 
                                            'word count' leichte Abweichungen je label -> wird aufgenommen

    *Trainiere AUTO-Modell:
        - Verwendung von Huggingface Datasets
        - Verwendung von AutoTokenizer mit vortrainiertem 'bert-base-german-cased' 
        - Vortrainiertes Modell: 'bert-base-german-cased' mit AutoModelForSequenceClassification
        - Trainingsdaten enthalten den vollständigen Text (context + ' ' + target)
        - Nutzung von gegebenen training-data und validation-data
        - Validierungsdaten sind unlabeled (keine Optimierung basierend auf Accuracy)

    *Trainiere CUSTOM-Modell:
        - Verwendung von Custom Datasets, die zusätzliche Features und Superlabels verarbeiten können
        - Verwendung von BertTokenizer mit vortrainiertem 'bert-base-german-cased' 
            ° wählt unter 3000 mit Chi^2 Feature-Selection die x wichtigsten Begriffe aus, die für die Klassifikation am relevantesten sind
        - Custom BERT-Modell (ebenfalls vortrainiert mit 'bert-base-german-cased')
            ° verarbeitet super labels
            ° verarbeitet context und target als getrennten input und startet target mit höherer Gewichtung (1.2), die dynamisch während des lernprozesses angepasst wird
            ° verarbeitet extra features (Textlängen werden als Feature übergeben)
        - keine Nutzung der validation-data
        - Trainingsdaten werden in Trainings- und Validation-Daten gesplittet, sodass wir gelabelte Validation Daten haben
        - Trainer optimiert auf Accuracy

    *Evaluieren:
        - wende trainiertes Model auf development-data an
        - Vergleiche Ergebnis mit tatsächlichen Labels
        - Erstelle Confusion Matrix und Klassifikationsreport zur Auswertung


TO DO:  - Ordnerpfad anpassen (Zeile 87) 
        - 

=============================================================================
'''


import os
import torch
import time
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import gridspec, pyplot as plt
from datetime import datetime
from collections import Counter
from datasets import Dataset as HFDataset
from joblib import dump, load
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                         Trainer, TrainingArguments, DataCollatorWithPadding,
                         BertTokenizer, BertModel, TrainerCallback)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             classification_report, confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from safetensors.torch import load_model
from sklearn.feature_selection import SelectKBest, chi2