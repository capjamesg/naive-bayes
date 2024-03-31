from collections import Counter

import numpy as np
import pandas as pd

df = pd.read_csv("ds.csv", header=0)

df = df.sample(frac=1).reset_index(drop=True)

df["class_name"] = df["class_name"].str.lower().str.strip()
df["text"] = df["text"].str.lower().str.strip()
classes = df["class_name"].unique()

word_counts_by_class = {class_name: Counter() for class_name in classes}

for i, row in df.iterrows():
    class_name = row["class_name"].strip()
    text = row["text"]
    words = text.split()
    word_counts_by_class[class_name].update(words)

probability_table_by_class = {class_name: {} for class_name in classes}

for class_name, word_counts in word_counts_by_class.items():
    total_words = sum(word_counts.values())
    class_name = class_name.strip()
    for word, count in word_counts.items():
        probability_table_by_class[class_name][word] = count / total_words


def naive_bayes_classifier(text):
    words = text.split()
    class_probs = {class_item: [] for class_item in classes}

    for word in words:
        for class_name in classes:
            class_probs[class_name].append(
                probability_table_by_class[class_name].get(word, 0)
            )

    for class_name, probs in class_probs.items():
        class_probs[class_name] = np.prod(probs)

    return max(class_probs, key=class_probs.get), max(class_probs.values())


example = "Taylor Swift is award winning"
print("Example:", example)
print("Class:", naive_bayes_classifier(example.lower().strip())[0])
