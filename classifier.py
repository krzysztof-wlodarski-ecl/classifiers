"""
GUI which allows experimenting with different classifiers and their parameters.
Requires your own set of images divided into two classes.
"""
import itertools
import os
import pickle
import gradio as gr
import random

import numpy as np

from PIL import Image
from tqdm import tqdm

from utils.embedding import EmbeddingEngine

engine = EmbeddingEngine()

# -------------------------------------------------------------
# data

# step 1 - compose paths and their labels

false_path = "T:\\dev\\classifiers\\images\\sharpened\\clean"
false_path = os.path.abspath(false_path)
false_keys = [os.path.join(false_path, f) for f in os.listdir(false_path) if f.split(".")[-1].lower() in ["png", "jpg", "jpeg"]]
false_keys = random.choices(false_keys, k=500)

true_path = "T:\\dev\\classifiers\\images\\sharpened\\sharpened"
true_path = os.path.abspath(true_path)
true_keys = [os.path.join(true_path, f) for f in os.listdir(true_path) if f.split(".")[-1].lower() in ["png", "jpg", "jpeg"]]
true_keys = random.choices(true_keys, k=500)

false_labeled_keys = zip(false_keys, [0] * len(false_keys))
true_labeled_keys = zip(true_keys, [1] * len(true_keys))
labeled_keys = list(itertools.chain(false_labeled_keys, true_labeled_keys))
random.shuffle(labeled_keys)

data = {}
# step 2 - enrich data with embeddings
for i in tqdm(range(len(labeled_keys))):
    vector = engine.embed_single(labeled_keys[i][0])
    data[labeled_keys[i][0]] = { "vector": vector, "label": labeled_keys[i][1]}

# step 3 - divide into train set and test set
# one day I'll learn how to use library for this
keys = list(data.keys())
divider = int(len(keys)*0.66)
train_keys = keys[:divider]
test_keys = keys[divider:]

train_vectors = np.array([data[key]["vector"] for key in train_keys])
train_labels = [data[key]["label"] for key in train_keys]

test_data = np.array([data[key]["vector"] for key in test_keys])
test_labels = [data[key]["label"] for key in test_keys]
test_labels = np.array(test_labels)

# ------------------------
# classification

def prepare_roc(test_labels, probs):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    fpr, tpr, thresholds = roc_curve(test_labels, probs)  # FP rate, TP rate, progi decyzyjne
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Linie losowej klasyfikacji
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("temp.png")


def decision_tree(max_leaf_nodes, max_features, min_samples_split, max_depth):
    max_leaf_nodes = None if max_leaf_nodes in [None, ""] else int(max_leaf_nodes)
    max_features = None if max_features in [None, ""] else int(max_features)
    min_samples_split = 2 if min_samples_split in [None, ""] else int(min_samples_split)
    max_depth = None if max_depth in [None, ""] else int(max_depth)

    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, max_features=max_features, min_samples_split=min_samples_split)
    clf.fit(train_vectors, train_labels)

    probs = clf.predict_proba(test_data)[:, 1]
    prepare_roc(test_labels, probs)

    im = Image.open("temp.png")
    return im, clf


def svm(C, kernel, gamma, degree, max_iter):
    C = float(C)
    C = 10 ** C
    degree = int(degree)
    max_iter = int(max_iter)
    from sklearn.svm import SVC

    clf = SVC(probability=True, kernel=kernel, C=C, gamma=gamma, degree=degree, max_iter=max_iter)  # Możesz dostosować kernel, C i gamma
    clf.fit(train_vectors, train_labels)

    probs = clf.predict_proba(test_data)[:, 1]  # Prawdopodobieństwa dla klasy pozytywnej
    prepare_roc(test_labels, probs)

    im = Image.open("temp.png")
    return im, clf

def test(classifier, path):
    result = engine.embed_path(path)
    paths, vectors = zip(*result)
    vectors = np.array(vectors)
    labels = classifier.predict(vectors)

    positive = np.where(labels == 1)[0]
    negative = np.where(labels == 0)[0]

    positive = [paths[idx] for idx in positive]
    negative = [paths[idx] for idx in negative]
    return positive, negative


with gr.Blocks() as interface:
    trained_classifier = gr.State()
    with gr.Row():
        with gr.Tab("Decision tree"):
            with gr.Row():
                max_leaf_nodes = gr.Textbox(label="Max leaf nodes")
                max_features = gr.Textbox(label="Max features")
                min_samples_split = gr.Textbox(label="Min samples split")
                max_depth = gr.Textbox(label="Max depth")
            with gr.Row():
                run_decision_tree = gr.Button("Run")
        with gr.Tab("SVM"):
            with gr.Row():
                C = gr.Slider(label="Exponent of C (C operates on log scale)", minimum=-1, maximum=2, step=0.1, value=0)
                kernel = gr.Dropdown(choices=["rbf", "linear", "poly", "sigmoid"], label="Kernel", value="rbf")
                gamma = gr.Textbox(label="Gamma: [scale, auto, 0.01, 0.1, 1]", value="scale")
                degree = gr.Textbox(value="3", label="Degree of polynomial kernel")
                max_iter = gr.Slider(minimum=-1, maximum=10000, value=-1, label="Max iterations")
            with gr.Row():
                run_svm = gr.Button("Run")

    with gr.Row():
        with gr.Tab("Evaluation"):
            roc = gr.Image()
        with gr.Tab("Test"):
            with gr.Row():
                test_path = gr.Textbox(label="Tested path")
                run_test = gr.Button("Run test")
            with gr.Row():
                with gr.Tab("Positive finds"):
                    positive_test_results = gr.Gallery(show_label=True, object_fit="contain", columns=10)
                with gr.Tab("Negative finds"):
                    negative_test_results = gr.Gallery(show_label=True, object_fit="contain", columns=10)

    run_test.click(fn=test, inputs=[trained_classifier, test_path], outputs=[positive_test_results, negative_test_results])
    run_decision_tree.click(fn=decision_tree, inputs=[max_leaf_nodes, max_features, min_samples_split, max_depth], outputs=[roc, trained_classifier])
    run_svm.click(fn=svm, inputs=[C, kernel, gamma, degree, max_iter], outputs=[roc, trained_classifier])


interface.launch(allowed_paths=["T:\\", "F:\\"])