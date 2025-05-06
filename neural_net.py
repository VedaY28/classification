import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time

digits = ("data/digitdata/trainingimages", "data/digitdata/traininglabels",
        "data/digitdata/validationimages", "data/digitdata/validationlabels",
        "data/digitdata/testimages", "data/digitdata/testlabels")
face = ("data/facedata/facedatatrain", "data/facedata/facedatatrainlabels",
        "data/facedata/facedatavalidation", "data/facedata/facedatavalidationlabels",
        "data/facedata/facedatatest", "data/facedata/facedatatestlabels")

CHAR_TO_PIXEL = {' ': 0, '+': 1, '#': 1}

def parse_images(image_file, label_file, height, width):

    labels = []
    with open(label_file, 'r') as lf:
        for line in lf:
            if line.strip():
                labels.append(int(line.strip()))

    lines = []
    with open(image_file, 'r') as imf:
        lines = [ln.rstrip('\n') for ln in imf]

    images = []
    n = len(labels)
    for i in range(n):
        block = lines[i * height : (i + 1) * height]
        flat = []
        for row in block:
            row = row.ljust(width)
            for ch in row:
                flat.append(CHAR_TO_PIXEL.get(ch, 0))
        images.append(flat)

    return np.array(images), np.array(labels)


class NeuralNetwork:
    def __init__(self, input_size, hidden1, hidden2, output_size, learning_rate=0.01):
        self.lr = learning_rate

        self.W1 = np.random.randn(hidden1, input_size) * np.sqrt(1.0 / input_size)
        self.b1 = np.zeros((hidden1, 1))
        
        self.W2 = np.random.randn(hidden2, hidden1) * np.sqrt(1.0 / hidden1)
        self.b2 = np.zeros((hidden2, 1))
        
        self.W3 = np.random.randn(output_size, hidden2) * np.sqrt(1.0 / hidden2)
        self.b3 = np.zeros((output_size, 1))


    def relu(self, Z):
        return np.maximum(0, Z)


    def relu_deriv(self, Z):
        return (Z > 0).astype(float)


    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)


    def forward(self, x):
        z1 = self.W1 @ x + self.b1
        a1 = self.relu(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = self.relu(z2)
        z3 = self.W3 @ a2 + self.b3
        a3 = self.softmax(z3)
        return a1, a2, a3


    def backward(self, x, y_true, a1, a2, a3):
        y = np.zeros_like(a3)
        y[y_true] = 1

        dz3 = a3 - y
        dW3 = dz3 @ a2.T
        db3 = dz3

        dz2 = (self.W3.T @ dz3) * self.relu_deriv(a2)
        dW2 = dz2 @ a1.T
        db2 = dz2

        dz1 = (self.W2.T @ dz2) * self.relu_deriv(a1)
        dW1 = dz1 @ x.T
        db1 = dz1

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


    def predict(self, X):
        preds = []
        for xi in X:
            xi = xi.reshape(-1, 1)
            _, _, a3 = self.forward(xi)
            preds.append(np.argmax(a3))
        return np.array(preds)


def evaluate(model, X, y):
    preds = model.predict(X)
    return np.mean(preds == y)


def train(model, X_train, y_train, X_val, y_val, epochs=5):

    best_model = copy.deepcopy(model)
    best_acc = 0.0

    for epoch in range(epochs):

        for i in range(len(X_train)):
            x = X_train[i].reshape(-1, 1)
            a1, a2, a3 = model.forward(x)
            model.backward(x, y_train[i], a1, a2, a3)

        val_acc = evaluate(model, X_val, y_val)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)

    return best_model


def run(
    train_img, train_lbl, val_img, val_lbl, test_img, test_lbl,
    classes, height, width, title):

    X_train, y_train = parse_images(train_img, train_lbl, height, width)
    X_val,   y_val   = parse_images(val_img,   val_lbl,   height, width)
    X_test,  y_test  = parse_images(test_img,  test_lbl,  height, width)
    inp_size = height * width

    stats = [] 
    
    print(f"\n{title}:")
    total = time.time()

    for perc in range(10, 101, 10):
        accs = []
        start = time.time()
        N = len(X_train)

        for _ in range(5):
            idx = list(range(N))
            random.shuffle(idx)
            k = int(perc / 100 * N)
            sel = idx[:k]

            # # 2) sanity checks right here
            # print(f"\n--- Sampling {perc}% of {N} examples → {k} samples ---")
            # assert k == len(sel), f"sel should have {k} but has {len(sel)}"

            # X_sub = X_train[sel]
            # y_sub = y_train[sel]

            # print(f"X_sub shape: {X_sub.shape}, y_sub length: {len(y_sub)}")
            # unique, counts = np.unique(y_sub, return_counts=True)
            # print("Class counts in this subset:", dict(zip(unique, counts)))

            # print(f"Requested {perc}% of data → {k} samples")
            # assert X_sub.shape[0] == k, "Wrong number of samples in X_sub!"
            # assert len(y_sub)   == k, "Wrong number of samples in y_sub!"

            model = NeuralNetwork(inp_size, 64, 16, classes, learning_rate=0.01)
            best_model = train(model, X_train[sel], y_train[sel], X_val, y_val, epochs=5)
            accs.append(evaluate(best_model, X_test, y_test))

        duration = time.time() - start
        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        stats.append((perc, mean_acc, std_acc, duration))
        print(f"{perc}% → Accuracy {mean_acc:.4f} | STD {std_acc:.4f} | Time {duration:.2f}s")

    perc_list, mean_list, std_list, time_list = zip(*stats)
    total = sum(duration for _, _, _, duration in stats)
    print(f"Total Time: {total}")

    return stats



if __name__ == "__main__":
    # Digit Classification
    stats1 = run(*digits, classes=10, height=28, width=28, title="Digits")

    # Face Classification
    stats2 = run(*face, classes=2, height=70, width=60, title="Faces")

    perc_list1, mean_list1, std_list1, time_list1 = zip(*stats1)
    perc_list2, mean_list2, std_list2, time_list2 = zip(*stats2)

    # Plot: Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(perc_list1, mean_list1, linewidth=2, label="Digits")
    plt.plot(perc_list2, mean_list2, linewidth=2, label="Faces")
    plt.title("Test Accuracy vs. Training Size")
    plt.xlabel("Training Data (%)")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.show()

    # Plot: Std Dev of Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(perc_list1, std_list1, linewidth=2, label="Digits")
    plt.plot(perc_list2, std_list2, linewidth=2, label="Faces")
    plt.title("Std Dev of Accuracy vs. Training Size")
    plt.xlabel("Training Data (%)")
    plt.ylabel("Std Dev of Accuracy")
    plt.legend()
    plt.show()

    # Plot: Training Time
    plt.figure(figsize=(8,5))
    plt.plot(perc_list1, time_list1, linewidth=2, label="Digits")
    plt.plot(perc_list2, time_list2, linewidth=2, label="Faces")
    plt.title("Training Time vs. Training Size")
    plt.xlabel("Training Data (%)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()

