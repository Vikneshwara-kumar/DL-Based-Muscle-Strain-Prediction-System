import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import numpy as np

labels = ['Resting', 'Low Strain','Medium Strain','Max Strain']

def metric(y_test, classes):
    cm = confusion_matrix(np.argmax(y_test, axis=1), classes)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    print(classification_report(y_test, classes))