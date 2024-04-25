from sklearn.metrics import classification_report

def print_report(truelabels, predictions):

    truelabels = truelabels.detach().numpy()
    predictions = predictions.detach().numpy()
    print(classification_report(truelabels, predictions, digits=4))

