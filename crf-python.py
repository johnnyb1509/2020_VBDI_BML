!pip install python-crfsuite
import pycrfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report




#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
trainer = pycrfsuite.Trainer(verbose=True)
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)


# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,  

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('crf.model')

# Generate predictions
tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

# Let's take a look at a random sample in the testing set
#i = 12
#for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
#    print("%s (%s)" % (y, x))

# Create a mapping of labels to indices
labels = {"N": 1, "I": 0}

# Convert the sequences of tags into a 1-dimensional array
predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])

# Print out the classification report
print(classification_report(
    truths, predictions,
    target_names=["I", "N"]))