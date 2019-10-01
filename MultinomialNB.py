import numpy as np
import pandas as pd

# -----------------------------------------------------
# | code written by Matthew Belford and Rolin Buckoke |
# -----------------------------------------------------

# Predicts classification based on prior data
def ApplyMultinomialNB(C,V,prior,condprob,d):
    W = []
    score = [0,0]
    for x in d.split():
        if x in V:
            W.append(x)
    for i in range(0,2):
        score[i] = np.log(prior[i])
        for t in W:
            score[i] += np.log(condprob[i][t])
    if score[0] > score[1]:
        return 0, score[0]
    else:
        return 1, score[1]
    return

# builds conditional probability matrix
def TrainMultiNomialNB(C, D):
    V = ExtractVocab(D)
    N = len(D)
    probability = [0,0]
    textc = [[],[]]
    condprob = [{},{}]
    for c in range(0, 2): # only has classes 0 and 1 for our purposes
        count = CountDocsInClass(C, c)
        probability[c] = count/N
        textc[c] = ConcatenateTextofAllDocsInClass(D, C, c)
        tct = {}
        for x in V:
            tct[x] = textc[c].count(x)
        for x in V:
            condprob[c][x] = (tct[x] + 1) / (len(textc[c]) + len(V))
    
    return V, probability, condprob

# gets all words in a class, returns list
def ConcatenateTextofAllDocsInClass(D, C, c):
    textInClass = []
    for i in range(0, len(D)):
        if str(c) == C[i]:
            for word in D[i].split():
                textInClass.append(word)
    return textInClass

# gets all unique words in data set
def ExtractVocab(D):
    words  = [] # unique words in sentences
    for line in D:
        for x in line.split():
            if x not in words:
                words.append(x)
    return words            

# counts number of instances of each class
def CountDocsInClass(C, c):
    count = 0
    for line in C:
        if str(c) in line:
            count += 1
    return count

if __name__ == "__main__":
    trainData = open("traindata.txt", "r")
    trainLabels = open("trainlabels.txt", "r")
    D = [line.rstrip() for line in trainData]
    C = [line.rstrip() for line in trainLabels]
    V,probability,condprob = TrainMultiNomialNB(C,D)
    
    testData = open("testdata.txt", "r")
    testLabels = open("testlabels.txt", "r")
    results = open("results.txt", "w+")

    guesses = []
    correctCount = 0
    incorrectCount = 0

    # train data
    for i in range(0,len(D)):
        guess, certainty = ApplyMultinomialNB(C, V, probability, condprob, D[i])
        guesses.append(guess)

    for i in range(0, len(C)):
        if str(guesses[i]) == C[i]:
            correctCount += 1
        else:
            incorrectCount += 1

    accuracy = correctCount / (correctCount + incorrectCount) * 100
    results.write("BEGIN USING TRAINING DATA\n")
    results.write("Training data file: traindata.txt\n")
    results.write("Training labels file: trainlabels.txt\n")
    results.write("Correct classifications: " + str(correctCount) + "\nIncorrect classifications: " + str(incorrectCount))
    results.write("\nOn training data accuracy = " + str(accuracy) + "%\n")
    results.write("_________________________________________________________\n\n")

    # TEST data
    guesses = []
    correctCount = 0
    incorrectCount = 0

    D = [line.rstrip() for line in testData]
    
    for i in range(0,len(D)):
        guess, certainty = ApplyMultinomialNB(C, V, probability, condprob, D[i])
        guesses.append(guess)

    C = [line.rstrip() for line in testLabels]

    for i in range(0, len(C)):
        if str(guesses[i]) == C[i]:
            correctCount += 1
        else:
            incorrectCount += 1

    accuracy = correctCount / (correctCount + incorrectCount) * 100
    
    results.write("BEGIN USING TEST DATA\n")
    results.write("Training data file: traindata.txt\n")
    results.write("Training labels file: trainlabels.txt\n")
    results.write("Test data file: testdata.txt\n")
    results.write("Compared classifications with: testlabels.txt\n")
    results.write("Correct classifications: " + str(correctCount) + "\nIncorrect classifications: " + str(incorrectCount))
    results.write("\nOn training data accuracy = " + str(accuracy) + "%\n")
    results.write("_________________________________________________________\n\n")

    
    trainData.close()
    trainLabels.close()
    testData.close()
    testLabels.close()
