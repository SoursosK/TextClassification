import re
import string
from nltk.stem.porter import PorterStemmer
from math import log10
from numpy.linalg import norm
from numpy import inner
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import roc_auc_score
from sklearn import  metrics

# nltk.download()

def removeNumbers(words):
    pattern = '[0-9]'
    words = [re.sub(pattern, '', i) for i in words]
    return words


def punctuationAndLower(words):
    words = words.translate(str.maketrans('', '', string.punctuation))
    words = words.lower()
    return words


def stemming(words):
    porter_stemmer = PorterStemmer()

    for i in range(len(words)):
        words[i] = porter_stemmer.stem(words[i])

    return words


def stopwordRemoval(words):
    stopwordList = set()
    with open("stoplist.txt", mode="r", encoding="utf-8") as file:
        for line in file:
            stopwordList.add(line.strip())

    # stopwordList = set(stopwords.words('english'))

    for word in reversed(words):
        if (word in stopwordList):
            words.remove(word)

    return words

def lengthLimit(words, limit):
    toRemove = list()
    for word in words:
        if(len(word) < limit+1):
            toRemove.append(word)

    for i in range(0, len(toRemove)):
        words.remove(toRemove[i])

    return words


def cleanText(dirNum, dataType):
    unknownWords = list()
    knownWords = list()

    with open("dataset/" + dataType + "/EN" + dirNum + "/known01.txt", mode="r", encoding='utf-8-sig') as file:
        words = file.read()
        words = punctuationAndLower(words)
        words = words.split()
        words = removeNumbers(words)
        words = list(filter(None, words))
        words = lengthLimit(words, 3)
        words = stopwordRemoval(words)
        knownWords = stemming(words)

    with open("dataset/" + dataType + "/EN" + dirNum + "/unknown.txt", mode="r", encoding='utf-8-sig') as file:
        words = file.read()
        words = punctuationAndLower(words)
        words = words.split()
        words = removeNumbers(words)
        words = list(filter(None, words))
        words = lengthLimit(words, 3)
        words = stopwordRemoval(words)
        unknownWords = stemming(words)

    return knownWords, unknownWords

def iterateFiles(dataType, num):
    txts = list()
    uniqueStemmedWords = set()

    for dirNum in range(1, num+1):
        if (dirNum >= 10):
            knownWords, unknownWords = cleanText(str(dirNum), dataType)
        else:
            knownWords, unknownWords = cleanText('0' + str(dirNum), dataType)

        txts.insert(dirNum - 1, [knownWords, unknownWords])
        uniqueStemmedWords.update(knownWords)
        uniqueStemmedWords.update(unknownWords)

    # tuple me tis le3eis ka8e keimenou
    # print(len(txts))
    # print(txts[0][0])
    # print(len(uniqueStemmedWords))

    return txts, uniqueStemmedWords

def createTFmatrix(uniqueStemmedWords, txts):
    # uniqueStemmedWords to list for indexing purposes
    uniqueStemmedWordsList = [word for word in uniqueStemmedWords]
    # print(uniqueStemmedWordsList)

    # Fill the matrix with 0s
    tfMatxix = [[0 for i in range(0, len(uniqueStemmedWordsList))] for y in range(0, len(txts) * 2)]

    # Iterate each one of the txts
    for i in range(0, len(txts) * 2):

        # For every uniqueStemmedWord (top row)
        for y in range(0, len(uniqueStemmedWordsList)):

            # check if the txt contains the uniqueStemmedWordsList[i]
            if (uniqueStemmedWordsList[y] in txts[int(i / 2)][i % 2]):

                # if it does, tf is the count of the word in the txt divided by the number of all words in the text
                tfMatxix[i][y] = txts[int(i / 2)][i % 2].count(uniqueStemmedWordsList[y]) / len(txts[int(i / 2)][i % 2])

    return tfMatxix

def createTFIDFMatrix(uniqueStemmedWords, txts):
    # uniqueStemmedWords to list for indexing purposes
    uniqueStemmedWordsList = [word for word in uniqueStemmedWords]
    # print(uniqueStemmedWordsList)

    # Fill the matrix with 0s
    tfidfMatxix = [[0 for i in range(0, len(uniqueStemmedWordsList))] for y in range(0, len(txts) * 2)]
    termAppearances = [0 for i in range(0, len(uniqueStemmedWordsList))]

    # iterate each one of the txts
    for i in range(0, len(txts) * 2):

        # For every uniqueStemmedWord (top row)
        for y in range(0, len(uniqueStemmedWordsList)):

            # check the txt contains the uniqueStemmedWordsList[i]
            if (uniqueStemmedWordsList[y] in txts[int(i / 2)][i % 2]):
                # if it does, increase the termAppearances counter by 1
                termAppearances[y] += 1
                # if it does, store the term frequency in the matrix's cell
                tfidfMatxix[i][y] = txts[int(i / 2)][i % 2].count(uniqueStemmedWordsList[y]) / len(txts[int(i / 2)][i % 2])

    # iterate each one of the txts
    for i in range(0, len(txts) * 2):
        # For every uniqueStemmedWord (top row)
        for y in range(0, len(uniqueStemmedWordsList)):

            # check the txt contains the uniqueStemmedWordsList[i]
            if (uniqueStemmedWordsList[y] in txts[int(i / 2)][i % 2]):

                # set the tf-idf index as the multiplification of: term frequency * idf ( log10(numberofTxts / termAppearances) )
                tfidfMatxix[i][y] = tfidfMatxix[i][y] * log10((len(txts)*2) /termAppearances[y])

    return tfidfMatxix

def euclideanDistance(tfMatrix, tfidfMatrix):
    euclideanTFDistance = list()
    euclideanTFIDFDistance = list()

    for i in range(0, len(tfMatrix), 2):
        euclideanTFDistance.append(sum((vectorΚnown - vectorUnknown) ** 2 for vectorΚnown, vectorUnknown in
                                       zip(tfMatrix[i], tfMatrix[i + 1])) ** 0.5)
        euclideanTFIDFDistance.append(sum((vectorΚnown - vectorUnknown) ** 2 for vectorΚnown, vectorUnknown in
                                          zip(tfidfMatrix[i], tfidfMatrix[i + 1])) ** 0.5)

    return euclideanTFDistance, euclideanTFIDFDistance

def cosineSimilatiry(tfMatrix, tfidfMatrix):
    cosineSimilatiryTF = list()
    cosineSimilatiryTFIDF = list()

    for i in range(0, len(tfMatrix), 2):
        cosineSimilatiryTF.append( inner(tfMatrix[i], tfMatrix[i + 1]) /
                                   (norm(tfMatrix[i]) * norm(tfMatrix[i + 1])) )
        cosineSimilatiryTFIDF.append( inner(tfidfMatrix[i], tfidfMatrix[i + 1]) /
                                      (norm(tfidfMatrix[i]) * norm(tfidfMatrix[i + 1])) )

    return cosineSimilatiryTF, cosineSimilatiryTFIDF

# Not used, it has no effect
def sortenMatrixes(tfMatrix):
    # sorten the matrixes
    indexesToRemove = list()
    for i in range(0, len(tfMatrix[1])):
        value = tfMatrix[0][i]
        if (tfMatrix[0][i] == value and tfMatrix[1][i] == value and tfMatrix[2][i] == value and tfMatrix[3][
            i] == value and tfMatrix[4][i] == value and
                tfMatrix[5][i] == value and tfMatrix[6][i] == value and tfMatrix[7][i] == value and tfMatrix[8][
                    i] == value and tfMatrix[9][i] == value and
                tfMatrix[10][i] == value and tfMatrix[11][i] == value and tfMatrix[12][i] == value and tfMatrix[13][
                    i] == value and
                tfMatrix[14][i] == value and tfMatrix[15][i] == value and tfMatrix[16][i] == value and tfMatrix[17][
                    i] == value and
                tfMatrix[18][i] == value and tfMatrix[19][i] == value):
            indexesToRemove.append(i)
            print(value)

    print(indexesToRemove)

def trainSVM(metricsVector1, metricsVector2, answers):
    # Combines the features of vector1 and vector2 into a common list of points [[x1,y1],[x2,y2], ..]
    mergeList = list()
    for i in range(0, len(metricsVector1)):
        mergeList.append([metricsVector1[i], metricsVector2[i]])

    # Create an svm Classifier
    SVMmodel = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    SVMmodel.fit(mergeList, answers)

    return SVMmodel

def trainKNN(metricsVector1, metricsVector2, answers):
    # Combines the features of vector1 and vector2 into a common list of points [[x1,y1],[x2,y2], ..]
    mergeList = list()
    for i in range(0, len(metricsVector1)):
        mergeList.append([metricsVector1[i], metricsVector2[i]])

    # Create a KNN Classifier
    KNNmodel= neighbors.KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training sets
    KNNmodel.fit(mergeList, answers)

    return KNNmodel

def testClassifier(metricsVector1, metricsVector2, classifier):
    # Combines the features of vector1 and vector2 into a common list of points [[x1,y1],[x2,y2], ..]
    mergeList = list()
    for i in range(0, len(metricsVector1)):
        mergeList.append([metricsVector1[i], metricsVector2[i]])

    result = classifier.predict(mergeList)

    return result

def evaluate(testAnswers, result):
    TP, FP = 0, 0
    # Calculate TP
    for i in range(0, len(testAnswers)):
        if testAnswers[i] == result[i] and result[i] == 1:
            TP += 1

    # Calculate FP
    for i in range(0, len(testAnswers)):
        if testAnswers[i] == 0 and result[i] == 1:
            FP += 1

    # Calculate Precision
    if result.count(1) == 0:
        precision = 0
    else:
        precision = TP / result.count(1)
    print("The precision of the model is: " + str(precision))

    # Calculate Recall
    if testAnswers.count(1) == 0:
        recall = 0
    else:
        recall = TP / testAnswers.count(1)
    print("The recall of the model is: " + str(recall))

    TPTN = 0
    # Calculate True Positives + True Negatives
    for i in range(0, len(testAnswers)):
        if testAnswers[i] == result[i]:
            TPTN += 1

    # Calculate Accuracy
    accuracy = TPTN / len(result)
    print("The accuracy of the model is: " + str(accuracy))

    # Calculate AUC
    auc = roc_auc_score(testAnswers, result)
    print("The auc of the model is: " + str(auc))

def printResults(testAnswers, result1, result2, result3, result4):
    results = [result1, result2, result3, result4]

    for i in range(0,len(results)):
        result = list(results[i])
        print("\n\n")
        if i == 0:
            print("######## SVM Euclidean Distance ########")
        elif i == 1:
            print("######## SVM Cosine Similarity ########")
        elif i == 2:
            print("######## KNN Euclidean Distance ########")
        elif i == 3:
            print("######## KNN Cosine Similarity ########")

        print("The test set's answers are: ")
        print(testAnswers)
        print("The model predicts the following classes: ")
        print(result)
        print()

        evaluate(testAnswers, result)

def main():
    # TRAIN DATA - steps 1,2

    # [ (txt1)[known, unknown], (txt2)[], ...] , {unique words of all txts}
    # txts, uniqueStemmedWords = iterateFiles("test", 20)
    txts, uniqueStemmedWords = iterateFiles("training", 10)

    # creates the tf and tf-idf matrixes
    tfMatrix = createTFmatrix(uniqueStemmedWords, txts)
    tfidfMatrix = createTFIDFMatrix(uniqueStemmedWords, txts)

    # calculates the euclidean distance matrixes of the text's vectors of the tf and the tf-idf matrixes
    euclideanTFDistance, euclideanTFIDFDistance = euclideanDistance(tfMatrix, tfidfMatrix)

    # calculates the cosine similatiry matrixes of the text's vectors of the tf and the tf-idf matrixes
    cosineSimilatiryTF, cosineSimilatiryTFIDF = cosineSimilatiry(tfMatrix, tfidfMatrix)

    # TEST DATA - steps 1,2

    # testtxts, testuniqueStemmedWords = iterateFiles("training", 10)
    testtxts, testuniqueStemmedWords = iterateFiles("test", 20)

    # creates the tf and tf-idf matrixes
    txttfMatrix = createTFmatrix(testuniqueStemmedWords, testtxts)
    txttfidfMatrix = createTFIDFMatrix(testuniqueStemmedWords, testtxts)

    # calculates the euclidean distance matrixes of the text's vectors of the tf and the tf-idf matrixes
    txteuclideanTFDistance, txteuclideanTFIDFDistance = euclideanDistance(txttfMatrix, txttfidfMatrix)

    # calculates the cosine similarity matrixes of the text's vectors of the tf and the tf-idf matrixes
    txtcosineSimilatiryTF, txtcosineSimilatiryTFIDF = cosineSimilatiry(txttfMatrix, txttfidfMatrix)

    # training and test data answers
    trainAnswers = [1,0,1,0,1,1,0,1,0,0]
    testAnswers = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
                  # [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    # END OF DATA PREPERATION
    # /////////////////////////////////////////////
    # /////////////////////////////////////////////

    # MODEL Training,Testing - step 3

    # Euclidean SVM
    SVMmodel = trainSVM(euclideanTFDistance, euclideanTFIDFDistance, trainAnswers)
    result1 = testClassifier(txteuclideanTFDistance, txteuclideanTFIDFDistance, SVMmodel)

    # Cosine SVM
    SVMmodel = trainSVM(cosineSimilatiryTF, cosineSimilatiryTFIDF, trainAnswers)
    result2 = testClassifier(txtcosineSimilatiryTF, txtcosineSimilatiryTFIDF, SVMmodel)

    # Euclidean KNN
    KNNmodel = trainKNN(euclideanTFDistance, euclideanTFIDFDistance, trainAnswers)
    result3 = testClassifier(txteuclideanTFDistance, txteuclideanTFIDFDistance, KNNmodel)

    # Cosine KNN
    KNNmodel = trainKNN(cosineSimilatiryTF, cosineSimilatiryTFIDF, trainAnswers)
    result4 = testClassifier(txtcosineSimilatiryTF, txtcosineSimilatiryTFIDF, KNNmodel)


    # Evaluation - step 4

    printResults(testAnswers, result1, result2, result3, result4)

if __name__ == '__main__':
    main()
