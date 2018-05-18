def splitClass(X2,labelsOfUnlabeled):
    import numpy as np
    classZero = np.empty([1, 128])
    classOne = np.empty([1, 128])
    classTwo = np.empty([1, 128])
    classThree = np.empty([1, 128])
    classFour = np.empty([1, 128])
    classFive = np.empty([1, 128])
    classSix = np.empty([1, 128])
    classSeven = np.empty([1, 128])
    classEight = np.empty([1, 128])
    classNine = np.empty([1, 128])

    for x in range(21000):
        if labelsOfUnlabeled[x] == 0:
            classZero = np.concatenate((classZero, np.reshape(X2[x,], [1, 128])), axis=0)
        elif labelsOfUnlabeled[x] == 1:
            classOne = np.concatenate((classOne, np.reshape(X2[x,], [1, 128])), axis=0)
        elif labelsOfUnlabeled[x] == 2:
            classTwo = np.concatenate((classTwo, np.reshape(X2[x,], [1, 128])), axis=0)
        elif labelsOfUnlabeled[x] == 3:
            classThree = np.concatenate((classThree, np.reshape(X2[x,], [1, 128])), axis=0)
        elif labelsOfUnlabeled[x] == 4:
            classFour = np.concatenate((classFour, np.reshape(X2[x,], [1, 128])), axis=0)
        elif labelsOfUnlabeled[x] == 5:
            classFive = np.concatenate((classFive, np.reshape(X2[x,], [1, 128])), axis=0)
        elif labelsOfUnlabeled[x] == 6:
            classSix = np.concatenate((classSix, np.reshape(X2[x,], [1, 128])), axis=0)
        elif labelsOfUnlabeled[x] == 7:
            classSeven = np.concatenate((classSeven, np.reshape(X2[x,], [1, 128])), axis=0)
        elif labelsOfUnlabeled[x] == 8:
            classEight = np.concatenate((classEight, np.reshape(X2[x,], [1, 128])), axis=0)
        elif labelsOfUnlabeled[x] == 9:
            classNine = np.concatenate((classNine, np.reshape(X2[x,], [1, 128])), axis=0)

    classes = [classZero,classOne,classTwo,classThree,classFour,classFive,classSix,classSeven,classEight,classNine]

    return classes