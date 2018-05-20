from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn import metrics


data = arff.loadarff(open('result','r'))

df = pd.DataFrame(data[0])
print(df.loc[1,'class'].decode("utf-8") )

# {A,B,C,D,E,F,G,H,J,K}
result = np.empty([8000],dtype=int)
for i in range(8000):
    print(i)
    # result[i] = i
    if ("A" == df.loc[i,'class'].decode("utf-8")):
        print("here")
        # result = np.append(result,0)
        result[i] = int(0)
    elif (df.loc[i,'class'].decode("utf-8")=="B"):
        # result = np.append(result,1)
        result[i] = int(1)
    elif (df.loc[i,'class'].decode("utf-8")=="C"):
        # result = np.append(result,2)
        result[i] = int(2)
    elif (df.loc[i,'class'].decode("utf-8")=="D"):
        # result = np.append(result,3)
        result[i] = int(3)
    elif (df.loc[i,'class'].decode("utf-8")=="E"):
        # result = np.append(result,4)
        result[i] = int(4)
    elif (df.loc[i,'class'].decode("utf-8")=="F"):
        # result = np.append(result,5)
        result[i] = int(5)
    elif (df.loc[i,'class'].decode("utf-8")=="G"):
        # result = np.append(result,6)
        result[i] = int(6)
    elif (df.loc[i,'class'].decode("utf-8")=="H"):
        # result = np.append(result,7)
        result[i] = int(7)
    elif (df.loc[i,'class'].decode("utf-8")=="J"):
        # result = np.append(result,8)
        result[i] = int(8)
    elif (df.loc[i,'class'].decode("utf-8")=="K"):
        # result = np.append(result,9)
        result[i] = int(9)

benchmark =np.loadtxt(open('HugoBenchMark.csv'),delimiter = ",", skiprows = 0, usecols = (0))

print(metrics.accuracy_score(benchmark,result))

# np.savetxt("triTrainResultOne.csv",result,delimiter=",")