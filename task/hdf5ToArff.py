import numpy as np
import pandas as pd


def pandas2arff(df, filename, wekaname="pandasdata", cleanstringdata=True, cleannan=True):
    """
    converts the pandas dataframe to a weka compatible file
    df: dataframe in pandas format
    filename: the filename you want the weka compatible file to be in
    wekaname: the name you want to give to the weka dataset (this will be visible to you when you open it in Weka)
    cleanstringdata: clean up data which may have spaces and replace with "_", special characters etc which seem to annoy Weka.
                     To suppress this, set this to False
    cleannan: replaces all nan values with "?" which is Weka's standard for missing values.
              To suppress this, set this to False
    """
    import re

    def cleanstring(s):
        if s != "?":
            return re.sub('[^A-Za-z0-9]+', "_", str(s))
        else:
            return "?"

    dfcopy = df  # all cleaning operations get done on this copy

    if cleannan != False:
        dfcopy = dfcopy.fillna(-999999999)  # this is so that we can swap this out for "?"
        # this makes sure that certain numerical columns with missing values don't get stuck with "object" type

    f = open(filename, "w")
    arffList = []
    arffList.append("@relation " + wekaname + "\n")
    # look at each column's dtype. If it's an "object", make it "nominal" under Weka for now (can be changed in source for dates.. etc)
    for i in range(df.shape[1]):
        # print(i)
        if dfcopy.dtypes[i] == 'O' or (df.columns[i] in ["Class", "CLASS", "class"]):
            if cleannan != False:
                dfcopy.iloc[:, i] = dfcopy.iloc[:, i].replace(to_replace=-999999999, value="?")
            if cleanstringdata != False:
                dfcopy.iloc[:, i] = dfcopy.iloc[:, i].apply(cleanstring)
            _uniqueNominalVals = [str(_i) for _i in np.unique(dfcopy.iloc[:, i])]
            _uniqueNominalVals = ",".join(_uniqueNominalVals)
            # _uniqueNominalVals = _uniqueNominalVals.replace("[", "")
            # _uniqueNominalVals = _uniqueNominalVals.replace("]", "")
            _uniqueValuesString = "{" + _uniqueNominalVals + "}"
            arffList.append("@attribute " + df.columns[i] + _uniqueValuesString + "\n")
        else:
            arffList.append("@attribute " + df.columns[i] + " real\n")
            # even if it is an integer, let's just deal with it as a real number for now
    arffList.append("@data\n")
    for i in range(dfcopy.shape[0]):  # instances
        print(i)
        _instanceString = ""
        for j in range(df.shape[1]):  # features
            # print(j)
            if dfcopy.dtypes[j] == 'O':
                _instanceString += str(dfcopy.iloc[i, j])
            else:
                _instanceString += str(dfcopy.iloc[i, j])
            if j != dfcopy.shape[1] - 1:  # if it's not the last feature, add a comma
                _instanceString += ","
        _instanceString += "\n"
        if cleannan != False:
            _instanceString = _instanceString.replace("-999999999.0", "?")  # for numeric missing values
            _instanceString = _instanceString.replace("\"?\"", "?")  # for categorical missing values
        arffList.append(_instanceString)
    f.writelines(arffList)
    f.close()
    del dfcopy
    return True






# train_labeled = pd.read_hdf("train_labeled.h5", "train")
#
# for i in range(9000):
#     if train_labeled.loc[i,'y']==0:
#         train_labeled.loc[i, 'y'] = 'A'
#     elif train_labeled.loc[i,'y']==1:
#         train_labeled.loc[i, 'y'] = 'B'
#     elif train_labeled.loc[i,'y']==2:
#         train_labeled.loc[i, 'y'] = 'C'
#     elif train_labeled.loc[i,'y']==3:
#         train_labeled.loc[i, 'y'] = 'D'
#     elif train_labeled.loc[i,'y']==4:
#         train_labeled.loc[i, 'y'] = 'E'
#     elif train_labeled.loc[i,'y']==5:
#         train_labeled.loc[i, 'y'] = 'F'
#     elif train_labeled.loc[i,'y']==6:
#         train_labeled.loc[i, 'y'] = 'G'
#     elif train_labeled.loc[i,'y']==7:
#         train_labeled.loc[i, 'y'] = 'H'
#     elif train_labeled.loc[i,'y']==8:
#         train_labeled.loc[i, 'y'] = 'J'
#     elif train_labeled.loc[i,'y']==9:
#         train_labeled.loc[i, 'y'] = 'K'
#
# train_labeled=train_labeled.rename(columns = {'y':'class'})



# pandas2arff(train_labeled, "train_labeled5.arff")

# train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
# train_unlabeled.insert(0,"class",'A')
# pandas2arff(train_unlabeled, "train_unlabeled.arff")


test = pd.read_hdf("test.h5", "test")
test.insert(0,"class",'A')
pandas2arff(test, "test.arff")

print("finished")