import pandas as pd
import numpy as np
import math
from pprint import pprint

f = open('data.test.col', encoding="utf8")                                                       # Import Data
allData=[]
for x in f:
    line = x.strip().split(' ', 2)
    allData.append(line)                                                                        # Seperate and add each line to a list
f.close()
df = pd.DataFrame(allData, columns =['Tag', 'Spot', 'Sentence'])                                # Convert list to dataframe
df['Prep'] = df['Art/Adj'] = df['Pronoun'] = df['Verb/Conj'] = df['Or'] = ""#df['Or Not'] = ""  # Add feature columns
features = df[['Prep', 'Art/Adj','Pronoun','Verb/Conj','Or']]# ,'Or Not']

prepositions = ["About","Above","Abroad","Across","After","Against","Ago","Along","Amidst","Among","Amongst","Apart","Around","As","Aside","At","Away","Before","Behind","Below","Beneath","Beside","Besides","Between","Beyond","But","By","Despite","Down","During","Except","For","From","Hence","In","Inside","Into","Like","Near","Next","Notwithstanding","Of","Off","On","Onto","Opposite","Out","Outside","Over","Past","Per","Round","Since","Than","Through","Throughout","Till","Times","To","Toward","Towards","Under","Underneath","Unlike"] #prepositions2 = ["According to","Ahead of","Because of","Close to","Due to","In to","Next to","On to","Out from","Out of","Owing to","Prior to","Instead of"] #prepositions3 = ["As well as","As far as","By means of","In addition to","In case of","In front of","In lieu of","In place of","In spite of","In accordance with","On behalf of","On top of","On account of"]
articles = ["a","an", "the"]
adjectives = ["hot","cold","good","bad","sunny","cloudy","breezy", "bright","clear","damp","drizzly","dry","dull","foggy","hazy","rainy","showery","snowy","stormy","sunny","wet","miserable","pleasant","blistering","humid","difficult","inclement","calm","dry","warm","toasty","frigid","humid","windy","mild","damp","fair","cloudless","hazy","severe","intense","testy","scary","harmless"]
pronouns = ["he","she","to","I","we","they","you","it","that","someone","everyone","this","these","those","that","one","somebody","something","everybody","my","your","his","her","its","our","your","their","whose","each",]
verbs = ["is","was","being","be","has","had","having","does", "do","did","can","could","does","may","might","must","need","ought","shall","should","will","would","can","may","might","could","should","would","will","must"]
conjunctions = ["for","and","nor","but","yet","so"]

for index, row in df.iterrows():                                                                # For each sentence in DataFrame
    currentSpot = 0
    possible = False
    breakdown = row['Sentence'].split()
    for word in breakdown:                                                                      # For each word in that sentence
        if possible is True:                                                                    # If previous word was "or"
            possible= False
            if word.lower() == "not":   a=5#row['Or Not']=1                                     # And current word is "not"
            else:                       row['Or']=1                                             # And current word is not "not"
        if currentSpot == int(row['Spot'])-1:                                                   # If current word is the word before the blank
            for x in prepositions:                                                              # For each preposition
                if x.lower() == word.lower():
                    row['Prep']=1
                    break
            for y in articles:                                                                  # For each article
                if y.lower() == word.lower():
                    row['Art/Adj']=1
                    break
            for z in adjectives:                                                                # For each adjective
                if z.lower() == word.lower():
                    row['Art/Adj']=1
                    break
        if currentSpot == int(row['Spot'])+1:                                                   #If current word is the word after the blank
            for a in pronouns:                                                                  # For each pronoun
                if a.lower() == word.lower():
                    row['Pronoun']=1
                    break
            for b in articles:                                                                  # For each article
                if b.lower() == word.lower():
                    row['Pronoun']=1
                    break                    
            for c in verbs:                                                                     # For each Verb
                if c.lower() == word.lower():
                    row['Verb/Conj']=1
                    break
            for d in conjunctions:                                                              # For each conjunction
                if d.lower() == word.lower():
                    row['Verb/Conj']=1
                    break
            if ((word == ".") or (word == "-")):    row['Verb/Conj']=1                          # If the following word is a . or -
        if ((word.lower() == "or") and (currentSpot >= int(row['Spot'])+1)): possible = True    # If "or" occurs anywhere after the blank
        if row['Prep']!=1:      row['Prep']=0
        if row['Art/Adj']!=1:   row['Art/Adj']=0
        if row['Pronoun']!=1:   row['Pronoun']=0
        if row['Verb/Conj']!=1: row['Verb/Conj']=0
        if row['Or']!=1:        row['Or']=0
        #if row['Or Not']!=1:    row['Or Not']=0
        currentSpot+=1

def tagEntropy(inputDataFrame):                                                                 # Given a DataFrame, returns the entropy with respect to the tag column
    whetherNum = inputDataFrame['Tag'].value_counts()['whether']
    weatherNum = inputDataFrame['Tag'].value_counts()['weather']
    whetherProb = whetherNum / (whetherNum + weatherNum)
    weatherProb = weatherNum / (whetherNum + weatherNum)
    answer = -((whetherProb * math.log(whetherProb, 2)) + (weatherProb * math.log(weatherProb, 2)))
    return answer

def entropy(inputDataFrame, column, number):                                                    # Given the column of a specific dataframe, return its respective entropy
    attributeValue = inputDataFrame[column==number]                                             # Dataframe only containing rows of column feature specificied in input
    whetherCount = attributeValue['Tag'].value_counts()['whether']
    weatherCount = attributeValue['Tag'].value_counts()['weather']
    whetherProb = whetherCount / (whetherCount + weatherCount)
    weatherProb = weatherCount / (whetherCount + weatherCount)
    answer = -((whetherProb * math.log(whetherProb, 2)) + (weatherProb * math.log(weatherProb, 2)))
    return answer

def infoGain(inputDataFrame, column):
    zeroFeature = inputDataFrame[column==0]
    oneFeature = inputDataFrame[column==1]
    zeroCount = len(zeroFeature.index)
    oneCount = len(oneFeature.index)
    answer = tagEntropy(inputDataFrame) - ((zeroCount/(zeroCount+oneCount)) * entropy(inputDataFrame, column,0)) - ((oneCount/(zeroCount+oneCount)) * entropy(inputDataFrame, column,1))
    return answer

def bestInfoGain (inputDataFrame):                                                              # Of all features in input dataframe, return the feature with the highest info gain
    columns =[]
    values = [0.0]
    for column in inputDataFrame.columns:   columns.append(column)                              # Get All Column Names
    del columns[0:3]                                                                            # Remove non-feature columns
    del values[0:1]
    for g in columns:   
        values.append(infoGain(inputDataFrame,inputDataFrame[g]))                               # Cycle through each column, only starting from the feature columns       
    return columns[values.index(max(values))]

def buildTree(currentDF,inputDataFrame,featureList,parentDF=None, limiter=None):
    if (df['Tag'] == df['Tag'][0]).all():                                                       # Check if all tags in current df are the same
        return np.unique(df['Tag'])[0]                                                          # If so return those values via a leaf node
    elif (len(currentDF)==0 or (limiter == 0)):
        return np.unique(df['Tag'])[np.argmax(np.unique(inputDataFrame[df['Tag']],return_counts=True)[1])]
    else:                                                                                       # Otherwise start building tree
        limiter-=1
        parentDF = np.unique(df['Tag'])[np.argmax(np.unique(df['Tag'],return_counts=True)[1])]        
        featureChosen = bestInfoGain(currentDF)                                                 # Use this function to find highest info gain of features in current dataframe
        tree = {featureChosen:{}}                                                               # Assign that feature
        [ x for x in featureList if featureChosen not in x ]                                    # Remove that feature from list
        for y in range(0, 2):
            childDF = currentDF.where(currentDF[featureChosen] == x).dropna()                   # For both children, use same data minus the feature being used to split them
            childTree = buildTree(childDF,inputDataFrame,featureList,parentDF,limiter=None)
            tree[featureChosen][x] = childTree
        return(tree)
print("This is the entire .col file broken down into a dataframe with each lines corresponding features:")
print(df)
print("The entropy of the Pronoun feature being true is: ")
print(entropy(df, df["Pronoun"], 1))
print("The feature with the best information gain is: ")
print(bestInfoGain(df))