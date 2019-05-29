
# coding: utf-8

# In[55]:


import sklearn
import numpy as np
from joblib import dump, load
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
import sklearn.naive_bayes as nb
import sklearn.svm as svm
import sklearn.neural_network as nn
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
from textblob import Word
import os
import json

from sklearn import preprocessing
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import nltk
from nltk import word_tokenize
from nltk.tag import StanfordPOSTagger

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


#Applies PosTag as a step in the pipeline
class PosTag(TransformerMixin):
    def __init__(self):
        pass
        
    def transform(self, X, *_):
        nltk.pos_tag(X)

        #Tagger files
        tagger="./stanford-postagger/models/spanish.tagger"
        jar="./stanford-postagger/stanford-postagger.jar"

        etiquetador=StanfordPOSTagger(tagger,jar)
        i=0

        XS = np.zeros((len(X),6))

        for tok in X:
            etiquetas=etiquetador.tag(word_tokenize(tok))
            e = ""
            for etiqueta in etiquetas:
                e = e + " " + etiqueta[1]
            
            #Combines the sentence itself with its POS tagging
            X[i] = X[i] + e
            
            '''XS[i][0] = e.count("ao0000") + e.count("aq0000") ###NUMERO DE ADJETIVOS
            XS[i][1] = e.count("nc0") + e.count("np0") ###NUMERO DE SUSTANTIVOS
            XS[i][2] = e.count("va") + e.count("vm") + e.count("vs") ###NUMERO DE VERBOS
            XS[i][3] = e.count("rg") + e.count("rn") ###NUMERO DE ADVERBIOS
            XS[i][4] = e.count("000000") ###NUMERO DE PRONOMBRES
            XS[i][4] = e.count("da0000") + e.count("dd0000") + e.count("de0000")
            + e.count("di0000") + e.count("dn0000") + e.count("do0000") 
            + e.count("dp0000") + e.count("dt0000") ###NUMERO DE DETERMINANTES'''
            
            i+=1
        
        return X
    
    def fit(self, *_):
        return self

#Method for the 'param' parameter of the trainModel method
def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("Error in param")

#Method for the 'param' parameter of the trainModel method
def autoconvert(s):
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s



class upmPolarity:
    def __init__(self):
        self.dictionary = {}
        self.model = ""
        self.testFile = ""
        self.classL = []
        
        
    def importDict(self,loc):
        """
        Imports a dictionary. The dictionary must be a textfile containing lines with: a word, a tab, and a numeric 
        value (positive or negative) stating the polarity of that word.

        Args:
            loc: location of the textfile.

        Returns:
            A confirmation that the dictionary was imported correctly
        """
        try:
            #Builds the dict
            with open(loc) as d:
                for line in d:
                    (key, val) = line.split()
                    self.dictionary[key] = val
                
        except Exception:  
            print ("Could not import dictionary\n")
        
        else:
            print ("Success importing the dictionary\n")
        
    
    
    def trainModel(self,loc,params,classifier="",saveLocation="", classes=""):
        """
        Imports labeled texts from a file, and creates a model by using them as training data. 
        The file must contain a text per line, in the format: text,polarity. As such, the text has to be
        already well preprocessed, and can't include any commas.

        Args:
            loc: location of the file with the training texts.
            params: location of the file with the params for the classifier. Depending on each classifier
                    (refer to Scikit Learn documentation), one or more parameters may be provided to tune
                    classification. The format for this params file is: parameterName:value.
                    
            classifier (optional): if given, the classifier used for training. The classifier can be one of these:
                                   'supportVectorMachine', 'multilayerPerceptron', 'decisionTree', 'sgd', 
                                   'naiveBayes', 'kNeighbors', 'nearestCentroid' or 'logisticRegression'. 
                                   If not given, the default classifier will be 'naiveBayes'.
            
            saveLocation (optional): if given, the location where the model will be saved
            
            classes: if given, the classes, in a list of strings format, wanted to be used for training the model

        Returns:
            A confirmation that the folder was imported correctly, and the created model if the user wants to save it.
        """
        try:

            paramD = {}
            
            #To obtain and format the params
            with open(params,'r') as infile:
                for line in infile:
                    (key,val) = line.split(",")
                    key = key.replace("\n","")
                    val = val.replace("\n","")
                    paramD[key] = autoconvert(val)
            
            #Classifier selection. If not given, defaults to Naive Bayes
            if(classifier == "supportVectorMachine"):
                clf = svm.SVC(**paramD)
            elif(classifier == "multilayerPerceptron"):
                clf = nn.MLPClassifier(**paramD)
            elif(classifier == "decisionTree"):
                clf = DecisionTreeClassifier(**paramD)
            elif(classifier == "sgd"):
                clf = SGDClassifier(**paramD)
            elif(classifier == "naiveBayes"):
                clf = nb.MultinomialNB(**paramD)
            elif(classifier == "kNeighbors"):
                clf = KNeighborsClassifier(**paramD)
            elif(classifier == "nearestCentroid"):
                clf = NearestCentroid(**paramD)
            elif(classifier == "logisticRegression"):    
                clf = LogisticRegression(**paramD)
            else:
                clf = nb.MultinomialNB(**paramD)              
            
            
            #For using just the classes selected in the corresponding parameter
            if(classes != ""):
                with open(loc,'r') as infile, open('./auxx.txt','a+') as outfile:
                    for line in infile:
                        (t,p) = line.split(",")
                        if(p.replace("\n","") in classes):
                            outfile.write(line)
                loc = "./auxx.txt"
            
            data = np.genfromtxt(loc, delimiter=',', dtype = None, encoding='latin-1')
            
            if(classes != ""):
                os.remove("auxx.txt")
                
            
            X = data[:,0].tolist()
            
            #Pipeline to have all the process in the same object and then save it as a model
            self.model = Pipeline([('PostT', PosTag()),
                                   ('tfidf', TfidfVectorizer(stop_words= None, decode_error="strict", ngram_range=(1,3), analyzer="word", norm="l2",sublinear_tf=False, use_idf=True)),
                                    ('clf',clf)])
            
            
            y = data[:,1]
            
            self.classL = np.unique(y)
            
            #Necessary to convert the classes to labels
            label_encoder = preprocessing.LabelEncoder()
            y = label_encoder.fit_transform(y)
            
            self.model.fit(X, y)        
            
            if(saveLocation != ""):
                #Saves the model as a .joblib
                dump(self.model, saveLocation + '/model.joblib') 
            
                
        except Exception as e:  
            print ("Could not import folder or compute training over it\n")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(e)
            print(exc_type, fname, exc_tb.tb_lineno)        
        
        else:
            print ("Success importing the folder and computing training over it\n")
                
   
      
    def importModel(self,loc,classes):
        """
        Imports a folder containing an already trained model for machine learning approach. The model must consist
        of a .joblib file including a sklearn generated classifier. Also, a classes list must be provided.

        Args:
            loc: location of the model.
            classes: list containing the names of all the classes included. If the model was generated using 
                     this framework, no particular order is needed for the list.

        Returns:
            A confirmation that the model was imported correctly
        """
        try:
            
            #Loads the model and the corresponding classes
            self.model = load(loc) 
            self.classL = np.unique(classes)
            
                
        except Exception:  
            print ("Could not import model\n")
            
        
        else:
            print ("Success importing the model\n")
            

    
            

    def polarity(self,locInput,locOutput=None,approach="machineL",locInverse=None,mode=""):
        """
        Uses polarity detection techniques to obtain the polarity of the test texts

        Args:
            locInput: the location of the input textfile
            locInput: the location of the output json file with the results. If no argument is given,
                      it is assumed that locInput is not a filename but a text to be directly analysed.
            approach: the type of technique to be used. "lexicon" needs a dictionary, "machineL" uses machine Learning
                      techniques from a model, and "mix" computes both approaches and return a computed average of 
                      the results.
            locInverse: the optional location of a textfile containing all words that should invert the next word's
                        polarity, for the lexicon approach. If given, every word included (one per line) in that
                        textfile will invert the polarity of the next word whose polarity != 0 (for example, the word
                        "no" is expected to be a polarity changer)
            mode: the way the test file will be interpreted, either "oneLine" or no value for usual line-per-line 
                      classification, or "paragraph" for considering each line of the file as a phrase from a paragraph,
                      thus classifying the full paragraph computing the majority vote for the full file (single line
                      classification is still also performed).

        Returns:
            The resulting polarities of each line in the selected file, or a print with the resulting polarity
            if no locOutput introduced
        """
        invList = []
        #Builds the invList for lexicon approach
        if (locInverse != None):
            with open(locInverse,'r') as infile:
                for line in infile:
                    line = line.replace("\n","")
                    invList.append(line)
        
        
        if (locOutput == None): ### TEXT MODE
            
            text = locInput
            try:
                #Starts building the json
                res = {}
                q = {}
                q['mode'] = "oneLine"
                q['approach'] = approach
                q['results'] = res
                
                line = text.replace("\n","")

                # Dictionary approach
                if approach == "lexicon":
                    
                    res['dictionary'] = []
                    #print("Computing lexicon based analysis\n")

                    d = self.dictionary
                    next = 1

                    blob = TextBlob(line)
                    pol = 0
                    for word in blob.words:
                        if word in invList:
                            #inverts polarity of next word whose polarity != 0
                            next = -1

                        elif word in d:
                            #adds polarity
                            pol = pol + int(d[word])*next*10
                            if int(d[word]) != 0:
                                next = 1

                    if(pol>0):
                        res['dictionary'].append({"sentence" : line.replace("\n",""),
                                                      "class": 'pos'})

                    elif(pol<0): 
                        res['dictionary'].append({"sentence" : line.replace("\n",""),
                                                      "class": 'neg'})

                    elif(pol==0): 
                        res['dictionary'].append({"sentence" : line.replace("\n",""),
                                                      "class": 'neu'})


                    if mode == "paragraph":
                        q['mode'] = "paragraph"


                    else:
                        print(json.dumps(q, indent=4, sort_keys=False))

                        
                # Machine Learning approach
                elif approach == "machineL":
                    res['machineLearning'] = []
                    #print("Computing machine learning based analysis\n")

                    #predicts using the model
                    predicted = self.model.predict([line])
                                 
                    res['machineLearning'].append({"sentence" : line.replace("\n",""),
                                                      "class": self.classL[predicted[0]]})
                   

                    if mode == "paragraph":
                        #q['mode'] = "paragraph"
                        print("Paragraph mode not allowed for text mode")

                    else:
                        print(json.dumps(q, indent=4, sort_keys=False))
                        

                # Dictionary + Machine Learning approach
                elif approach == "mix":
                    res['dictionary'] = []
                    res['machineLearning'] = []
                    #print("Computing both lexicon based and machine learning based analysis\n")

                    d = self.dictionary
                    next = 1

                    blob = TextBlob(line)
                    pol = 0
                    for word in blob.words:
                        if word in invList:
                            #inverts polarity of next word whose polarity != 0
                            next = -1

                        elif word in d:
                            #adds polarity
                            pol = pol + int(d[word])*next*10
                            if int(d[word]) != 0:
                                next = 1

                    if(pol>0):
                        res['dictionary'].append({"sentence" : line.replace("\n",""),
                                                      "class": 'pos'})

                    elif(pol<0): 
                        res['dictionary'].append({"sentence" : line.replace("\n",""),
                                                      "class": 'neg'})

                    elif(pol==0): 
                        res['dictionary'].append({"sentence" : line.replace("\n",""),
                                                      "class": 'neu'})


                    if mode == "paragraph":
                        q['mode'] = "paragraph"


                    #predicts using the model
                    predicted = self.model.predict([text])
                

                    res['machineLearning'].append({"sentence" : line.replace("\n",""),
                                                      "class": self.classL[predicted[0]]})
                        

                    if mode == "paragraph":
                        #q['mode'] = "paragraph"
                        print("Paragraph mode not allowed for text mode")

                    else:
                        print(json.dumps(q, indent=4, sort_keys=False))



                else:
                    print("Could not obtain polarity. Either a 'lexicon', 'machineL' or 'mix' parameter must be given'\n")

            except Exception as e:  
                print (e)

            else:
                pass

        
        else: ### FILE MODE
        
            try:
                if not (os.path.isfile(locInput)):
                    raise Exception()

                else:
                    self.testFile=locInput

            except Exception:  
                print ("Could not import file\n")


            try:
                #Starts building the json
                res = {}
                q = {}
                q['mode'] = "oneLine"
                q['approach'] = approach
                q['results'] = res
                count=[]
                
                # Dictionary approach
                if approach == "lexicon":
                    res['dictionary'] = []
                    
                    print("Computing lexicon based analysis\n")

                    d = self.dictionary
                    next = 1
                    

                    with open(self.testFile,'r') as infile, open(locOutput,"w+") as outfile:
                        for line in infile:
                            blob = TextBlob(line.replace("\n",""))
                            line = str(line)
                            pol = 0
                            for word in blob.words:
                                if word in invList:
                                    #inverts polarity of next word whose polarity != 0
                                    next = -1

                                elif word in d:
                                    #adds polarity
                                    pol = pol + int(d[word])*next*10
                                    if int(d[word]) != 0:
                                        next = 1

                            if(pol>0):
                                res['dictionary'].append({"sentence" : line.replace("\n",""),
                                                              "class": 'pos'})
                                count.append("pos")

                            elif(pol<0): 
                                res['dictionary'].append({"sentence" : line.replace("\n",""),
                                                              "class": 'neg'})
                                count.append("neg")
                            elif(pol==0): 
                                res['dictionary'].append({"sentence" : line.replace("\n",""),
                                                              "class": 'neu'}) 
                                count.append("neu")    

                        if mode == "paragraph":
                            unique,pos = np.unique(np.array(count),return_inverse=True)
                            counts = np.bincount(pos)
                            maxpos = counts.argmax()
                            
                            q['mode'] = "paragraph"
                            q['paragraphResults'] = {}
                            q['paragraphResults']['dictionary'] = []
                            q['paragraphResults']['dictionary'].append({"nPhrases" : len(count),
                                                                        "polarity" : unique[maxpos]})
                            
                        json.dump(q, outfile)


                # Machine Learning approach
                elif approach == "machineL":
                    res['machineLearning'] = []
                    print("Computing machine learning based analysis\n")


                    with open(self.testFile,'r') as f1, open(locOutput,"w+") as outfile:
                        for line in f1:

                            #predicts using the model
                            predicted = self.model.predict([line.replace("\n","")])

                            
                            res['machineLearning'].append({"sentence" : line.replace("\n",""),
                                                      "class": self.classL[predicted[0]]})
                            count.append(self.classL[predicted[0]]) 


                        if mode == "paragraph":
                            unique,pos = np.unique(np.array(count),return_inverse=True)
                            counts = np.bincount(pos)
                            maxpos = counts.argmax()
                            
                            
                            q['mode'] = "paragraph"
                            q['paragraphResults'] = {}
                            q['paragraphResults']['machineLearning'] = []
                            q['paragraphResults']['machineLearning'].append({"nPhrases" : len(count),
                                                                        "polarity" : unique[maxpos]})

                        
                        json.dump(q, outfile)

                # Dictionary + Machine Learning approach
                elif approach == "mix":
                    print("Computing both lexicon based and machine learning based analysis\n")


                    res['dictionary'] = []

                    d = self.dictionary
                    next = 1

                    with open(self.testFile,'r') as infile, open(locOutput,"w+") as outfile:
                        for line in infile:
                            blob = TextBlob(line.replace("\n",""))
                            pol = 0
                            for word in blob.words:
                                if (word in invList):
                                    next = -1

                                elif word in d:
                                    pol = pol + int(d[word])*next*10
                                    if pol != 0: next = 1

                            if(pol>0):
                                res['dictionary'].append({"sentence" : line.replace("\n",""),
                                                              "class": 'pos'})
                                count.append("pos")

                            elif(pol<0): 
                                res['dictionary'].append({"sentence" : line.replace("\n",""),
                                                              "class": 'neg'})
                                count.append("neg")
                            elif(pol==0): 
                                res['dictionary'].append({"sentence" : line.replace("\n",""),
                                                              "class": 'neu'}) 
                                count.append("neu")


                        if mode == "paragraph":
                            unique,pos = np.unique(np.array(count),return_inverse=True)
                            counts = np.bincount(pos)
                            maxpos = counts.argmax()
                            
                            q['mode'] = "paragraph"
                            q['paragraphResults'] = {}
                            q['paragraphResults']['dictionary'] = []
                            q['paragraphResults']['dictionary'].append({"nPhrases" : len(count),
                                                                        "polarity" : unique[maxpos]})
                            count=[]



                    res['machineLearning'] = []

                    with open(self.testFile,'r') as f1, open(locOutput,"w") as outfile:
                        for line in f1:

                            #predicts using the model
                            predicted = self.model.predict([line.replace("\n","")])

                            res['machineLearning'].append({"sentence" : line.replace("\n",""),
                                                      "class": self.classL[predicted[0]]})
                            count.append(self.classL[predicted[0]]) 
         
                        if mode == "paragraph":
                            unique,pos = np.unique(np.array(count),return_inverse=True)
                            counts = np.bincount(pos)
                            maxpos = counts.argmax()
                            
                            
                            q['mode'] = "paragraph"
                            #q['paragraphResults'] = {}
                            q['paragraphResults']['machineLearning'] = []
                            q['paragraphResults']['machineLearning'].append({"nPhrases" : len(count),
                                                                        "polarity" : unique[maxpos]})
                        
                        json.dump(q, outfile)


                else:
                    print("Could not obtain polarity. Either a 'lexicon', 'machineL' or 'mix' parameter must be given'\n")

            except Exception as e:  
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(e)
                print(exc_type, fname, exc_tb.tb_lineno)

            else:
                pass

