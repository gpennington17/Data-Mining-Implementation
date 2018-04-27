Data-Mining-Implementation ReadMe:

Our algorthims are found on 3 seperate files 
this document provides instructions for how to test each:

Decision Tree Classifier (ID3):


Files Needed:
Nodes.py
DCT.py

To Test run this file (in jupyter notebook):
Testing_DCT.ipynb

To allow for user inputed data preprocessing is needed, some
examples of this is already shown in the jupyter notebook.

A rough outline of what is needed:
1. Download (or open) the dataset and put it into a dataframe
2. Make the following variables:
	Attbs = A list of the attribute (column) names
	Labels = A list of possible labels (The file calls df['class'].unique() to get this)
	X = Dataset tuples
	y = Corresponding Labels
	
3. to build the classifier object:
	Classifier = DCT.DCT_classifer(Attbs,Labels)
4. To fit call:
	Classifier.fit(X,y)
5. To show the tree:
	Classifier.show()
6. To Predict a Tuple (or set of tuples = Test_X)
	Classifier.predict(Test_X)
7. To clear the tree (for a future build)
	Classifier.clear()
8. To see the tree over 10-fold: (X=all tuples, y= all corresponding labels, Classifer)
	Test_Classifier(X,y,Classifier)
9. To see the accuracy over 10-fold: (X=all tuples, y= all corresponding labels, Classifer)
	Score(X,y,test)

Notes:
	Only works for CATAGORICAL data.
	





	
	
	