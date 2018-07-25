#from tensorflow.contrib.learn.python.learn.datasets import base
#from tensorflow.python.framework import dtypes
import feature as ft
import numpy as np

class ReaderTS(object):

	def __init__(self, numSteps, numDim, instXday, paddType, nChannels=1):
		self.numSteps = numSteps
		self.numDim = numDim
		self.instXday = instXday
		self.numChannels = nChannels
		self.paddType = paddType

	def generateDataSet( self):
		dirRoute = 'tctodd/tctodd'
		#dirRoute = '/home/yunli/Documents/Spring 2016/Data Mining/articles/australian language/tctodd/tctodd'
		
		rangeFolders = np.array([1,2,3,4,5,6,7,8,9])
		arrPalabras = np.array(['alive', 'all','answer','boy','building','buy','cold','change_mind_','come','computer_PC_','cost','crazy','danger','deaf','different','draw','drink','eat','exit','flash-light','forget','girl','give','glove','go','God','happy','head','hear','hello','his_hers','hot','how','hurry','hurt','I','innocent','is_true_','joke','juice','know','later','lose','love','make','man','maybe','mine','money','more','name','no','Norway','not-my-problem','paper','pen','please','polite','question','read','ready','research','responsible','right','sad','same','science','share','shop','soon','sorry','spend','stubborn','surprise','take','temper','thank','think','tray','us','voluntary','wait_notyet_','what','when','where','which','who','why','wild','will','write','wrong','yes','you','zero'])
		arrMatrixes = []
		arrClasses = []
		arrLeng = []
		totalWords = len(arrPalabras)
		for i in range(totalWords):
			for j in rangeFolders:
				for k in range(self.instXday):
					currentWord = arrPalabras[i]
					myFileName = dirRoute + str(j) + '/' + currentWord + '-' + str(k+1) + '.tsd'
					arrMatrix, lenMatrix = ft.feature.getMatrix(myFileName, self.numSteps, self.numDim, self.paddType)
					arrMatrixes.append(arrMatrix)
					arrLeng.append(lenMatrix)
					target = np.zeros((1,totalWords))
					target[0,i] = 1
					arrClasses.append(target)
		totalBatch = len(arrMatrixes)
		finalArrMatrixes = np.zeros([totalBatch, self.numSteps, self.numDim])
		finalArrClasses = np.zeros([totalBatch, totalWords])
		finalArrLens = np.zeros([totalBatch])
		for i in range(totalBatch):
			#if I did one more transpose outside, numSteps-arrLeng[i]
			finalArrMatrixes[i, 0: arrLeng[i],:] = arrMatrixes[i] #np.transpose(arrMatrixes[i])
			finalArrClasses[i] = arrClasses[i]
			finalArrLens[i] = arrLeng[i]
		return finalArrMatrixes, finalArrClasses, totalWords, arrPalabras,finalArrLens

	def generateDataSetCNN(self):
		dirRoute = 'tctodd/tctodd'
		#/home/gbejara/Documents/ASL/signLanguage/tctodd/tctodd
		#dirRoute = '/home/yunli/Documents/Spring 2016/Data Mining/articles/australian language/tctodd/tctodd'
		
		rangeFolders = np.array([1,2,3,4,5,6,7,8,9])
		arrPalabras = np.array(['alive', 'all','answer','boy','building','buy','cold','change_mind_','come','computer_PC_','cost','crazy','danger','deaf','different','draw','drink','eat','exit','flash-light','forget','girl','give','glove','go','God','happy','head','hear','hello','his_hers','hot','how','hurry','hurt','I','innocent','is_true_','joke','juice','know','later','lose','love','make','man','maybe','mine','money','more','name','no','Norway','not-my-problem','paper','pen','please','polite','question','read','ready','research','responsible','right','sad','same','science','share','shop','soon','sorry','spend','stubborn','surprise','take','temper','thank','think','tray','us','voluntary','wait_notyet_','what','when','where','which','who','why','wild','will','write','wrong','yes','you','zero'])
		arrMatrixes = []
		arrClasses = []
		arrLeng = []
		totalWords = len(arrPalabras)
		for i in range(totalWords):
			for j in rangeFolders:
				for k in range(self.instXday):
					currentWord = arrPalabras[i]
					myFileName = dirRoute + str(j) + '/' + currentWord + '-' + str(k+1) + '.tsd'
					arrMatrix, lenMatrix = ft.feature.getMatrix(myFileName, self.numSteps, self.numDim, self.paddType)
					arrMatrixes.append(arrMatrix)
					arrLeng.append(lenMatrix)
					target = np.zeros((1,totalWords))
					target[0,i] = 1
					arrClasses.append(target)
					#self.DS.appendLinked(ft.feature.getImageFeatureVector(myFileName, numSteps),i)
		totalBatch = len(arrMatrixes)
		finalArrMatrixes = np.zeros([totalBatch, self.numSteps, self.numDim, self.numChannels])
		finalArrClasses = np.zeros([totalBatch, totalWords])
		finalArrLens = np.zeros([totalBatch])
		for i in range(totalBatch):
			finalArrMatrixes[i,0:arrLeng[i],:,:] = np.reshape(arrMatrixes[i], (arrLeng[i], self.numDim, self.numChannels))
			finalArrClasses[i] = arrClasses[i]
			finalArrLens[i] = arrLeng[i]
		return finalArrMatrixes, finalArrClasses,  totalWords,arrPalabras,finalArrLens
		
