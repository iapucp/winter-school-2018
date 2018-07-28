import numpy as np

class feature:
	#Return a 10X10 array from a monochrome BMP Image
	@staticmethod
	def getImageArray(imagepath):
		try:
			f = open(imagepath)
			l = []
			l = [ line.split() for line in f]
			imagearray=np.asarray(l).astype(float)
			return imagearray
		except IOError:
			print ("File Not Found " +  imagepath)
			return np.zeros((10,10))

	# Returns a Normalized 2D Array from an Input Array. 
	# normalizeArrayCell is called for all non zero cells
	# paddType (0 complete with zeros, 1 complete with last values, 2, dont complete)
	@staticmethod
	def getNormalizedMatrix( imagearray, numSteps, numDim, paddType):
		inputMatrix = np.transpose(imagearray)
		mins = np.amin(inputMatrix, axis=1)
		maxs = np.amax(inputMatrix, axis=1)		
		for minRow, maxRow,i in zip(mins, maxs, range(inputMatrix.shape[0])):
			currentFeature = inputMatrix[i].nonzero()
			if minRow == maxRow:
				inputMatrix[i] = inputMatrix[i]*0
				#print i
			else:
				inputMatrix[i] = (inputMatrix[i] - minRow)/(maxRow - minRow)
			#feature.normalizeArrayCell(imagearray, nonzerocells[i][0],nonzerocells[i][1])
		iLastStep = inputMatrix.shape[1]
		if (iLastStep < numSteps and paddType ==1):
			vLastStep = np.reshape(inputMatrix[:,iLastStep-1],(inputMatrix.shape[0],1))
			additional = np.repeat(vLastStep, numSteps-iLastStep,axis=1)
			inputMatrix = np.append(inputMatrix,additional,axis=1)
			iLastStep = numSteps
		else:
			if (iLastStep > numSteps):
				inputMatrix = inputMatrix[:,0:numSteps]
				iLastStep = numSteps
		return inputMatrix[0:numDim,:], iLastStep

	# Returns Normalized 2D array from an Image
	@staticmethod
	def getMatrix( imagepath, numSteps, numDim, flgPadd):
		#matrix, lenSeq = feature.getNormalizedMatrix(feature.getImageArray(imagepath), numSteps, numDim, flgPadd)
		matrix = feature.getImageArray(imagepath)
		return matrix, matrix.shape[0]
