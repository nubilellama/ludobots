### Ludobots core03
### Reddit user nubile_llama August 28, 2014

import random
#import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import math as mat

import config as cfg

###Compare desiredSynapseValues to last row of synapseValues

### Update synapseValues rows 1-9 using parent

### Loop through 1000 generations
###### Create child synapse matrix by copying and preturbing the parent synapse matrix
###### Set elements in first row of synapseValues to .5
###### Update synapseValues rows 1-9 using child
###### If childFitness is better than parentFitness, replace parent with child and
#######set parentFitness to childFitness value

# neurons have values
# synapses have weights

def EvolveNeuralNetwork():
    # Initialize parent, a matrix of synapse weights
    parent = MatrixCreate(10,10)
    parent = MatrixRandomize(parent)
    for n in range(0, cfg.numNeurons):
        parent[0,n] = 0.5
    
    
    # How many times should the HillClimber iterate?
    Iterations = 1000

    HillClimber(parent, Iterations)  
    
def HillClimber(parent, Iterations):
    
    # Define array to hold each parentFitness value, including initial value
    fitnessArray = []
    neuronValues = MatrixCreate(cfg.numUpdates, cfg.numNeurons)
            
    for n in range(0, cfg.numNeurons):
        neuronValues[0,n] = 0.5
    neuronValues = Update(neuronValues, parent)
    parentFitness = Fitness(parent) 
    
    PlotInitialNeuronValues(neuronValues)
    
    fitnessArray.append(parentFitness)
    
    # Run the neural network

    for currentGeneration in range(1000): 
        # Initialize child, which is a matrix that holds synapse weights
        child = MatrixPerturb(parent, 0.05)      
        
        neuronValues = Update(neuronValues, child)
        childFitness = Fitness(child)
        if childFitness > parentFitness:
            parent = child 
            parentFitness = childFitness

        fitnessArray.append(parentFitness)           


    PlotFitnessArray(fitnessArray)
    PlotFinalNeuronValues(neuronValues)

def PlotFitnessArray(fitnessArray):
    plt.close()
    plt.plot(fitnessArray, 'g')
    plt.ylim(0, 1)
    plt.xlim(0,1000)
    plt.savefig('f-FitnessOverGenerations.jpg')

def Fitness(synapseMatrix):
    neuronValues = MatrixCreate(cfg.numUpdates, cfg.numNeurons)
            
    for n in range(0, cfg.numNeurons):
        neuronValues[0,n] = 0.5
    
    desiredNeuronValues = VectorCreate(cfg.numNeurons)  
    for j in range(0, cfg.numNeurons, 2):
        desiredNeuronValues[0,j]=1
    
    nvArray = neuronValues[9,0:]
    normDist = Distance(desiredNeuronValues, nvArray)   
    
    f = 1.0 - normDist

    return f

def Distance(v1, v2):
    
    # Normalize the vector differences
    
    d = 0.0
    
    for i in range(cfg.numNeurons):
        d += (v2[0,i]-v1[0,i])**2
    if d != 0:
        d = d / mat.sqrt(d)
        normD = d / mat.sqrt(cfg.numNeurons) 
    else:
        normD = d
    return normD

def Fitness2(synapseMatrix):
 
    neuronValues = MatrixCreate(cfg.numUpdates, cfg.numNeurons)
            
    for n in range(0, cfg.numNeurons):
        neuronValues[0,n] = 0.5
        
    for i in range(0, cfg.numUpdates-1):
        neuronValues = Update(neuronValues, synapseMatrix, i)
        
    
    diff=0.0
    
    for i in range(0,9): 
    
        for j in range(0,9):
    
            diff=diff + abs(synapseMatrix[i,j]-synapseMatrix[i,j+1])
    
            diff=diff + abs(synapseMatrix[i+1,j]-synapseMatrix[i,j]) 
    
    diff=diff/(2*9*9)
    
    return diff, neuronValues
            
def PlotInitialNeuronValues(neuronValues):
    # Plot the initial synapse values 
    plt.close()
    plt.ylabel('Time')
    plt.xlabel('Neuron')
    plt.title('Initial Neuron Values')
    plt.imshow(neuronValues, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    plt.savefig('d-TestNN-Initial.jpg')
                

def PlotFinalNeuronValues(neuronValues):
    # Plot the final synapse values
    plt.close()
    plt.ylabel('Time')
    plt.xlabel('Neuron')
    plt.title('Final Neuron Values')
    plt.imshow(neuronValues, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    plt.savefig('e-TestNN-Final.jpg')   

def VectorCreate(width):
    v = npm.zeros((width), dtype='f')
    return v

def MatrixCreate(rows, columns):

    # Creates and returns a matrix with the given dimensions
    
    matrix = npm.zeros((rows,columns))
    return matrix

def MatrixRandomize(v):

    # Randomizes and returns a given vector
    
    for x in npm.nditer(v, op_flags=['readwrite']):
        x[...] = random.uniform(-1, 1)
    return v
  
def MatrixPerturb(parent,prob):

    # Influences changes in a given matrix with a given probability of change    
    child = parent.copy()
    x,y = child.shape
    for i in range(x):
        for j in range(y):
            if prob > random.random():
                child[i,j] = random.uniform(-1,1)
    return child
    
def Update(neuronValues, synapses):
    for i in range(0,cfg.numNeurons-1):
    
        # Update the ith row of the neuron values matrix
        for j in range(cfg.numUpdates):
            tempvalue = 0
            for k in range(cfg.numNeurons):
                tempvalue += synapses[j,k] * neuronValues[i,k]
            if tempvalue < -1:
                tempvalue = 0
            elif tempvalue > 1:
                tempvalue = 1
            neuronValues[i+1,j] = tempvalue

    return neuronValues
if __name__ == "__main__": 
    # Main method
    EvolveNeuralNetwork()
