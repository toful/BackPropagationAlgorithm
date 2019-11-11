# BackPropagationAlgorithm

[![License](https://img.shields.io/github/license/toful/BackPropagationAlgorithm)](https://github.com/toful/BackPropagationAlgorithm)


Implementation of the Back Propagation Algorithm. Neural and Evolutionary Computation subject, MESIIA Master, URV 


## Build
Use the Makefile available in the src folder:
```
$ make
```

## Usage
```
./backPropagation.obj dataset_file #_epochs %_of_dataset_used_as_trainset learning_rate momentum #_hidden_layers [#_neurons_in_each_hidden_layer] 
```
e.g. Trainnig NN with a 5-neurons single hidden layer and 1-neuron output layer with the "data_test.txt" dataset:<br />
* **dataset_file** --> "data_test.txt" 
* **# epochs** --> 10000
* **% of_dataset_used_as_trainset** --> 80
* **learning_rate** --> 0.05
* **momentum** --> 0.25
* **# hidden_layers** --> 2 
* **[# neurons_in_each_hidden_layer]** --> 5 1

```
./backPropagation.obj "data_test.txt" 10000 80 0.05 0.25 2 5 1 
```
### Dataset file
The first line of the dataset file must contain the number of samples (lines) contained in the file, the number of features for each sample and the number of outputs. e.g. "data_test.txt":

	1: 100 2 1
	2: 1 1 1
	3: 1 2 2
	4: 1 3 3
	5: 1 4 4
	6: 1 5 5  
	...
	100: 10 9 90
	101: 10 10 100

The number of outputs indicated in the data file has to be the same than the number of neurons of the last hidden (output) layer specified as an input.

## Authors

* **Cristòfol Daudén Esmel** - [toful](https://github.com/toful)