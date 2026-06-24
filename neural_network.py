import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
from functions import fixed_digits, labels_to_tensor

class model(nn.Module):
    def __init__(self, structure, output_function):
        super().__init__() # because of inheritance
        layers = [] 

        for i in range(len(structure) - 1):  # n layers need n-1 Linear maps
            layers.append(nn.Linear(structure[i], structure[i + 1]))

        self.structure = structure
        self.layers = nn.ModuleList(layers)
        self.output_function = output_function
        self.pspace_dimensions = []

        for i in range(len(structure) - 1):
            self.pspace_dimensions.append(structure[i] * structure[i+1]) # size of weight matrix
            self.pspace_dimensions.append(structure[i+1]) # size of bias vector

        self.total_dimension = 0
        for i in self.pspace_dimensions:
            self.total_dimension += i
     
    def forward(self, x):

        for i in self.layers[:-1]: # for all but last transformation, we wanna relu
            x = F.relu(i(x))
        
        if self.output_function == "softmax":
            x = F.softmax(self.layers[-1](x), dim = 1)

        if self.output_function == "sigmoid":
            x = F.sigmoid(self.layers[-1](x), dim = 1)

        return x

def run_mnist_model(model, training_inputs, test_inputs, training_outputs, test_outputs,
                    test_labels,epochs, nabla, loss, opt, batch_size, info_at, plotting):
    
    training_size = training_inputs.shape[0]
    test_size = test_inputs.shape[0]
    iterations = []
    performance = []

    if loss == "MSE":
        loss_function = nn.MSELoss()

    if opt == "SGD":
        optimiser = optim.SGD(model.parameters(), lr=nabla) # Instantiate a new 'SGD' optimiser

    print(f"\nNeural Network Structure: {model.structure}")

    for epoch in range(epochs):
        print(f"\n----- Epoch {epoch} ----------------------")
        permutation = torch.randperm(training_size) # big random row vector (though in tensor form) - tensor([blah, blah...])
        training_inputs = training_inputs[permutation] 
        training_outputs = training_outputs[permutation]

        for batch in range(int(training_size / batch_size)): # now handful of batched steps per epoch
           
            x = training_inputs[(0 + batch * batch_size) : (batch_size + batch * batch_size)]
            y = training_outputs[(0 + batch * batch_size) : (batch_size + batch * batch_size)]

            nn_output = model.forward(x)
            loss_scalar = loss_function(nn_output, y)
            optimiser.zero_grad() 

            loss_scalar.backward()
            optimiser.step() 

            # Following gives progress reports as we run through epochs

            if (batch + epoch * (training_size / batch_size)) % info_at == 0:
                percentage = 0
                cost = 0

                for i in range(test_size):
                    x_test = test_inputs[i]
                    x_test = x_test.reshape(1, 784)
                    y_test = test_outputs[i]
                    y_test = y_test.reshape(1,10)
                    label_test = test_labels[i]
                    output_vector = model.forward(x_test)
                    cost += loss_function(output_vector, y_test)
                    guess = torch.argmax(output_vector) # From probability distribution, choose model's highest 'bet' on correct digit
                    guess = int(guess.item()) # Again, tensor -> int
            
                    if guess == label_test:
                        percentage += (100 / test_size)


                cost = cost / test_size
                print("\nPerformance:\n")
                print(f"{round(percentage, 2)} % | Steps Taken: {batch + epoch * (training_size / batch_size)} | Cost: {cost}\n")

                print("Parameter Spaces:\n")
                index = 0
                total_magnitude = 0
                for name, param in model.named_parameters():
                    if operator.contains(name, "weight"):
                        magnitude = float(torch.norm(param.grad))
                        print(f"PSpace: {name} | Step Magnitude: {fixed_digits(magnitude, 8)} | PSpace Dimension = {model.pspace_dimensions[index]}")
                        total_magnitude += magnitude ** 2 # since sums and square roots, undo square root to combine all together
                    else:
                        magnitude = float(torch.norm(param.grad))
                        print(f"PSpace: {name}   | Step Magnitude: {fixed_digits(magnitude, 8)} | PSpace Dimension = {model.pspace_dimensions[index]}")
                        total_magnitude += magnitude ** 2
                    index += 1
                
                total_magnitude = np.sqrt(total_magnitude)
                print(f"\n                        | Total Magnitude: {fixed_digits(total_magnitude, 7)} | Full Dimension = {model.total_dimension}")

                print("\n------------")

                # Storing Performance

                if plotting == "on":
                    iterations.append(batch + epoch * (60000 / batch_size))
                    performance.append(percentage)

    # Plotting Performance

    if plotting == "on":
        plt.plot(iterations, performance)
        plt.show()
    else:
        pass

    return model
    
