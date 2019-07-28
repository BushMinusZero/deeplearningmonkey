import os
import sys
from dnn import initialize_parameters
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

parameters = initialize_parameters(3,2,1)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))