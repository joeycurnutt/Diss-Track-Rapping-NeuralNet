# Diss-Track-Rapping-NeuralNet
Rapping Neural Net almost entirely based off of Robbie Barrat's rapping neural network

This is a neural network made to emulate the style of Eminem. It was originally designed because of a diss-track we were writing.
We inevitably ran out of roasts and God knows Eminem has plenty, so I decided to rework Robbie Barat's neural network after watching a documentary about him.
All credit goes to him for a majority of the code, so I recommend checking out his repository.

Requirements:
*Python 3.x, not 3.10
*pronouncing
*markovify
*numpy
*tensorflow
*Keras

All of these can be installed using "pip install "

Training the program:
Once all of the dependencies are installed, make sure you have training_mode set to True, and modify the depth to your liking
Run using "python documented_model.py" in the command line of your choice (running in the directory with your file, of course)
This will write rhymes to a rhyme file and train the network to write a rap in the next step.

Generating a rap:
Set training_mode to False, and this will write your rap to a file. Enjoy!

WARNING:
This code is still a work in progress
