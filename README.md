# flask-nl-emocon

Using Bert NLP model, convert an input of text to an emocon that represent the emotion of the sentence.

# How to use

The stack contains a pre-trained model that is ready to use.

First create a python environment using the requirements.txt. Then cd to ./src and run `flask run` to start the program. When the program is ready, use your favorite browser and navigate to http://localhost:5000/. It will be the index page and allow user to enter their prompt.

./src/app.py: Define the flask app. Contains three route
"/", GET : The root, index page. For user to eneter prompt
"/api/nlplab", POST: Url for posting the input data. Return predicted emocon coresponding to the predicted emotion.
"/test": Return the accuracy of model for validation set. Implemented for testing the model.

./src/usrlib/nlp.py: Contains functions related to the Bert classification.
"convert_input_to_torch_format": function to convert the user input into tokenized id and attention mask.
"get_emocon_for_input": Function to convert user input to coresponding emocon.


For training and fine tuning, use model.ipynb. See the file for further instruction.