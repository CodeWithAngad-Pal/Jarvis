import speech_recognition as sr    # pip install SpeechRecognition

# Initialize recognizer
recognizer = sr.Recognizer()

# Use the microphone as the audio source
with sr.Microphone() as source:
    print("Say something...")
    audio = recognizer.listen(source)

# Recognize speech using Google Speech Recognition
try:
    print("You said: " + recognizer.recognize_google(audio))
except sr.UnknownValueError:
    print("Sorry, I could not understand the audio")
except sr.RequestError:
    print("Could not request results from Google Speech Recognition service")

#-------------------------------------------------------------------------------------------------

import pyttsx3          #  pip install pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Set properties (optional)
engine.setProperty('rate', 150)  # Speed of speech (higher is faster)
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Speak the text
engine.say("Hello, how are you?")

# Wait until the speech is finished
engine.runAndWait()

#----------------------------------------------------------------------

import datetime

# Get the current date and time
now = datetime.datetime.now()
print("Current date and time:", now)

# Get only the current date
current_date = now.date()
print("Current date:", current_date)

# Get only the current time
current_time = now.time()
print("Current time:", current_time)

# Get the current year, month, day, hour, minute, second
print("Year:", now.year)
print("Month:", now.month)
print("Day:", now.day)
print("Hour:", now.hour)
print("Minute:", now.minute)
print("Second:", now.second)

# Format the current date and time in a specific format
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print("Formatted date and time:", formatted)

# Create a specific date and time
specific_date = datetime.datetime(2024, 12, 25, 10, 30)
print("Specific date and time:", specific_date)

# Add 7 days to the current date
future_date = now + datetime.timedelta(days=7)
print("Date after 7 days:", future_date)

# Subtract 7 days from the current date
past_date = now - datetime.timedelta(days=7)
print("Date 7 days ago:", past_date)

#--------------------------------------------------------------------

import os
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

os.chdir('/path/to/directory')  # Replace with your desired directory path
print("Changed to directory:", os.getcwd())

files_and_dirs = os.listdir()
print("Files and directories in current directory:", files_and_dirs)

os.mkdir('new_folder')  # Creates a new directory called 'new_folder'
print("New directory created")

os.rmdir('new_folder')  # Removes the directory called 'new_folder'
print("Directory removed")

if os.path.exists('file_or_folder'):
    print("The file or directory exists")
else:
    print("The file or directory does not exist")

file_path = 'example.txt'
if os.path.exists(file_path):
    file_size = os.path.getsize(file_path)
    print(f"File size of {file_path} is {file_size} bytes")

os.rename('old_name.txt', 'new_name.txt')
print("File renamed")

os.remove('file_to_delete.txt')  # Deletes the specified file
print("File deleted")

os.system('echo Hello, world!')  # Runs a command in the system shell

# Get an environment variable
user_home = os.getenv('HOME')  # On Unix-based systems
print("User home directory:", user_home)

# Set an environment variable
os.environ['MY_VAR'] = 'some_value'
print("MY_VAR set to:", os.getenv('MY_VAR'))

# Join paths
file_path = os.path.join('folder', 'subfolder', 'file.txt')
print("Full file path:", file_path)

# Split a path into directory and file
directory, filename = os.path.split(file_path)
print("Directory:", directory)
print("Filename:", filename)

#--------------------------------------------------------------------------------

import sys
python_version = sys.version
print("Python version:", python_version)

print("Command-line arguments:", sys.argv)

sys.exit("Exiting the program")

platform = sys.platform
print("Platform:", platform)

recursion_limit = sys.getrecursionlimit()
print("Recursion limit:", recursion_limit)

# Set a new recursion limit (be cautious with this)
sys.setrecursionlimit(1500)

# Print to standard output
sys.stdout.write("Hello, World!\n")

# Print to standard error
sys.stderr.write("This is an error message\n")

with open('output.txt', 'w') as f:
    sys.stdout = f
    print("This will be written to the file.")
    sys.stdout = sys.__stdout__  # Reset back to the default stdout

size = sys.getsizeof("Hello")
print("Size of the string 'Hello':", size, "bytes")

sys.path.append('/path/to/your/module')
import your_module  # Now you can import the module

# ---------------------------------------------------------------------------------

import random

# Random integer between 1 and 10 (inclusive)
random_integer = random.randint(1, 10)
print("Random integer:", random_integer)

# Random float between 0 and 1
random_float = random.random()
print("Random float:", random_float)

# Random float between 1 and 10
random_float_range = random.uniform(1, 10)
print("Random float in range:", random_float_range)

items = ['apple', 'banana', 'cherry', 'date']
random_item = random.choice(items)
print("Random item:", random_item)

# Randomly shuffle the list in place
random.shuffle(items)
print("Shuffled list:", items)

# Randomly select 2 unique items from the list
random_sample = random.sample(items, 2)
print("Random sample:", random_sample)

# Random boolean value (True or False)
random_boolean = random.choice([True, False])
print("Random boolean:", random_boolean)

# Random number from a normal distribution with mean 0 and standard deviation 1
random_gaussian = random.gauss(0, 1)
print("Random Gaussian number:", random_gaussian)

import string

# Randomly generate a password of 8 characters
password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
print("Random password:", password)

# Randomly select an item from a list with weights
items = ['apple', 'banana', 'cherry', 'date']
weights = [0.1, 0.3, 0.4, 0.2]  # Probabilities for each item
random_weighted_item = random.choices(items, weights=weights, k=1)[0]
print("Random weighted item:", random_weighted_item)

#-----------------------------------------------------------------------------------------------

import requests          # pip install requests

# Sending a GET request to a URL
response = requests.get('https://api.github.com')

# Print the status code and response content
print("Status Code:", response.status_code)
print("Response Content:", response.text)  # Returns the content of the response

# Sending a GET request with parameters
params = {'q': 'Python', 'sort': 'stars'}
response = requests.get('https://api.github.com/search/repositories', params=params)

# Print the status code and response content
print("Status Code:", response.status_code)
print("Response Content:", response.json())  # Returns the response as JSON

# Sending a POST request with JSON data
url = 'https://httpbin.org/post'
data = {'name': 'John', 'age': 30}
response = requests.post(url, json=data)

# Print the status code and response content
print("Status Code:", response.status_code)
print("Response Content:", response.json())  # Response in JSON format

# Sending a POST request with form data
url = 'https://httpbin.org/post'
data = {'name': 'Alice', 'age': 25}
response = requests.post(url, data=data)

# Print the status code and response content
print("Status Code:", response.status_code)
print("Response Content:", response.json())

response = requests.get('https://httpbin.org/status/404')

# Check if the request was successful
if response.status_code == 200:
    print("Request was successful.")
else:
    print(f"Request failed with status code {response.status_code}.")

# Sending a GET request with custom headers
headers = {'User-Agent': 'my-app'}
response = requests.get('https://httpbin.org/headers', headers=headers)

# Print the status code and response content
print("Status Code:", response.status_code)
print("Response Content:", response.json())

try:
    response = requests.get('https://httpbin.org/delay/5', timeout=3)
    print("Response:", response.text)
except requests.Timeout:
    print("The request timed out.")

try:
    response = requests.get('https://nonexistentwebsite.com')
except requests.exceptions.RequestException as e:
    print("Request failed:", e)

url = 'https://httpbin.org/put'
data = {'name': 'John', 'age': 31}
response = requests.put(url, json=data)

# Print the status code and response content
print("Status Code:", response.status_code)
print("Response Content:", response.json())

url = 'https://httpbin.org/delete'
response = requests.delete(url)

# Print the status code and response content
print("Status Code:", response.status_code)
print("Response Content:", response.json())

#----------------------------------------------------------------------------

import nltk          #  pip install nltk

nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For removing stopwords
nltk.download('averaged_perceptron_tagger')  # For POS tagging
nltk.download('movie_reviews')  # For classification task
nltk.download('wordnet')  # For WordNet
nltk.download('maxent_ne_chunker')  # For Named Entity Recognition
nltk.download('words')  # For Named Entity Recognition

from nltk.tokenize import sent_tokenize

text = "Hello! How are you? I'm fine, thank you."
sentences = sent_tokenize(text)
print("Sentence Tokenization:", sentences)

from nltk.tokenize import word_tokenize

words = word_tokenize(text)
print("Word Tokenization:", words)

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print("Filtered Words:", filtered_words)

from nltk.stem import PorterStemmer

ps = PorterStemmer()
words = ['running', 'ran', 'easily', 'fairly']
stemmed_words = [ps.stem(word) for word in words]
print("Stemmed Words:", stemmed_words)

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in words]
print("Lemmatized Words:", lemmatized_words)

from nltk import pos_tag

tagged_words = pos_tag(words)
print("POS Tagging:", tagged_words)

from nltk import ne_chunk

tree = ne_chunk(tagged_words)
print("Named Entity Recognition:", tree)

from nltk.probability import FreqDist

fdist = FreqDist(filtered_words)
print("Word Frequency Distribution:", fdist)
fdist.plot()  # Visualizes the word frequency distribution

from nltk.text import Text

text = Text(filtered_words)
text.concordance('running')  # Shows the context in which 'running' appears

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

bigram_finder = BigramCollocationFinder.from_words(filtered_words)
collocations = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 5)
print("Collocations:", collocations)

from nltk.corpus import wordnet as wn

# Get synonyms
synonyms = wn.synsets('good')
print("Synonyms of 'good':", [syn.name() for syn in synonyms])

# Get definitions
print("Definitions of 'good':", [syn.definition() for syn in synonyms])

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import random

# Create a feature extractor
def extract_features(words):
    return {word: True for word in words}

# Prepare the training data
positive_reviews = movie_reviews.fileids('pos')
negative_reviews = movie_reviews.fileids('neg')
positive_data = [(extract_features(movie_reviews.words(fileid)), 'pos') for fileid in positive_reviews]
negative_data = [(extract_features(movie_reviews.words(fileid)), 'neg') for fileid in negative_reviews]
train_data = positive_data + negative_data
random.shuffle(train_data)

# Train the classifier
classifier = NaiveBayesClassifier.train(train_data)

# Test the classifier
test_review = "This movie was fantastic!"
test_features = extract_features(test_review.split())
print("Classification:", classifier.classify(test_features))

from gensim.models import Word2Vec            #  pip install gensim

# Example corpus of sentences
sentences = [["I", "love", "programming"], ["Python", "is", "awesome"]]

# Train Word2Vec model
model = Word2Vec(sentences, min_count=1)

# Get vector representation of a word
vector = model.wv['programming']
print("Vector Representation of 'programming':", vector)

from nltk import RegexpParser

# Define a simple grammar
grammar = "NP: {<DT>?<JJ>*<NN>}"

# Create a parser
parser = RegexpParser(grammar)

# Sample sentence
sentence = [("the", "DT"), ("big", "JJ"), ("dog", "NN")]

# Parse the sentence
tree = parser.parse(sentence)
tree.draw()  # Visualizes the parse tree

from nltk import bigrams, trigrams

text = "This is an example sentence."
words = word_tokenize(text)

# Bigram
bigrams_list = list(bigrams(words))
print("Bigrams:", bigrams_list)

# Trigram
trigrams_list = list(trigrams(words))
print("Trigrams:", trigrams_list)

from nltk.corpus import reuters
from nltk import FreqDist

# Fetch reuters text
words = reuters.words()
fdist = FreqDist(words)

# Get the most common words
common_words = fdist.most_common(10)
print("Most common words:", common_words)



from nltk.chat.util import Chat, reflections

# Define a list of patterns and responses
patterns = [
    (r"hi|hello", ["Hello, how can I help you?"]),
    (r"my name is (.*)", ["Hello %1, how can I assist you today?"]),
    (r"what is your name?", ["I am a chatbot created by NLTK."]),
    (r"how are you?", ["I'm doing great, thank you! How can I help you?"]),
    (r"quit", ["Goodbye! Have a nice day!"]),
]

# Create a chatbot
chatbot = Chat(patterns, reflections)

# Start the conversation
chatbot.converse()

# --------------------------------------------------------------------------

import pandas as pd              #  pip install pandas

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 27, 22, 32],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)
print(df)

df = pd.read_csv('file.csv')
print(df)

print(df.head())  # Default is 5 rows

print(df.tail())  # Default is 5 rows

print(df.describe())

print(df.info())

print(df['Name'])

print(df[['Name', 'Age']])

print(df.iloc[2])  # Row 2 (third row)

print(df.at[2, 'Age'])  # Age of the person in the third row

filtered_df = df[df['Age'] > 25]
print(filtered_df)

filtered_df = df[(df['Age'] > 25) & (df['City'] == 'Los Angeles')]
print(filtered_df)

df['Country'] = ['USA', 'USA', 'USA', 'USA']
print(df)

df['Age'] = df['Age'] + 1  # Increase age by 1
print(df)

print(df.isnull())

df_cleaned = df.dropna()  # Drops rows with any NaN values
print(df_cleaned)

df_filled = df.fillna(0)  # Replace NaN with 0
print(df_filled)

df_sorted = df.sort_values(by='Age', ascending=False)
print(df_sorted)

df_sorted = df.sort_values(by=['City', 'Age'], ascending=[True, False])
print(df_sorted)

grouped = df.groupby('City')['Age'].mean()
print(grouped)

df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'Age': [24, 27, 22]})

merged_df = pd.merge(df1, df2, on='ID', how='inner')  # 'inner' join by default
print(merged_df)

df.to_csv('output.csv', index=False)  # Save DataFrame to a CSV file without row index


import matplotlib.pyplot as plt

df['Age'].plot(kind='bar')  # Bar plot of the 'Age' column
plt.show()

df.plot(kind='scatter', x='Age', y='Name')
plt.show()

#---------------------------------------------------------------------------------------------------------

import numpy as np        # pip install numpy

arr = np.array([1, 2, 3, 4, 5])
print(arr)

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d)

arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr_3d)

zeros = np.zeros((3, 3))  # Create a 3x3 matrix of zeros
print(zeros)

ones = np.ones((2, 4))  # Create a 2x4 matrix of ones
print(ones)

identity = np.eye(4)  # Create a 4x4 identity matrix
print(identity)

arr = np.array([1, 2, 3, 4, 5])
print(arr.shape)  # Output: (5,)

arr_reshaped = arr.reshape((1, 5))  # Convert the array to 1 row, 5 columns
print(arr_reshaped)

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_flattened = arr_2d.flatten()  # Flatten to 1D
print(arr_flattened)

arr = np.array([1, 2, 3, 4])
arr_plus_2 = arr + 2  # Add 2 to each element
print(arr_plus_2)

arr_mult = arr * 2  # Multiply each element by 2
print(arr_mult)

arr_exp = np.power(arr, 2)  # Square each element
print(arr_exp)

arr_sum = np.sum(arr)  # Sum of all elements
print(arr_sum)

arr_mean = np.mean(arr)  # Mean of elements
print(arr_mean)

arr_std = np.std(arr)  # Standard deviation
print(arr_std)

arr = np.array([1, 2, 3, 4, 5])
print(arr[2])  # Access element at index 2 (third element)

arr_slice = arr[1:4]  # Get elements from index 1 to 3 (4 is not included)
print(arr_slice)

arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr_2d[1, 2])  # Access element at second row, third column (5)
print(arr_2d[:, 1])  # Access all rows, second column

angles = np.array([0, np.pi/2, np.pi])
sin_vals = np.sin(angles)
print(sin_vals)

arr = np.array([1, 2, 3])
exp_vals = np.exp(arr)  # e^x for each element
log_vals = np.log(arr)  # natural log for each element
print(exp_vals)
print(log_vals)

random_values = np.random.rand(3, 3)  # 3x3 array of random values between 0 and 1
print(random_values)

random_ints = np.random.randint(1, 10, size=(2, 3))  # 2x3 array of random integers between 1 and 10
print(random_ints)

arr = np.array([1, 2, 3, 4, 5])
random_choice = np.random.choice(arr, 3)  # Choose 3 random values from the array
print(random_choice)

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concatenated = np.concatenate((arr1, arr2))
print(concatenated)

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
concatenated_axis0 = np.concatenate((arr1, arr2), axis=0)  # Vertically (row-wise)
concatenated_axis1 = np.concatenate((arr1, arr2), axis=1)  # Horizontally (column-wise)
print(concatenated_axis0)
print(concatenated_axis1)

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
matrix_multiplication = np.dot(arr1, arr2)
print(matrix_multiplication)

det = np.linalg.det(arr1)
print(det)

inverse = np.linalg.inv(arr1)
print(inverse)

np.save('array.npy', arr)  # Save array to a .npy file

arr_loaded = np.load('array.npy')  # Load array from .npy file
print(arr_loaded)

#-------------------------------------------------------------------------------------------------------

import torch       #  pip install torch

tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(tensor_1d)

tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor_2d)

tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor_3d)

zeros = torch.zeros((3, 3))  # Create a 3x3 tensor of zeros
print(zeros)

ones = torch.ones((2, 4))  # Create a 2x4 tensor of ones
print(ones)

random_tensor = torch.rand((3, 3))  # Create a 3x3 tensor with random values between 0 and 1
print(random_tensor)

tensor = torch.tensor([1, 2, 3, 4, 5])
tensor_add = tensor + 2  # Add 2 to each element
print(tensor_add)

tensor_mult = tensor * 2  # Multiply each element by 2
print(tensor_mult)

tensor_exp = torch.pow(tensor, 2)  # Square each element
print(tensor_exp)

tensor_sum = torch.sum(tensor)  # Sum of all elements
print(tensor_sum)

tensor_mean = torch.mean(tensor.float())  # Mean of elements (convert to float for mean)
print(tensor_mean)

tensor = torch.tensor([1, 2, 3, 4, 5])
print(tensor[2])  # Access element at index 2 (third element)

tensor_slice = tensor[1:4]  # Get elements from index 1 to 3 (4 is not included)
print(tensor_slice)

tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(tensor_2d[1, 2])  # Access element at second row, third column
print(tensor_2d[:, 1])  # Access all rows, second column

tensor = torch.tensor([1, 2, 3, 4, 5, 6])
reshaped_tensor = tensor.view(2, 3)  # Reshape to 2x3
print(reshaped_tensor)

tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
flattened_tensor = tensor_2d.view(-1)  # Flatten to 1D
print(flattened_tensor)

tensor_1 = torch.tensor([[1, 2], [3, 4]])
tensor_2 = torch.tensor([[5, 6], [7, 8]])
matrix_mult = torch.mm(tensor_1, tensor_2)  # Matrix multiplication
print(matrix_mult)

tensor_transposed = tensor_1.t()  # Transpose of the matrix
print(tensor_transposed)



import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 5)  # Input layer (3 features) -> Hidden layer (5 neurons)
        self.fc2 = nn.Linear(5, 2)  # Hidden layer (5 neurons) -> Output layer (2 neurons)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)  # Output layer
        return x

# Initialize the network
model = SimpleNN()

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Sample input data (3 features)
inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Sample target data (2 classes)
targets = torch.tensor([0, 1])  # Class labels

# Forward pass
outputs = model(inputs)

# Compute loss
loss = criterion(outputs, targets)

# Backward pass and optimize
optimizer.zero_grad()  # Zero the gradients
loss.backward()  # Backpropagate
optimizer.step()  # Update weights

print(f"Outputs: {outputs}")
print(f"Loss: {loss.item()}")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

tensor = torch.tensor([1, 2, 3, 4, 5])
tensor_gpu = tensor.to(device)  # Move tensor to GPU
print(tensor_gpu)


if torch.cuda.is_available():
    print(f"CUDA is available! Using device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")


torch.save(model.state_dict(), 'simple_nn.pth')

model = SimpleNN()  # Re-initialize the model
model.load_state_dict(torch.load('simple_nn.pth'))
model.eval()  # Set to evaluation mode

#----------------------------------------------------------------------------------------------------------------

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification            #  pip install transformers torch

from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Analyze sentiment
result = sentiment_pipeline("I love learning new things about AI!")
print(result)  # Outputs: [{'label': 'POSITIVE', 'score': 0.99}]


from transformers import pipeline

# Load the text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Generate text
result = generator("Once upon a time", max_length=50, num_return_sequences=1)
print(result)

from transformers import pipeline

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering")

# Define context and question
context = "Transformers are a type of neural network architecture used for NLP tasks."
question = "What are transformers used for?"

# Answer the question
result = qa_pipeline(question=question, context=context)
print(result)

from transformers import pipeline

# Load translation pipeline (English to French)
translator = pipeline("translation_en_to_fr")

# Translate text
result = translator("How are you?")
print(result)  # Outputs translation in French


from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization")

# Summarize text
text = "Transformers are deep learning models that utilize self-attention mechanisms for NLP tasks. They have revolutionized the field of machine learning."
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Tokenize input text
inputs = tokenizer("I am learning NLP with Hugging Face Transformers!", return_tensors="pt")

# Get model predictions
outputs = model(**inputs)
print(outputs)


from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Encode input text
input_ids = tokenizer("Once upon a time", return_tensors="pt").input_ids

# Generate text
outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Load a dataset (use your own dataset if available)
dataset = load_dataset("imdb")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# Train the model
trainer.train()

model.save_pretrained("./my_finetuned_model")
tokenizer.save_pretrained("./my_finetuned_model")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./my_finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("./my_finetuned_model")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Move model to GPU
model.to(device)

# Encode input and move to GPU
inputs = tokenizer("I love working with transformers!", return_tensors="pt").to(device)

# Get model output
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Encode input with [MASK] token
inputs = tokenizer("Transformers are [MASK] for NLP tasks.", return_tensors="pt")
mask_token_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)

# Get predictions
outputs = model(**inputs)
logits = outputs.logits
mask_token_logits = logits[0, mask_token_index]

# Get the top predicted tokens
top_token = torch.argmax(mask_token_logits).item()
predicted_token = tokenizer.decode([top_token])
print(predicted_token)

#---------------------------------------------------------------------------------------------------------

import tensorflow as tf              #  pip install tensorflow

tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print(tensor)

random_tensor = tf.random.normal((3, 3))
print(random_tensor)

zeros = tf.zeros((3, 3))
ones = tf.ones((3, 3))
print(zeros, ones)

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

add = tf.add(a, b)
sub = tf.subtract(a, b)
mul = tf.multiply(a, b)
div = tf.divide(a, b)
print(add, sub, mul, div)

matrix_a = tf.constant([[1, 2], [3, 4]])
matrix_b = tf.constant([[5, 6], [7, 8]])

matrix_product = tf.matmul(matrix_a, matrix_b)
print(matrix_product)

tensor = tf.constant([[1, 2], [3, 4]])

sum_result = tf.reduce_sum(tensor)
mean_result = tf.reduce_mean(tensor)
print(sum_result, mean_result)

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2 + 3 * x + 5  # Simple function

grad = tape.gradient(y, x)
print(grad)  # Should output the derivative of the function at x = 3.0


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Summary of the model
model.summary()

import numpy as np

# Example data
X_train = np.random.rand(100, 10)  # 100 samples, 10 features each
y_train = np.random.rand(100, 1)   # 100 samples, 1 target each

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=16)

X_test = np.random.rand(20, 10)  # 20 samples for testing
y_test = np.random.rand(20, 1)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

predictions = model.predict(X_test)
print(predictions)


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # Output layer for 10 classes
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()

# Example dataset (using MNIST)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

cnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))


from tensorflow.keras.layers import SimpleRNN, Embedding

rnn_model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    SimpleRNN(64, return_sequences=True),
    SimpleRNN(64),
    Dense(1, activation='sigmoid')
])

rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.summary()

from tensorflow.keras.applications import MobileNetV2

# Load the MobileNetV2 model with pre-trained weights
mobilenet = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the pre-trained model
transfer_model = Sequential([
    mobilenet,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Output layer for 10 classes
])

transfer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
transfer_model.summary()

model.save("my_model.h5")  # Saves the entire model as a .h5 file


from tensorflow.keras.models import load_model

loaded_model = load_model("my_model.h5")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model.save("saved_model/my_model")

tensorflow_model_server --rest_api_port=8501 --model_name=my_model --model_base_path="$(pwd)/saved_model/my_model"


from tensorflow.keras.callbacks import TensorBoard

tensorboard_callback = TensorBoard(log_dir="logs", histogram_freq=1)

# Add to model's fit
model.fit(X_train, y_train, epochs=5, callbacks=[tensorboard_callback])


Bash = 'tensorboard --logdir=logs'


#-----------------------------------------------------------------------------------------------------------------------------------------------

