# # Prompt Analysis
# Using NLP to determine which prompts our network got wrong to better understand which prompts are best for each diffusion model.

# ## Import Libraries
# Import all necessary libraries

# +
from collections import Counter
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# For using NLTK later
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# -

# ## Import Models
# For this case, we only import ResNet-18.

from models.transferlearning import load_pretrained_model

# ## Import Dataset Class and Transforms
# Used to help load in the images for heatmap generation

from utilities.dataset import ImageDataset
from utilities.transforms import data_transforms


# ## Function to Determine Confusion Matrix
# The same framework as testing through the model, and then looking at prediction and output.

def generate_confusion_matrix(transform, weights_path, batch_size, network, first, second):
    # Loading in initial data
    print("\nLoading in data...")
    test_data = ImageDataset("test", transform, 0.6, first, second)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=True)
    print("Done loading in data.")

    # Test Loading
    dataiter = iter(testloader)
    images, labels, img_names = next(dataiter)

    # Loading in a new example of the neural net, and loading in the weights
    net = network
    if (torch.cuda.is_available()):
        net.to('cuda')
    net.load_state_dict(torch.load(weights_path))
    
    # Generate confusion matrix -- also count all real but fake images, and all images
    print("Generate Confusion Matrix")
    matrix = [[0, 0], [0, 0]]
    real_but_fake = []
    all_images = []
    
    # Perform same as testing/inferencing, but logging data.
    with torch.no_grad():
        for data in testloader:
            images, labels, paths = data
            for i in range(len(paths)):
                all_images.append(paths[i])
            images_cuda, labels_cuda = images.cuda(), labels.cuda()
            
            # Feed the images through our network and evaluate them
            outputs = net(images_cuda)
            _, predicted = torch.max(outputs.cpu().data, 1)
            
            # Calculate the right part of the matrix of prediction and actual
            for i in range(len(predicted)):
                prediction = int(predicted[i].item())
                actual = int(labels[i].item())
                matrix[prediction][actual] += 1
                
                # Adding to set of images to further analyze if real, but fake
                if (prediction is 1 and actual is 0):
                    real_but_fake.append(paths[i])
    
    return matrix, real_but_fake, all_images

# ## Main Function
# Runs all of the necessary functions!

def main(model_type, weights, first, second):
    # Generate the necessary heatmaps
    model = load_pretrained_model()
    matrix, real_but_fake, all_images = generate_confusion_matrix(data_transforms[model_type], weights, 200, model, first, second)
    
    return matrix, real_but_fake, all_images

# ## Run all code!
# Runs all of the code for Transfer Learning.

matrix, real_but_fake, all_images = main(model_type = "TransferLearning", weights = 'weights/TransferLearning/stable-diffusion/TransferLearning-0.6.pth', first='stable-diffusion', second='real')
print(matrix)


# ## Function for Total Number of Nouns + Length
# Using NLTK to calculate total number of nouns and length (number of words).

def nouns_and_tokens(description):
    tokens = nltk.word_tokenize(description)
    tagged = nltk.pos_tag(tokens)
    counts = Counter(tag for word,tag in tagged)
    num_nouns = counts['NN'] + counts['NNS'] + counts['NNP'] + counts['NNPS']
    return num_nouns, tokens


# ## Function for Linguistic Analysis
# Using the above function by looking at a specific dataset.

def linguistic_analysis(all_images):
    # Used to refer to mappings
    df = pd.read_csv('dataset/reference.csv')
    
    # Keeping track of total nouns and lengths
    total = 0
    all_nouns = []
    all_lengths = []
    
    # Iterate through each imags and calculate
    for img in all_images:
        # Grab the description
        index = int(img[-9:-4])
        description = df.iloc[index]['description']
        
        # Update nouns and tokens variables
        num_nouns, tokens = nouns_and_tokens(description)
        total += num_nouns
        all_nouns.append(num_nouns)
        all_lengths.append(len(tokens))
    
    # Print out statistics on number of nouns
    print("Number of Nouns:")
    print("Mean: " + str(np.mean(all_nouns)))
    print("Variance: " + str(np.var(all_nouns)))
    print("")
    
    # Print out statistics on lengths
    print("Length of Message:")
    print("Mean: " + str(np.mean(all_lengths)))
    print("Variance: " + str(np.var(all_lengths)))
    print("")

# ## Function to Print Specific Nouns
# We can use this information to suggest an adversarial dataset.


def specific_nouns(all_images):
    # Keeping track of all nouns and the tags for nouns
    all_nouns = []
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    
    # Iterate through each image
    for img in all_images:
        # Get the description
        index = int(img[-9:-4])
        description = df.iloc[index]['description']
        
        # Get the tokens and their tags
        tokens = nltk.word_tokenize(description)
        tagged = nltk.pos_tag(tokens)
        
        # Add to set if a noun
        for word, tag in tagged:
            if (tag in noun_tags):
                all_nouns.append(word)
    
    return all_nouns


# ## Applying Linguistic Analysis Functions
# Calling above functions to print out data!

# +
print("Analyzing All Images:")
linguistic_analysis(all_images)

print("\nAnalyzing Real, but Fake Images:")
linguistic_analysis(real_but_fake)


# -
# ## Super Awesome Bootstrapping Techniques!
# Ensuring statistical significance to make claims.

def bootstrapping(total_observations, subsection_of_interest):
    sample_mean = np.mean(total_observations)
    mean_difference = abs(sample_mean - np.mean(subsection_of_interest))
    subsection_length = len(subsection_of_interest)
    count = 0.0
    iteration_count = 10000
    for _ in range(iteration_count):
        sampled_lengths = np.random.choice(total_observations, subsection_length, replace=True)
        if abs(np.mean(sampled_lengths) - sample_mean) >= mean_difference:
            count += 1
    print(count / iteration_count)


# ## Calculating Statistical Significance
# Calling above function!

print("Running simple bootstrapping to test against null hypothesis")
print("Statistical significance of prompt lengths")
bootstrapping(all_lengths, rbf_lengths)
print("Statistical significance of noun counts")
bootstrapping(all_nouns, rbf_nouns)

# ## Looking at Nouns in Real, but Fake
# See motivation above!

print("Looking at nouns of real, but fake images set:")
print(specific_nouns(real_but_fake))
