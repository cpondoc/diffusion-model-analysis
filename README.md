# Seeing Through the Facade: Leveraging Computer Vision to Understand the Realism, Expressivity, and Limitations of Diffusion Models
By: Christopher Pondoc, Joseph O'Brien, and Joseph Guman

## Introduction
Recent research in natural language processing and computer vision has shown progress in image generation from text. Projects such as DALLE-2 and Stable Diffusion 2.0 are examples of diffusion models trained on large language datasets and can create photorealistic images, transfer styles from one image to another, and process complex queries. For some, generative models amplify human creativity by serving as a tool that supercharges the creative processes. However, many others feel that developments will only lead to more problems, such as troubling deepfakes, the replacement of human artists by artificial intelligence, and disinformation that can be more quickly created and disseminated.

In the context of these malicious applications, we explore the task of binary classification between real and AI-generated images. In practice, our work could be applied to help tag disinformation and deepfakes around the Internet. Our process aims to answer three questions:
- **How can we achieve the highest test accuracy on classifying between real and fake images?** What is our limit with a custom-built convolutional neural network (CNN), and how we can leverage state-of-the-art computer vision techniques to push the bound higher?
- **When making classifications, what features distinguish real images from AI-generated images, and DALLE-2 and Stable Diffusion 2.0?** What visualization and training techniques can we use to generate these insights?
- Finally, **which prompts generate the most photorealistic results and successfully trick the highest performing classifier?** Can this information connect back to the expressivity of each diffusion model?

With this approach, we hope to offer both (a.) insights into the limitations of existing diffusion models as well as (b.) potential areas for improvement in the task of photorealistic image generation.

## Code Breakdown
We split up our code into several components.

### Notebooks
The main folder contains all of the notebooks used for the main part of the experiment. `Model Training.ipynb` is where we train all of our networks. `Generating Heatmaps.ipynb` is where heatmaps are generated after training a network. `Prompt Analysis.ipynb` is where we use natural language processing (NLP) to analyze and determine the characteristics of a good prompt for both DALLE-2 and Stable Diffusion.

### Utilities
This folder contains code pertaining to repeatedly used functions. `dataset.py` defines the dataset class used for training. `transforms.py` defines the image transforms applied when preprocessing.

### Models
This folder contains code pertaining to our neural network architectures. `convbasic.py` contains code for our baseline CNN, while `transferlearning.py` contains code for our ResNet-18 architecture.

### Analysis
This folder contains code for all of the analysis done outside of model training. `Bias vs. Variance.ipynb` contains code for generating graphs to visualize the bias-variance tradeoff experiment we ran to determine the optimal data.

### Results
This folder contains all of our graphs and output logs pertaining to training and testing. `ConvBasic` contains all results for our baseline CNN, while `TransferLearning` contains all results for our ResNet-18 architecture. Each folder also has a subfolder pertaining to the specific task it was trained on: real vs. DALLE-2, Stable Diffusion, or pitting Stable Diffusion against DALLE-2. Finally, `BiasVariance` contains the graphs necessary for making a conclusion on the bias-variance tradeoff experiment, which we outlined in the **Methods** section of our paper.

Each folder contains graphs of training and test accuracy against the proportion of training data used. Full text logs of training and testing are also included. 

### Reference
This folder contains miscellaneous items: old scripts used when trying out old datasets (i.e. ImageNet for our milestone), scripts for scraping the DALLE-2 data using the OpenAI API, random AI-generated images, and graphs from old training runs.

## A Note on the Dataset + Weights
We decided not to upload our dataset nor our weights to GitHub. If you are interested in gaining access, please contact either Chris (clpondoc@stanford.edu), O'Brien (jobrien3@stanford.edu), or Guman (joeytg@stanford.edu) for more information.
