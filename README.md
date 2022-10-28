# CS 229 Final Project
By: Chris Pondoc, Joey O'Brien, Joey Guman

## Proposal:
Recent research in natural language processing and computer vision has shown progress in image generation from text. Projects such as DALLE-2 [1] and Stable Diffusion [2] are examples of models that have been trained on large language datasets and can create photorealistic images, transfer styles from one image to another, and process complex queries. For some, generative models amplify human creativity by serving as a tool that supercharges the creative processes [3]. However, many others feel that developments will only lead to more problems, such as troubling deepfakes [4], the replacement of human artists by artificial intelligence [5], and disinformation that can be more quickly created and disseminated.

Our project aims to build a binary classifier that distinguishes between real and generative images. In practice, our work could be applied to help tag disinformation at scale on social media platforms. We’ll attempt various approaches to build our classifier, starting by building our own convolutional neural network (CNN) from scratch. We can also employ the methodology of transfer learning [6], which will take a model previously trained on another classification task and apply it to this task. Finally, we can also delve into the world of generative adversarial networks (GANs) [7] and focus on the half that distinguishes between the training and generative data.

In developing our classifier, we will run numerous particular experiments, especially in the context of our training and test data. More specifically, the main question will revolve around how to create the most balanced dataset between real and generative images. By scraping Google and other search engines, we could leverage existing large datasets, such as ImageNet [8], or have more parity between the prompts and the real images. When evaluating our neural network’s progress, we intend to analyze both in terms of overall accuracy and tackling key sets of images, such as humans, animals, or other subjects.

## Sources:
[1] https://openai.com/dall-e-2/
[2] https://arxiv.org/abs/2112.10752
[3] https://www.linkedin.com/pulse/blitzscaling-creativity-dall-e-reid-hoffman/
[4] https://techcrunch.com/2022/08/24/deepfakes-for-all-uncensored-ai-art-model-prompts-ethics-questions/
[5] https://artofericwayne.com/2022/05/31/will-ai-replace-human-artists/
[6] https://machinelearningmastery.com/transfer-learning-for-deep-learning/
[7] https://arxiv.org/abs/1406.2661
[8] https://www.image-net.org/
