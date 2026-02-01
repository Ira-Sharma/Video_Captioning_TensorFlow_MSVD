# Video_Captioning_TensorFlow_MSVD

* [Introduction](#introduction)
* [MSVD Dataset](#msvd-dataset)
* [Video Feature Extraction](#video-feature-extraction)
* [Caption Analysis](#caption-analysis)
* [Prepare Captions](#prepare-captions)
* [Model Creation](#model-creation)
  * [Encoder](#encoder)
  * [Decoder](#decoder)
* [Training the Model](#training-the-model)
  * [Learning Rate](#learning-rate)
  * [Loss](#loss)
  * [Accuracy](#accuracy)
* [Model for Inference](#model-for-inference)
* [Predicting the Captions using Greedy Search](#predicting-the-captions-using-greedy-search)
* [BLEU Score](#bleu-score)
* [Testing the Model](#testing-the-model)
* [Future Work](#future-work)
* [References](#references)

## Introduction

Training machines to caption videos has been a very interesting topic in recent times. This task is a combination of video data understanding, conversion of videos into image frames, feature extraction, and translation of visual representations into natural languages such as English. It has applications including recommendations in editing applications, usage in virtual assistants, for video indexing, for visually impaired people, in social media, and several other natural language processing applications.

A working model of the video caption generator is built by using CNN (Convolutional Neural Network) and LSTM (Long Short-Term Memory) units. The MSVD dataset—a pre-captioned YouTube video repository from Microsoft—is used to train the model, consisting of approximately 2,000 videos with each video mapped to 10-15 English captions that describe the video. After training the model, predictions are made on the test data, and BLEU scores are calculated to evaluate the performance of the model.

## MSVD Dataset

The Microsoft Research Video Description Corpus (MSVD) dataset consists of about 120K sentences collected during the summer of 2010. Workers on Mechanical Turk were paid to watch a short video snippet and then summarize the action in a single sentence. The result is a set of roughly parallel descriptions of around 2,000 video snippets. Because the workers were urged to complete the task in the language of their choice, both paraphrases and bilingual alternatives are captured in the data. The dataset used for this project was downloaded from [here](https://www.oreilly.com/library/view/intelligent-projects-using/9781788996921/6c15342b-1fd9-450d-b5fb-9309634597ec.xhtml). It contains exactly 1,969 videos and 10-15 English captions per video.

## Video Feature Extraction

First, the videos in .avi format are converted into 80 image frames. The Convolutional Neural Network (CNN) method is used for feature extraction. The VGG16 model is a series of convolutional layers followed by a few dense (or fully connected) layers. The input layer to the last max pooling layer (labeled by 7 × 7 × 512) is regarded as the feature extraction part of the model, while the rest of the network is regarded as the classification part of the model. Each of the 80 frames is passed through a pre-trained VGG16, and 4,096 features are extracted from each frame. These features are stacked to form an (80, 4096) shaped array, where 80 is the number of frames and 4,096 is the number of extracted features from each frame.
<p align="center"><img align="center" src="Images and gifs/vgg16.jpg" width="500" height="500" /></p>

## Caption Analysis

Since the caption data was completely raw, the data had a lot of spelling errors and was in many different languages. First, only the English captions were extracted and were mapped to their unique VideoIDs. The extra captions whose videos were not available were removed, spelling errors were corrected, and the mapping data was split into train and test data containing the separate videos, features, and captions in different folders.

After loading the training caption data, some analysis is performed on the captions, including the visualization of the frequency distribution of the words—the most common 50 words and the least common 50 words.
<p align="center"><img align="center" src="Images and gifs/caption_analysis.png" width="500" height="500" /></p>

## Prepare Captions

1. Clean it by converting all words to lowercase and removing punctuation, words with numbers, and short words with a single character.
2. Add `<bos>` and `<eos>` tokens at the beginning and end of the sentence.
3. Tokenize the sentence by mapping each word to a numeric word ID. This is done by building a vocabulary of all the words that occur in the set of captions. 2,226 words occurred more than 5 times, so the number of words for the Tokenizer was taken to be 2,226.
4. Only the sentences containing words between 6 and 10 are taken. Then each sentence is extended to the same length by adding padding sequences. This is needed because the model expects every data sample to have the same fixed length.

## Model Creation

A Data Generator function is defined to get data in batches instead of taking it all together to avoid a session crash.

- The number of frames per video used for training is 80.
- The number of features from each frame taken is 4,096.
- The number of hidden features for LSTM is 512.
- The maximum length of each sentence is taken to be 10.
- The final number of tokens in the softmax layer is 2,226.
- The batch size is taken to be 320.

### Encoder

The encoder takes the videos or a sequence of images as input and produces the encoded vectors that capture the essential features of the video. LSTM is used for the encoder. It consists of the encoder inputs (Input Layer 1), which is connected to the encoder LSTM, which is further connected to the decoder LSTM.

### Decoder

The decoder generates captions word by word using LSTMs, which are able to sequentially generate words. The input for the decoder is the encoded feature vectors from the encoder LSTM.

It receives combined input from the encoder LSTM and decoder inputs (Input Layer 2). The decoder LSTM is connected to a Dense layer with Softmax activation function.
<p align="center"><img align="center" src="Images and gifs/model_train.png" width="500" height="500" /></p>
## Training the Model

### Learning Rate

Deep learning neural networks are trained using the stochastic gradient descent optimization algorithm. The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated. The Keras deep learning library allows you to easily configure the learning rate for a number of different variations of the stochastic gradient descent optimization algorithm. Adam Optimizer is used with a learning rate of 0.0007 (other learning rates were also tried, but 0.0007 gave the most stable results in less time).

### Loss

Cross Entropy Loss is the most popular and effective measurement for the performance of a classification model whose output is a probability value between 0 and 1. Categorical cross-entropy is applied in multiclass classification scenarios. In the formula, we multiply the actual outcome with the logarithm of the outcome produced by the model for more than two classes and then sum up. The categorical cross-entropy is appropriate in combination with an activation function such as softmax that can produce several probabilities for the number of classes that sum up to 1.
<p align="center"><img align="center" src="Images and gifs/loss.png" width="500" height="500" /></p>

### Accuracy

Finally, because it is a classification problem, the classification accuracy is collected and reported, defined via the metrics argument. 
<p align="center"><img align="center" src="Images and gifs/acurracy.png" width="500" height="500" /></p>

## Model for Inference

The encoder model is the same as that for the sequential training model used above. The encoder model gives us the predictions. We are only interested in the final output state, so all the other outputs from the encoder will be discarded. The final state of the encoder is fed into the decoder as its initial state, along with the `<bos>` token so that the decoder predicts the next word.

## Predicting the Captions using Greedy Search

Greedy search technique is used, which selects the most likely word at each step in the output sequence. This approach has the benefit that it is very fast.

The words are integer encoded, such that the column index can be used to look up the associated word in the vocabulary. Therefore, the task of decoding becomes the task of selecting a sequence of integers from the probability distributions.

The argmax() mathematical function can be used to select the index of an array that has the largest value. We can use this function to select the word index that is most likely at each step in the sequence.

## BLEU Score

BLEU (Bilingual Evaluation Understudy) is a well-acknowledged metric to measure the similarity of one hypothesis sentence to multiple reference sentences. Given a single hypothesis sentence and multiple reference sentences, it returns a value between 0 and 1. A metric close to 1 means that the two are very similar. The Python Natural Language Toolkit library, or NLTK, provides an implementation of the BLEU score that you can use to evaluate your generated text against a reference.

The model is tested on the test data. BLEU scores are evaluated to study the performance of the model with the predicted caption against the actual captions in a list of tokens. The BLEU score for all the testing data combined is:

- BLEU-1: 0.546524
- BLEU-2: 0.346926

<h2 id="Testing the model">Testing the model</h2>
<table>
 <tr>
  <th>Video</th>
 <th>Greedy Text(Time taken)</th>
 </tr>
<tr>
 <td><img src="Images and gifs/-_hbPLsZvvo_43_55.gif" width="320px"/></td>
 <td>a woman is slicing a carrot(0.72s)</td>
 </tr>
 <tr>
 <td><img src="Images and gifs/0hyZ__3YhZc_289_295.gif" width="320px"/></td>
 <td>a man is cooking the kitchen(0.80s)</td>
 </tr>
 <tr>
 <td><img src="Images and gifs/Ffv7fhL1EDY_177_184.gif" width="320px"/></td>
 <td>a man is playing with a ball(0.70s)</td>
 </tr>
 <tr>
 <td><img src="Images and gifs/0lh_UWF9ZP4_183_190.gif" width="320px"/></td>
 <td>a woman is slicing a piece of fish(0.76s)</td>
 </tr>
 <tr>
 <td><img src="Images and gifs/tn1d5DmdMqY_15_28.gif" width="320px"/></td>
 <td>a man is playing a guitar(0.73s)</td>
 </tr>
 <tr>
 <td><img src="Images and gifs/GWQTAe64m-0_160_166.gif" width="320px"/></td>
 <td>a girl is singing(0.71s)</td>
 </tr>
 <tr>
 <td><img src="Images and gifs/Wv3u2q3oGeU_35_45.gif" width="320px"/></td>
 <td>a man playing piano(0.75s)</td>
 </tr>
 <tr>
 <td><img src="Images and gifs/hJFBXHtxKIc_310_315.gif" width="320px"/></td>
 <td>a man is eating spaghetti(0.76s)</td>
 </tr>
 </table>
 
## Future Work

A pre-trained CNN network was directly used as part of our pipeline without fine-tuning, so the network does not adapt to this specific training dataset. Thus, by experimenting with different CNN pre-trained networks and enabling fine-tuning, we can expect to achieve a slightly higher BLEU-4 score.

## References

[1] https://medium.com/analytics-vidhya/video-captioning-with-keras-511984a2cfff

[2] How to Implement a Greedy Search Decoder for Natural Language Processing. https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/

[3] An Introduction to Neural Network Loss Functions. https://programmathically.com/an-introduction-to-neural-network-loss-functions/

[4] BLEU Score. https://en.wikipedia.org/wiki/BLEU

[5] MSVD Dataset. https://www.oreilly.com/library/view/intelligent-projects-using/9781788996921/6c15342b-1fd9-450d-b5fb-9309634597ec.xhtml

[6] Convolutional Neural Network. https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

[7] LSTM. https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21

[8] VGG-16 — CNN model. https://www.geeksforgeeks.org/vgg-16-cnn-model/

[9] Understand the Impact of Learning Rate on Neural Network Performance. https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/


