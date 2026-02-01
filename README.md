# Video_Captioning_TensorFlow_MSVD
* <a href="#Introduction">Introduction</a>
* <a href="#MSVD Dataset">MSVD Dataset</a>
* <a href="#Image Feature Extraction">Image Feature Extraction</a>
* <a href="#Caption Data Analysis">Caption Data Analysis</a>
* <a href="#Prepare Captions">Prepare Captions</a>
* <a href="#Model Architecture">Model Architecture</a>
  * <a href="#Encoder">Encoder</a>
  * <a href="#Decoder">Decoder</a>
* <a href="#Loss Function">Loss Function</a>
* <a href="#Training the model">Training the model</a>
* <a href="#Generating the captions">Generating the captions</a>
* <a href="#BLEU Score">BLEU Score</a>
* <a href="#Testing the model">Testing the model</a>
* <a href="#Results">Results</a>
* <a href="#Future Work">Future Work</a>
* <a href="#References">References</a>

<h2 id="Introduction">Introduction</h2>
Training the machines to caption videos has been a very interesting topic in recent times. This
task is a combination of video data understanding, conversion of videos into image frames, and feature
extraction and translation of visual representations into natural languages such as English. It has
its applications, including recommendations in editing applications, usage in virtual assistants, for
video indexing, for visually impaired people, in social media, and several other natural language
processing applications.
Working model of the Video caption generator is built by using CNN (Convolutional Neural Network) and LSTM (Long short term memory) units. MSVD dataset —a pre-captioned YouTube video repository from Microsoft is used to train the model, consisting of approximately 2000 videos and each video is mapped to 10-15 English captions that describe the video. After training the model, predictions are made on the test data, and BLEU scores are calculated to see the performance
of the model.
<h2 id="MSVD Dataset">MSVD Dataset</h2>
The Microsoft Research Video Description Corpus (MSVD) dataset consists of about 120K sentences
collected during the summer of 2010. Workers on Mechanical Turk were paid to watch a short video
snippet and then summarize the action in a single sentence. The result is a set of roughly parallel
descriptions of around 2,000 video snippets. Because the workers were urged to complete the task
in the language of their choice, both paraphrases and bilingual alternatives are captured in the data.
The dataset used for this project was downloaded from here(https://www.oreilly.com/library/view/intelligent-projects-using/9781788996921/6c15342b-1fd9-450d-b5fb-9309634597ec.xhtml). It contains exactly 1969 videos and
10-15 English captions per video.


<h2 id="Image Feature Extraction">Image Feature Extraction</h2>
Convolutional Neural Network (CNN) method is used for image feature extraction. The VGG16 model is a series of convolutional layers followed by a few dense (or fully connected) layers. The input layer to the last max pooling layer (labeled by 7 x 7 x 512) is regarded as the feature extraction part of the model, while the rest of the network is regarded as the classification part of the model. After loading VGG16 and restructuring by excluding the classification part of the model, we load the input image with the size expected by the model, in this case, 224 x 224, then we convert the image pixels to a numpy array whose dimensions are reshaped for the model. After preprocessing the images for VGG16, general features are extracted which are then used further for the image classifiers.
<h2 id="Caption Data Analysis">Caption Data Analysis</h2>
After loading the caption data, some analysis is done on the captions, including the visualisation of the frequency distribution of the words- the most common 50 words and the least common 50 words.
<p align = "center"><img align = "center" src = "Images/ana.png" /></p>
<h2 id="Prepare Captions">Prepare Captions</h2>
Each caption consists of an English sentence. To prepare this for training, we perform the following steps on each sentence:
1. Clean it by converting all words to lower case and removing punctuation, words with numbers, and short words with a single character.
2. Add ‘<startseq>’ and ‘<endseq>’ tokens at the beginning and end of the sentence.
3. Tokenize the sentence by mapping each word to a numeric word ID. It does this by building
a vocabulary of all the words that occur in the set of captions.
4. Extend each sentence to the same length by adding padding sequences. This is needed because the model expects every data sample to have the same fixed length.
<h2 id="Model Architecture">Model Architecture</h2>
<h3 id="Encoder">Encoder</h3>
  The encoder takes the images as input and produces the encoded image vectors that capture the essential features of the image.
1. Image Feature Layers- Consist of Input1, Dropout, and Dense layers with Relu activation function.
2. Sequence Feature Layers- Consist of Input2, Embedding, Dropout, and LSTM layers.
<h3 id="Decoder">Decoder</h3>
  The decoder generates image captions word by word using a Recurrent Neural Network - LSTM,s which is able to sequentially generate words. The input for the decoder is the encoded image feature vectors from the CNN and the encoded image captions produced in the data preprocessing stage.
It gets combined input from Input1 and Input2, consists of a Dense layer with Relu activation function, and the output as Dense layer with Softmax activation function.
  <p align = "center"><img align = "center" src = "Images/model.png" /></p>
<h2 id="Loss Function">Loss Function</h2>
The nature of our RNN output is a series likelihood of words’ occurences, and in order to quantify the quality of the RNN output, we propose to use Cross Entropy Loss. This is the most popular and effective measurement for the performance of a classification model whose output is a probability value between 0 and 1. The categorical cross-entropy is applied in multiclass classification scenarios. in the formula, we multiply the actual outcome with the logarithm of the outcome produced by the model for over more than two classes and then sum up.
The categorical cross-entropy is appropriate in combination with an activation function such as
softmax that can produce several probabilities for the number of classes that sum up to 1.

<h2 id="Training the model">Training the model</h2>
  
A Data Generator function is defined to get data in batches instead of taking it all together to avoid a session crash. The entire data is split into train and test, and the model training is done on the train data. The loss decreases gradually over the iterations; the number of epochs and batch size are assigned accordingly for better results.
  <p align = "center"><img align = "center" src = "Images/loss.png" width="500" height="300" /></p>
<h2 id="Generating the captions">Generating the captions</h2>
  Captions are generated for the image by converting the predicted index from the model into a word. All the words for the image are generated, the caption starts with ’startseq’ and the model continues to predict the caption until the ’endseq’ appeared.
<h2 id="BLEU Score">BLEU Score</h2>
  
BLEU (Bilingual Evaluation Understudy) is a well-acknowledged metric to measure the similarity of one hypothesis sentence to multiple reference sentences. Given a single hypothesis sentence and multiple reference sentences, it returns a value between 0 and 1. The metric close to 1 means that the two are very similar. The Python Natural Language Toolkit library, or NLTK, provides an implementation of the BLEU score that you can use to evaluate your generated text against a reference.
<h2 id="Testing the model">Testing the model</h2>
  Model is tested over the test data, BLEU Score is evaluated to study the performance of the model with the predicted caption against the actual captions, in a list of tokens.
<p align = "center"><img align = "center" src = "Images/bleu.png" width="500" height="300"/></p>
<h2 id="Results">Results</h2>
  Finally, the results are visualised for 6 test images containing the actual captions, a predicted caption, the BLEU score, and a comment on whether the predicted caption is Bad, Not Bad, and Good depending on the BLEU score against the actual captions for that particular image.
<p align = "center"><img align = "center" src = "Images/dog.png" width="500" height="300"/></p>
<p align = "center"><img align = "center" src = "Images/bike.png" width="500" height="300"/></p>
<p align = "center"><img align = "center" src = "Images/water.png" width="500" height="300"/></p>
<p align = "center"><img align = "center" src = "Images/guy.png" width="500" height="300" /></p>
<p align = "center"><img align = "center" src = "Images/horse.png" width="500" height="300"/></p>
<p align = "center"><img align = "center" src = "Images/car.png" width="500" height="300" /></p>
 
<h2 id="Future Work">Future Work</h2>
  A smaller dataset (Flickr8k) was used due to limited computational power. Potential improvement can be done by training on a combination of Flickr8k, Flickr30k, and MSCOCO. Pre-trained CNN network was directly used as part of our pipeline without fine-tuning, so the network does not adapt to this specific training dataset. Thus, by experimenting with different CNN pre-trained networks and enabling fine-tuning, we can expect to achieve a slightly higher BLEU- 4 score.
Video captioning is a text description of video content generation. Compared with image captioning, the scene changes greatly and contains more information than a static image. Therefore, for the generation of text description, video caption needs to extract more features, which can be the next repository (https://github.com/Ira-Sharma/Video_Captioning_TensorFlow_MSVD).
<h2 id="References">References</h2>
  
[1] Xu, Kelvin, Jimmy Ba, Ryan Kiros, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, and Yoshua Bengio.“Show, attend and tell: Neural image caption generation with visual at- tention.” arXiv preprint arXiv:1502.03044(2015).
[2] Qichen Fu, Yige Liu, Zijian Xie University of Michigan, Ann Arbor fuqichen, yigel, xiezj@umich.edu: EECS442 Final Project Report eecs442-report.pdf
[3] https://neurohive.io/en/popular-networks/vgg16/
[4] An Introduction to Neural Network Loss Functions https://programmathically.com/an-
introduction-to-neural-network-loss-functions/
[5] Bleu Score https://en.wikipedia.org/wiki/BLEU
[6] Flickr8k Dataset https://www.kaggle.com/datasets/adityajn105/flickr8k
[7] Convolutional Neural Network https://towardsdatascience.com/a-comprehensive- guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
[8] LTSM https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a- step-by-step-explanation-44e9eb85bf21
[9] VGG-16 — CNN model https://www.geeksforgeeks.org/vgg-16-cnn-model/
[10] Image captioning with visual attention https://www.tensorflow.org/tutorials/text/ image_captioning


<h2 id="Testing">Testing</h2>
<h3 id="Performance">Performance of greedy algorithms on testing data</h3>
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
