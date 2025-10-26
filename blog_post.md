# "Neural Babel: What Do Neural Networks Talk About?"

Imagine overhearing a conversation in a language you don’t speak. The speakers understand each other perfectly, but you have no idea what they're saying. In this project, the speakers were neural networks, and the language emerged spontaneously when they were trained to collaboratively solve a task. We tried to build a translator for this “neuralese” and this is what we found.

## Teaching Machines to Point at Things
Take a world of objects W and a subset of this world X.

>> Insert drawing
>> 
Before scrolling, how would YOU describe this selection? 
Open hidden section: rule:
A neural network called the speaker is given W and X and outputs a neuralese vector V that ideally captures this rule. Another network called the listener takes in V and an element of the world W_i e W and predicts whether or not W_i belongs in X.
The listener never sees X. It relies on the speaker’s neuralese output to describe X to it.
Andreas and Klein (2017) [cite] shows us that this "language" could be negated via linear transformation to take on the opposite meaning. Now, this project attempts to figure out whether these vectors can be outrightly translated.
For training data, Andreas and Klein use labels from the GENX dataset [cite]. We forewent this dataset and generated our own. Each object had a color, shape and outline thickness encoded row by row.
>>> Insert drawing that shows how a matrix is an object
Our dataset had 80_730 unique worlds of 5 objects each. Subsets were created using 72 unique rules which could be a single feature rule (‘red’), single feature negation (‘not red’), two features of different types joined by and/or (‘red and circle’, ‘triangle or thick-outline’). Skipping over world-rule combinations that resulted in empty subsets, we gathered a dataset of 1_705_833 entries.
>>> Insert drawing from explore_dataset
Training separate networks to evolve languages in order to play a communication game has also been done in Gupta et al (2021), Lazaridou et al (2018) and Andreas et al (2018).

## 2. Conference

### 2.1. Training the speaker-listener

The speaker-listener system achieved 99.56% test accuracy on an unseen test set, with accuracy climbing from 60% to 95%+ by epoch 1 implying that the task was easily learned.
>>> Insert diagram going over the architecture of the speaker-listener system.
To prevent the speaker from encoding positional shortcuts ("select positions [0,2,3]") and force it to learn semantic rules ("select purple circles"), the world objects are shuffled before being fed to the listener. 

### 2.2 Cross-Rule Validation: The Baseline Signal Problem

After training, we needed to verify that the speaker actually learned to encode rules meaningfully. Did "red objects" produce similar neuralese across different worlds? Did "red" neuralese differ from "triangle" neuralese?
Using the trained speaker, we generated 100 neuralese vectors for each of 9 different rules (like ‘red’, ‘green or triangle’, ‘not purple’, etc.). Then we measured how similar these vectors were to each other using cosine similarity.
We expected that neuralese for the same rule should be similar (high within-rule similarity), while neuralese for different rules should be different (low cross-rule similarity), but the similarities for both categories were high (0.908 ± 0.090, and 0.865 ± 0.097 respectively). 
We guessed that the neuralese contained a massive "baseline signal" that concealed the actual messages. So we normalized the neuralese by computing the average vector across all examples, then subtracting it from each vector. This brought the cosine similarity for same-rule neuralese down to 0.246 ± 0.519 (moderate similarity) and cross-rule similarity to -0.069 ± 0.500 (negative similarity).
This revision showed that rule information did exist in the neuralese, just hidden beneath the baseline. And we inferred we should normalize the neuralese before attempting to translate them.


## 3. Translation
### 3.1. Training the translator


