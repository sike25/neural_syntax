## Analogs of Linguistic Structure in Deep Representations

**Jacob Andreas et al, 2017**

This paper demonstrates that if you taught neural networks to pose as collaborative agents in 
a communication game, the "language" they learn to solve the task shares the compositionality of  natural language.

Paper: [https://arxiv.org/pdf/1707.08139](https://arxiv.org/pdf/1707.08139)

### 1. Task

The authors use a communication game to get two neural networks to develop a shared 'language',
so they can analyze this language and figure out whether it shares anything with our natural language.

The game has:
1. A world **W** of 1-20 objects. These objects have attributes (for example, shape and color-- red circle, blue circle, blue square).
2. A target subset **X** of **W** (for example, the blue objects).

The **speaker** (or encoder) is an RNN which can see the world **W** and the target subset **X**, and tries to learn a 64-dimensional vector to describe **X**.        
The **listener** (or decoder) is an MLP which is trained to decipher the 64-dimensional vector and correctly label an object $$W_i$$ from **W** as either belonging to **X** or not.

The GENX dataset is used for training. It contains 4,170 natural language expressions and their corresponding logical rule forms. For example, "everything but the red circles" and its logical rule form: $$! (circle \cap red)$$.        
These examples are scattered across 273 instances of the game (in other words, 273 worlds).

Terms:     
$$e(W, X)$$ - The logical expression translated from a human description of **X** within **W**. Or
the human description itself. For example: it refers to both $$! (circle \cap red)$$ and "everything but the red circles".

$$[e]_W$$ - The result of evaluating the rule e(W) against W. For example:    
If **W** is {red circle, red square, blue circle}, $$[e]_W$$ will be {red square, blue circle}

$$f(W, X)$$ - The 64-dimensional output of the encoder. Its own description of **X** within **W**.

$$[f]_W$$ - The full vector of decoder outputs on **W**.         
If **W** is {red circle, red square, blue circle}, $$[f]_W$$ will be {0, 1, 1}


### 3. Approach

As one could guess, the encoder-decoder system reaches near perfect accuracy on this pretty simple task. But the authors are only interested in analyzing HOW it does— in particular, what is the nature of the encoder's output vector? How does it resemble or not resemble natural language? 

Since the model performs so well, for each world in the dataset, we would expect e(W) = f(W) for a given rule and there is nothing revalatory about this. Equality here means that when the rule is executed on the world, it selects the same objects. The authors want to test semantic equivalence.

They collect worlds from GenX that have not been seen by the model (in training or testing), and test a single rule across all of these worlds. For example:

Human expression:     
e = "circle"

Worlds:     
W1 = {red circle, blue circle, green square}     
W2 = {red triangle, blue circle, green square, yellow circle}    
W3 = {red triangle, blue square, green square}

Human tabular meaning representation:          
rep(e) = [ [e]W1, [e]W2,[e]W3 ]

- $$[e]_{W1}$$ = {red circle, blue circle}
- $$[e]_{W2}$$ = {blue circle, yellow circle}
- $$[e]_{W3}$$ = {}

Model tabular meaning representation:   
rep(f) = [ [f]W1, [f]W2,[f]W3 ]

- $$[f]_{W1}$$ = decoder's prediction for each object in W1
- $$[f]_{W2}$$ = decoder's prediction for each object in W2
- $$[f]_{W3}$$ = decoder's prediction for each object in W3

Ideally, this should be the same as the human ones. 
The idea here is to understand whether the model has learned the global concept "circle".

### 4. Interpreting the Meaning of Messages

Now the authors ask: does the encoder's 64D output vector encode the same meanings as the human phrases? 
How often? And how explicitly?

They take a scene from their test set, and they check both the encoder's message $$f(W, X)$$ (a 64d vector) and compute its tabular representation $$rep(f)$$.
They then compute representations between three different theories of what the model might be doing:

1. Random Theory - Predicts object membership randomly.
2. Literal Theory - Predicts object membership based on whether objeect is selected in the original scene.
3. Human Theory - Picks object membership according to whether it would have been selected by the human message $$e(W, X)$$.

![image](https://github.com/user-attachments/assets/01b67dba-cb5c-4657-910f-4df3c3456797)

They then measure agreement between rep(f) and these representations on three levels.
1. Objects (do they predict the same objects belong to the set?)
2. Worlds (do they agree on all objects in a given world?)
3. Tabular meaning representations (do they agree across all tested worlds?)

![image](https://github.com/user-attachments/assets/bbc4022e-fa04-4896-97d5-149a479ac482)

These results show that the network's language for describing an object sets aligns with human language on a high-level.

### 5. Interpreting the Structure of Messages

Now, the authors check whether encoder outputs contain logical constructions (negation - not, conjuction - and, disjunction - or) that
are found in human language and are specifically rampant in this dataset/ task.

**Negation**                
To look for negation, they look for a consistent mathematical operation that could transform a vector $$f(W,X)$$ representing all "the circles" into one representing "not circles".

They collect examples of the form $$(e, f, e', f')$$ where:
1. $$e$$ and $$e'$$ are natural language negations.
2. $$f$$ and $$f'$$ are the corresponding RNN representations.
3. $$rep(e) = rep(f)$$ and $$rep(e') = rep(f')$$, so representations match the human phrases in meaning.

The idea is that if the model does not have some notion of negation, there would be no predictable relationship between $$(f, f')$$. But if the model does have some notion of negation, then there exists a transformation N such that $$N \cdot f = f'$$

Using example $$(f, f')$$, they solve for $$\hat{N}$$, a matrix which minimizes the sum of squared differences between the predicted negation $$N \cdot f$$ and the actual negation $$f'$$. 

They solve for N using the least squares solution:

F = $$[f_1, ... f_n]^T$$ (each f as a row)          
F' = $$[f'_1, ... f'_n]^T$$ (each f' as a row)    
- $$\hat{N} = (F^T F)^{-1} (F^T F')$$

Then they test how often $$rep(Nf) = rep(e')$$ on separate, test examples.     
They observe 97% agreement for individual objects and 45% agreement on full representations.

This implies that $$\hat{N}$$ is analogous to negation in natural language. And that the concept of negation is linear.

**Conjuction and Disjunction**    
Here, they collect examples of the form $$(e, f, e', f', e'', f'')$$ where:
1. $$rep(e) = rep(f)$$
2. $$rep(e') = rep(f')$$
3. $$rep(e'') = rep(f'')$$
4. $$e'' = e \cap e'$$ conjuction OR $$e'' = e \cup e'$$ disjunction.

They then solve for a matrix M that minimizes the difference between the predicted conjuction $$(Mf + Mf')$$ and the actual conjuction $$f"$$. Or disjunction, depending on the case. 

They run disjunction tests as they did with negation and observe 92% agreement for individual objects but 19% on full representation. They don't report their conjuction test results. We assume they are not favorable. I hypothesize that the model captures conjuction and disjunction in ways that are less linear than negation.

### 6. Conclusions

The authors discovered a way to explore whether the compositional structure found in natural language can also be found in vectors produced by a neural network which has never seen human language. They do this by concentrating on the truth of the vector representation when applied to solve a problem, rather than the direct structure of it.

They also posit, as a future question, asking how or whether these vectors capture hierarchical information like in "not (blue and circular)". 

---

### Reflection      

**What are the strengths?** 
1. Interesting and creative methodology.

**What are the weaknesses?**      
1. Very confusing. Explanations are incredibly opaque.
2. $$W_i$$ was used to refer to a specific object in a world $$W$$. Then $$W_i$$ was used to refer to a world in a set of worlds { $$W_i$$ }. I can not believe no reviewer caught this.
3. $$e(W), f(W)$$ are more accurately described as $$e(W,X), f(W,X)$$.



**New Terms**
1. GRU Cells
2. Least squares N calculation

