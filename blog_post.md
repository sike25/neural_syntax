# "Neural Babel: What Do Neural Networks Talk About?"
Sike Ogieva, Amherst College. 

Imagine overhearing a conversation in a language you don’t speak. 
The speakers understand each other perfectly, but you have no idea what they're saying. In this project, the speakers were neural networks, 
and the language emerged spontaneously when they were trained to collaboratively solve a task. The goal was to build a translator for this “neuralese” and this is what I found when I tried.

### Teaching Machines to Point at Things
Andreas et al (2017) (cite) developed a game on the GENX dataset (cite)
Diagram explaining the game
Other possible tasks…(driving, cite), same task but with images (cite)
In this, I forewent the GENX dataset for a simplified one of I generated
Diagram explaining how they were generated
Mention Number of rules, samples, etc

### Conference
We built the speaker-listenerr…. Blah blah blan introduction
Architecture
Object Encoder → Speaker → Neuralese → Listener
Explain the architecture and the reasoning behind those choices.
Cite object encoder research
Clean diagram showing the flow and input, output sizes

Training
The shuffling innovation to prevent learning masks
They are trained end-to-end… encoder to listener
Final accuracy on test set
Cross-Rule Validation, Original poor numbers (Within-rule:  0.908  ± 0.090, Between-rule: 0.865 ± 0.097)
Large baseline discovered and subsequent Normalization Insight

### Translation
Set-Up
Explain what we are doing and why (interpretability, trust, debugging)
The hypothesis: neuralese can be translated to natural language
Building the translator network (n times bigger then the speaker-listener) and trained for n times longer.
Present raw test accuracies for token and sequence accuracies

Probe and Adjusted Metrics
1. Some predicted rules accurately describe X but are not the ground truth. Give an example.
2. Some predicted rules accurately describe the objects in X, but do not comprehensively exclude the objects not in X. Give an example.
Report these adjusted metrics.
Then bring up the insight that only 60% of the predicted rules are valid. Give examples of rules that aren’t valid… So if we look at the adjusted metrics only for rules which are valid, what happens? Report those numbers.

### Conclusions

What the cross-rule validation tells us about neuralese. The concept of the rules exists there, but if it were theoretically equal to the selective rules- the cosine similarities should be almost 1 at least, not 0.3. It encodes other info (perhaps about the specific world) that helps the system solve the task.
The problem with batching the data… is that while shuffling removes the incentive for the speaker to learn how to tell the explicitly listener about the mask X ans where all the selected objects are position. It might learn other metadata about X (how many selected objects, for ex) so this is one short cut where the model could partially exploit ans create incentive to learn about things apart from the rule.

What the adjusted metric probes tell us. An MLP is not a great translator. Learning how to structure the three-token naturall language rules consistently should have been solved… Just 60% is abysmal. I wonder weather this architecture is not the bottleneck here and I will switch it out on my next iteration of this notebook.

It is interesting though that on the rules it can structure, it does much better. It backs up the cross-rule valid experiment that there is some concepts of riles learned
"When debugging multi-agent systems or trying to interpret model behavior, we often assume that effective solutions will be human-interpretable. This assumption can be costly. This post demonstrates why probing for human-like representations might fail even when the system works perfectly, and what that means for interpretability research."

### Interaction
Link to Github
Interactive version up in a few days

