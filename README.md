# Toxic comment detection using Transformers **BERT**

In this work, we propose an NLP system that detects toxicity in social media comments by taking into account the general context of these, while identifying the terms that make a comment toxic. The system takes a comment as input and determines whether it is toxic or not before publishing it.
We will use the BERT transformers for toxicity detection using the public multi-label dataset of over 200,000 user comments.

We will use the ROC-AUC score, the F1 score and the Accuracy score to evaluate the performance of the model. The ROC-AUC score is suitable for binary classification and will tell us if our model is able to differentiate toxic comments from non-toxic ones. The F1 score us
is useful because we are faced with unbalanced classes (more non-toxic comments than toxic ones). 

The Accuracy score will allow us to evaluate the training of the model.
