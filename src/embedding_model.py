"""
Embedding model for generating vector representations of text.

Learns a low-dimensional embedding of each player's behaviour by training a
small autoencoder (encoder -> bottleneck -> decoder) to reconstruct the
scaled behavioural feature vector. The bottleneck activations are the
"experience embedding."

This is implemented from scratch in NumPy (full-batch gradient descent, one
hidden layer) rather than via a deep learning framework — the dataset here
is a few hundred rows and a handful of features, so a framework buys nothing
except an extra dependency. It's a genuinely learned, nonlinear (tanh)
compression, not just a relabelled PCA — worth being precise about that
distinction if this comes up in an interview, since "embedding model" can
otherwise be a vague claim.

Separate from clustering.py: clustering operates directly on the scaled
features, and this embedding is what recommendation_engine.py can optionally
draw on for a denser player representation. They're two different views of
the same underlying features, not a pipeline where one feeds the other. 
"""
