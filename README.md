# GraphCodeSemantics
Code Structure
The implementation consists of two core components:

1. Enhanced TinyPDG with Code Embedding
Extracts Control Flow Graphs (CFGs) and Program Dependency Graphs (PDGs) using an improved TinyPDG approach.

Trains the CFG/PDG corpus with the Continuous Bag-of-Words (CBOW) model to generate code representations.

2. Hybrid Deep Learning for Feature Fusion
Introduces a modified PNIAT layer combining multi-head attention and BiLSTM to capture method-level semantic features.

Proposes two aggregation strategies:

Weighted summation

Linear fully-connected layers
to derive file-level semantic feature vectors.

Constructs joint features by integrating the above with manual annotation features.

Balances the dataset using SMOTETomek before training 11 machine learning classifiers for defect prediction.
