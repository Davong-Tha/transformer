# Summary 
A trasnformer implementation using pytorch. This is an encoder-decoder style transformer-based language model trained on Twitter's post dataset. The goal is to predict the reply of a given post. The model achieve near perfect memorization of the test set due to small sized datatset used. 
# Repo Content 
- multi_headed_attention.py: an implementation of multi-headed attention mechanism with argument for passing in padding and casual attention mask. This is the foundation upon which the encoder and decoder module is built on. 
- encoder.py: consist of multi_headed_attention, linear layer, normalization layer and drop out layer. This is used to encode the global context of our input and pass it to the decoder.
- decoder.py: consist of similar module to the decoder, however, its argument accept two mask: the padding mask for the decoder itself and the cross attention mask for the output of the encoder 
- transfomer.py: an implementation of the overall transformer model. Consist of n stacks of encoder and decoder layer as well as function for training and prediction. 
# Key Features

## Triangular mask
During the forward function inside multi_headed_attention.py, we produce an attention matrice which represent the "attention score" of one token to every other token. This "attention score" represent how closely related each token is to each other. However, we must ensure that the current token only have access to the token before it, not after. This ensure that the model can't see future context beforehand. To this extent, an upper traingular mask is used to invalidate the attention score of future token. In a nutshell, the attention score is set to -inf which ensure that it will output to zero after applying softmax function. 

## padding mask
Batch processing required every training data to have the same length. This is a requirement for such parallelization but in seq2seq modeling such as our twitter dataset, not every tweet is the same length. Therefore, in order to take advantage of parallization, we chose a max lenght and pad any sequence whose len is shorter than that max lenght. However, padding (usually all zero) could mess with our training by contaminating the gradient, therefore a padding mask is used to invalidate those padding timestep. Similar to triangualr mask, we set the attention score to -inf which ensure output it zero after softmax. 
## Positional Embedding
A key design feature of the attention mechanism is that it doesn't have any awareness of the order between token. Therefore, in order to enforce, in this case, the order of word in the sentence, we inject positional information through Sinusoidal Positional Encoding.
## Dropout
a dropout of 0.1 was used for both the encoder and decoder in order to ensure the model doesn't overfit during training.

## Teacher Forcing and objective function

## Text Generation Script

# Result

## Dataset

## train and validation loss

## Generation quality 

# Model Detail

# lesson learnt and troubleshooting 

# Limitation 