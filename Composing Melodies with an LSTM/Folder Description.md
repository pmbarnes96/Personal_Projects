# Composing Melodies with an LSTM

This folder is for simple melodies I composed with a model based on an LSTM [1].  The folder titled "Example Melodies" contains the sheet music for three melodies composed by my program as well as videos of me playing them on my cello.  The file titled "Music Training Generation and Composing.py" contains the code used to train the model and compose by sampling notes from the probability function fit by the model.

The model is trained on a number of basic melodies, especially holiday, folk, and simple classical tunes.  I wrote these tunes out by hand from memory or by looking through a book of pieces I practiced while learning the cello.

The melodies composed by the model tend not to contain too many large jumps between the pitches of notes, just like most real songs.  Upon examining the longer scale behavior of the melodies, one finds they tend to be in C major just like the melodies in the training set.  They are not atonal, and they do not seem to be in any of the other six keys derivable from the notes of the C major scale, such as A minor or G mixolydian.  One long scale tendency the model does not seem to have learned, however, is organizing music into 2, 4, or 8 bar phrases.

Bibliography:  
[1] https://pubmed.ncbi.nlm.nih.gov/9377276/
