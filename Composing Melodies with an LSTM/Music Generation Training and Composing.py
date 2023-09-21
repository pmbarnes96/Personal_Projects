import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.keras.utils.set_random_seed(0)

my_csv = pd.read_csv("C:/Users/Patrick Barnes/Documents/Python 3 Stuff/Music Generation/Songs.csv") #CSV with songs in it

##### Dictionaries to go from the names of pitches, e.g. 'c' or 'la' (stands for Low A),
##### and the names of durations e.g. 'dq' (stands for dotted quarter), to an integer key number.
pitch_name_to_number = {'lg':0, 'la':1, 'lb':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'a':8, 'b':9, 'hc':10, 'r':11}
duration_name_to_number = {'w':0, 'dh':1, 'h':2, 'dq':3, 'q':4, 'e':5}
number_to_pitch_name = {0:'lg', 1:'la', 2:'lb', 3:'c', 4:'d', 5:'e', 6:'f', 7:'g', 8:'a', 9:'b', 10:'hc', 11:'r'}
number_to_duration_name = {0:'w', 1:'dh', 2:'h', 3:'dq', 4:'q', 5:'e'}

num_pitches = len(pitch_name_to_number)
num_durations = len(duration_name_to_number)
one_hot_len = num_pitches * num_durations + 3 # length of one hot representations of notes
input_rep_len = one_hot_len # length of representations of notes input into the LSTM

##### Function to go from the names of a note's pitch and duration to a one hot representation
def names_to_one_hot(pitch_name, duration_name):
    if pitch_name == 'n': # Indicates a new measure
        one_hot_number = one_hot_len - 3
    elif pitch_name == 'm': #Indicates the end of piece
        one_hot_number = one_hot_len - 2
    elif pitch_name == 'v': #Indicates the piece has already ended
        one_hot_number = one_hot_len - 1
    else:
        pitch_number = pitch_name_to_number[pitch_name]
        duration_number = duration_name_to_number[duration_name]
        one_hot_number = pitch_number * num_durations + duration_number
    one_hot = np.zeros(one_hot_len)
    one_hot[one_hot_number] = 1
    return one_hot

##### Function to go from the one hot index of a note to the name of its pitch and duration, the inverse of names_to_one_hot
def one_hot_number_to_names(one_hot_number):
    if one_hot_number == one_hot_len - 3:
        pitch_name = 'n'
        duration_name = 'n'
    elif one_hot_number == one_hot_len - 2:
        pitch_name = 'm'
        duration_name = 'm'
    elif one_hot_number == one_hot_len - 1:
        pitch_name = 'v'
        duration_name = 'v'
    else:
        pitch_number = one_hot_number // num_durations
        duration_number = one_hot_number % num_durations
        pitch_name = number_to_pitch_name[pitch_number]
        duration_name = number_to_duration_name[duration_number]
    return [pitch_name, duration_name]

##### Function to go from the one hot index of a note to the numerical value of its duration, e.g. 1/4 for a quarter note
def one_hot_number_to_duration_value(one_hot_number):
    if one_hot_number==num_pitches*num_durations or one_hot_number==num_pitches*num_durations+1 or one_hot_number==num_pitches*num_durations+2:
        duration_value = 0
    else:
        duration_number = one_hot_number % num_durations
        duration_name = number_to_duration_name[duration_number]
        if duration_name == 'w':
            duration_value = 1
        elif duration_name == 'dh':
            duration_value = 3/4
        elif duration_name == 'h':
            duration_value = 1/2
        elif duration_name == 'dq':
            duration_value = 3/8
        elif duration_name == 'q':
            duration_value = 1/4
        elif duration_name == 'e':
            duration_value = 1/8
    return duration_value

##### Function to go from the names of a note's pitch and duration to the representation input into the LSTM
def names_to_input_rep(pitch_name, duration_name):
    input_rep = names_to_one_hot(pitch_name, duration_name)
    return input_rep
    
##### Begin to turn the CSV file into usable training data
m = 18 #Number of melodies used in training
Tx = len(my_csv)
Y = np.zeros((Tx, m, one_hot_len))

##### Target values of the training set
for i in range(m):
    current_song = my_csv.iloc[:, 3*i:3*i+2].to_numpy()
    for t in range(Tx):
        Y[t, i, :] = names_to_one_hot(current_song[t, 0], current_song[t, 1])

X = np.zeros((m, Tx, input_rep_len))

##### Input values of the training set
for i in range(m):
    current_song = my_csv.iloc[:, 3*i:3*i+2].to_numpy()
    for t in range(Tx-1):
        X[i, t+1, :] = names_to_input_rep(current_song[t, 0], current_song[t, 1])

n_a = 64 # number of dimensions for the hidden state of each LSTM layer
reshaper = Reshape((1, input_rep_len))
LSTM_layer = LSTM(n_a, return_state = True)
dense_layer = Dense(one_hot_len, activation='softmax')

def full_model_creator(Tx, LSTM_layer, dense_layer, reshaper):
    """
    Arguments:
        Tx -- length of the sequences in the training set
        LSTM_layer -- LSTM layer
        dense_layer -- Dense layer
        reshaper -- Reshape layer
    
    Returns:
        model -- a keras model with inputs [X, a0, c0]
    """
    
    X = Input(shape = (Tx, input_rep_len))
    a0 = Input(shape = n_a)
    c0 = Input(shape = n_a)
    a = a0 # LSTM hidden state
    c = c0 # LSTM cell state
    outputs = [] # empty list to append the outputs while iterating
    
    ##### Loop over Tx
    for t in range(Tx): 
        x = X[:,t,:] # Select the "t"th time step vector from X.
        x = reshaper(x) # Use reshaper to reshape x to be (1, one_hot_len)
        a, _, c = LSTM_layer(inputs = x, initial_state = [a, c]) # Perform one step of the LSTM_layer
        out = dense_layer(a) # Apply dense_layer to the hidden state output of LSTM_layer
        outputs.append(out) # Add the output to "outputs"
        
    model = Model(inputs=[X, a0, c0], outputs=outputs) # Create model instance
    
    return model

model = full_model_creator(Tx = Tx, LSTM_layer = LSTM_layer, dense_layer = dense_layer, reshaper = reshaper)

opt = Adam(learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)

model.compile(optimizer = opt, loss = 'categorical_crossentropy')

a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

model.fit([X, a0, c0], list(Y), epochs=400, verbose = 1)

##### Function to create a model that uses the LSTM layer directly
##### The model created by this will take in a note, a hidden state, and a cell state, and return a new hidden state and cell state
def LSTM_unit_model_creator():
    x0 = Input(shape = (1, input_rep_len))
    a0 = Input(shape = n_a)
    c0 = Input(shape = n_a)
    a, _, c = LSTM_layer(inputs = x0, initial_state = [a0, c0])
    model = Model(inputs = [x0, a0, c0], outputs = [a, c])
    return model

##### Function to create a model that uses the dense layer directly
##### The model created by this will take in the hidden state from the LSTM and output the probabilities of each note coming next
def dense_unit_model_creator():
    a = Input(shape = n_a)
    y_hat = dense_layer(a)
    model = Model(inputs = a, outputs = y_hat)
    return model

##### Create the LSTM and dense layer models to predict the next note
LSTM_unit_model = LSTM_unit_model_creator()
dense_unit_model = dense_unit_model_creator()

##### Function to determine if two values are approximately equal
def approx_equal(x, y, eps = 1e-4):
    if abs(x-y) < eps:
        return True
    return False

##### Compose some pieces
num_pieces_to_compose = 50
max_piece_len = 200
pieces = []
for i in range(num_pieces_to_compose):
    x = np.zeros((1, 1, input_rep_len)) # Input a zero vector to calculate first note of piece
    a = np.zeros((1, n_a)) # Initialize hidden state
    c = np.zeros((1, n_a)) #Initialize cell state
    current_piece = [] #Empty list to put notes into
    measure_duration = 0
    for i in range(max_piece_len):
        a, c = LSTM_unit_model.predict([x, a, c]) # Get next hidden and cell states
        y_hat = dense_unit_model.predict(a) #Probabilities of the next note
        not_returned_good_note = True
        while not_returned_good_note: #Don't continue until we sample a valid next note
            one_hot_number = np.random.choice(one_hot_len, p = y_hat[0]) # Sample the next note from the probability distribution
            proposed_duration = measure_duration + one_hot_number_to_duration_value(one_hot_number) # How long will the current measure be if we add this note?
            ##### A valid next step could be beginning a new measure or ending the piece if we have just filled up a measure
            if approx_equal(measure_duration, 1) and (one_hot_number == one_hot_len - 3 or one_hot_number == one_hot_len - 2):
                not_returned_good_note = False
                measure_duration = 0
            ##### The other valid next step is adding a note that does not bleed into the next measure or necessitate that the next note will
            elif (proposed_duration < 1 or approx_equal(proposed_duration, 1)) and one_hot_number < one_hot_len - 3:
                not_returned_good_note = False
                measure_duration = proposed_duration
        the_names = one_hot_number_to_names(one_hot_number) # Go from one hot index to names of pitch and duration
        current_piece.append(the_names) #Add the names of pitch and duration to the list of notes that make up the composition
        if one_hot_number == one_hot_len - 2: # If the model has returned the "note" that signals the end of the piece, stop sampling new notes
            break
        x = np.zeros((1, 1, input_rep_len))
        x[0, 0, :] = names_to_input_rep(the_names[0], the_names[1]) # Prepare this note to be fed back into the LSTM to predict the next one
    pieces.append(current_piece)

##### Print the pieces
for i in range(num_pieces_to_compose):
    print()
    print()
    print("Piece", str(i))
    print(pieces[i])
