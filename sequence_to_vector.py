# std lib imports
from typing import Dict
# external libs
import tensorflow as tf
from tensorflow.keras import layers, models

class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """

    # def __init__(self, input_dim: int, num_layers: int):
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        self.num_layers=num_layers
        self.hidden_layer= layers.Dense(self._input_dim,activation=tf.nn.relu)  #Setting activation as ReLu for adding non-linearity.
        self.dropout=dropout

        # TODO(students): end



    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start

        #Removing padding tokens in vector_sequence with the help of sequence mask:
        unmasked_vector_seq = tf.multiply(vector_sequence, sequence_mask[:, :, tf.newaxis])

        # Bernoulli Distribution with probability 'p':
        mask = tf.random.uniform((unmasked_vector_seq.shape), minval=0, maxval=1, dtype=tf.float32, seed=1024)
        condition = tf.greater(mask, self.dropout)  # applying dropout condition to retain words.
        mask_1_0 = tf.cast(condition, tf.float32)   # converting True/False to 1s and 0s.
        drop_vector = tf.multiply(unmasked_vector_seq, mask_1_0)    #applying mask to original unmasked word sequence.


        avg_vector_seq = tf.math.reduce_mean(drop_vector, axis=1)


        layer_representations = []
        dense_layer = avg_vector_seq
        for i in range(self.num_layers):
            dense_layer = self.hidden_layer(dense_layer)
            combined_vector = dense_layer
            layer_representations.append(dense_layer)
        layer_representations = tf.stack(layer_representations, axis=1)
        return {"combined_vector": combined_vector,
                    "layer_representations": layer_representations}

class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.num_layers=num_layers
        self.hidden_layer= layers.GRU(self._input_dim,activation='tanh',return_sequences=True)  # Using tanh activation for each layer , and using return_sequences to capture current representation at each step.
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        # ...


        layer_representations=[]

        for i in range(self.num_layers):
            gru = self.hidden_layer         ## Initialising GRU
            if i==0:
                hidden_layer = gru(vector_sequence, mask=sequence_mask)     # Applying sequence mask to vector_sequence only for 1st layer, and sending as input to the GRU layer. .
            else:
                hidden_layer = gru(hidden_layer)            # Passing output of previous layers to current layer

            ## Storing the result at last time step for each layer.
            combined_vector = hidden_layer[:,-1,:]

            layer_representations.append(combined_vector)       #Storing each layers output at last time step in a list.

        layer_representations = tf.stack(layer_representations, axis =1)    #Converting to tensor along axis=1


        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}