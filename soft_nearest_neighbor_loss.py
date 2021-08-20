import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np 

class SNNLCrossEntropy():
    STABILITY_EPS = 0.00001
    def __init__(self,
               model,
               temperature=100.,
               layer_names=None,
               factor=-10.,
               optimize_temperature=True,
               cos_distance=True):
        
        self.temperature = temperature
        self.factor = factor
        self.optimize_temperature = optimize_temperature
        self.cos_distance = cos_distance
        self.layer_names = layer_names
        self.model = model
        if not layer_names:
            self.layer_names = [layer.name for layer in model.layers][:-1]
        print(self.layer_names)
    
    @staticmethod
    def pairwise_euclid_distance(A, B):
        """Pairwise Euclidean distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise Euclidean between A and B.
        """
        batchA = tf.shape(A)[0]
        batchB = tf.shape(B)[0]

        sqr_norm_A = tf.reshape(tf.reduce_sum(tf.pow(A, 2), 1), [1, batchA])
        sqr_norm_B = tf.reshape(tf.reduce_sum(tf.pow(B, 2), 1), [batchB, 1])
        inner_prod = tf.matmul(B, A, transpose_b=True)

        tile_1 = tf.tile(sqr_norm_A, [batchB, 1])
        tile_2 = tf.tile(sqr_norm_B, [1, batchA])
        return (tile_1 + tile_2 - 2 * inner_prod)
    
    @staticmethod
    def pairwise_cos_distance(A, B):
        
        """Pairwise cosine distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise cosine between A and B.
        """
        normalized_A = tf.nn.l2_normalize(A, axis=1)
        normalized_B = tf.nn.l2_normalize(B, axis=1)
        prod = tf.matmul(normalized_A, normalized_B, adjoint_b=True)
        return 1 - prod
    
    @staticmethod
    def fits(A, B, temp, cos_distance):
        if cos_distance:
            distance_matrix = SNNLCrossEntropy.pairwise_cos_distance(A, B)
        else:
            distance_matrix = SNNLCrossEntropy.pairwise_euclid_distance(A, B)
            
        return tf.exp(-(distance_matrix / temp))
    
    @staticmethod
    def pick_probability(x, temp, cos_distance):
        """Row normalized exponentiated pairwise distance between all the elements
        of x. Conceptualized as the probability of sampling a neighbor point for
        every element of x, proportional to the distance between the points.
        :param x: a matrix
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or euclidean distance
        :returns: A tensor for the row normalized exponentiated pairwise distance
                  between all the elements of x.
        """
        f = SNNLCrossEntropy.fits(x, x, temp, cos_distance) - tf.eye(tf.shape(x)[0])
        return f / (SNNLCrossEntropy.STABILITY_EPS + tf.expand_dims(tf.reduce_sum(f, 1), 1))
    
    @staticmethod
    def same_label_mask(y, y2):
        """Masking matrix such that element i,j is 1 iff y[i] == y2[i].
        :param y: a list of labels
        :param y2: a list of labels
        :returns: A tensor for the masking matrix.
        """
        return tf.cast(tf.squeeze(tf.equal(y, tf.expand_dims(y2, 1))), tf.float32)
    
    @staticmethod
    def masked_pick_probability(x, y, temp, cos_distance):
        """The pairwise sampling probabilities for the elements of x for neighbor
        points which share labels.
        :param x: a matrix
        :param y: a list of labels for each element of x
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or Euclidean distance
        :returns: A tensor for the pairwise sampling probabilities.
        """
        return SNNLCrossEntropy.pick_probability(x, temp, cos_distance) * \
                                    SNNLCrossEntropy.same_label_mask(y, y)
    
    @staticmethod
    def SNNL(x, y, temp, cos_distance):
        """Soft Nearest Neighbor Loss
        :param x: a matrix.
        :param y: a list of labels for each element of x.
        :param temp: Temperature.
        :cos_distance: Boolean for using cosine or Euclidean distance.
        :returns: A tensor for the Soft Nearest Neighbor Loss of the points
                  in x with labels y.
        """
        summed_masked_pick_prob = tf.reduce_sum(
            SNNLCrossEntropy.masked_pick_probability(x, y, temp, cos_distance), 1)
        return tf.reduce_mean(
            -tf.math.log(SNNLCrossEntropy.STABILITY_EPS + summed_masked_pick_prob))
    
    @staticmethod
    def optimized_temp_SNNL(x, y, initial_temp, cos_distance):
                
        def inverse_temp(t):
            return tf.math.divide(initial_temp,t)
        
        t = tf.Variable(1., dtype=tf.float32, trainable=False, name="temp")
        
        with tf.GradientTape() as tap:
            tap.watch(t)
            ent_loss = SNNLCrossEntropy.SNNL(x, y, inverse_temp(t), cos_distance)
                
            t.assign_sub(t, tf.subtract(t, 0.1*tap.gradient(ent_loss, t)))
            
        inverse_t = inverse_temp(t)
        return SNNLCrossEntropy.SNNL(x, y, inverse_t, cos_distance)
    
    def fprop(self, x, y):
        inp = self.model.input          
        outputs = [layer.output for layer in self.model.layers]
        functor = [K.function([inp], out) for out in outputs]
        
        self.layer_output = [func([x,]) for func in functor]
        
        loss_fn = self.SNNL
        
        
        if self.optimize_temperature:
            loss_fn = self.optimized_temp_SNNL
            
        layers_SNNL = [loss_fn(
                               tf.keras.layers.Flatten()(layer),
                               y,
                               self.temperature,
                               self.cos_distance,
                                )
                       for layer in self.layer_output] 
        
        return  self.factor * tf.add_n(layers_SNNL)