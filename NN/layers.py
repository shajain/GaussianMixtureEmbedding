from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras import activations


class Polynomial(Layer):

  def __init__(self, minx):
      super(Polynomial, self).__init__()
      self.minx = minx
      #self.degree = degree

  def build(self, input_shape):
      self.w = self.add_weight(shape=(2,),
                               initializer='random_normal',
                               trainable=True)
      self.b = self.add_weight(shape=(1,),
                               initializer='random_normal',
                               trainable=True)

  def call(self, inputs):
      out = (inputs-self.minx) * self.w[0] + ((inputs-self.minx)**2)*self.w[1] + self.b
      return out


class PolynomialPos(Layer):

  def __init__(self, minx):
      super(PolynomialPos, self).__init__()
      self.minx = minx
      #self.degree = degree

  def build(self, input_shape):
      self.w = self.add_weight(shape=(2,),
                               initializer='random_normal',
                               trainable=True)
      self.b = self.add_weight(shape=(1,),
                               initializer='random_normal',
                               trainable=True)

  def call(self, inputs):
      out = (inputs-self.minx) * tf.abs(self.w[0]) + ((inputs-self.minx)**2)*tf.abs(self.w[1]) + tf.abs(self.b)
      return out



class Constant(Layer):

  def __init__(self):
      super(Constant, self).__init__()
      #self.degree = degree

  def build(self, input_shape):
      self.const = self.add_weight(shape=(1,),
                               initializer='random_normal',
                               trainable=True)

  def call(self, inputs):
      return tf.zeros_like(inputs) + self.const

