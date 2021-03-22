import numpy as np
import tensorflow as tf

from .mlp import MLP
from .gaussian_constant_policy import GaussianPolicy


class TRPOLearner:
    def __init__(self, state_len, action_len, name="", value_lr=0.001, value_hidden_len=128,
        value_gamma=0.96, max_kl=0.001, cg_damping=1e-3):

        self.max_kl = max_kl
        self.cg_damping = cg_damping
        with tf.variable_scope("TRPOLearner" + name): 
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_value = tf.placeholder(tf.float32, [None, 1], name="input_value")
            self.input_advantage = tf.placeholder(tf.float32, [None, 1], name="input_advantage")

            self.input_old_mu = tf.placeholder(tf.float32, [None, action_len], name="input_old_mu")
            self.input_old_logstd = tf.placeholder(tf.float32, [None, action_len], name="input_old_logstd")

            self.value_network = MLP("value", state_len, 1, value_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu)
            self.policy_network = GaussianPolicy("policy", state_len, action_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.tanh, output_nonlinearity=tf.nn.tanh)

            # Value network
            self.value_loss = tf.reduce_mean((self.value_network.layer_output - self.input_value) ** 2)
            self.value_train = tf.train.AdamOptimizer(value_lr).minimize(loss = self.value_loss, var_list = self.value_network.trainable_params)

            # Policy network
            batch_size = tf.cast(tf.shape(self.input_action)[0], tf.float32)
            self.kl = gauss_KL(self.input_old_mu, self.input_old_logstd, self.policy_network.mu, self.policy_network.logstd) / batch_size
            self.old_dist = gauss_log_prob(self.input_old_mu, self.input_old_logstd, self.input_action)
            self.new_dist = gauss_log_prob(self.policy_network.mu, self.policy_network.logstd, self.input_action)
            self.surr_loss = -tf.reduce_mean(self.input_advantage * tf.exp(self.new_dist - self.old_dist))
            print("self.old_dist", self.old_dist.shape)
            print("self.new_dist", self.new_dist.shape)
            print("self.input_advantage", self.input_advantage.shape)
            print("self.surr_loss", self.surr_loss.shape)

            self.pg = flatgrad(self.surr_loss, self.policy_network.trainable_params)
            self.gf = tf.concat([tf.reshape(v, [np.prod(var_shape(v))]) for v in self.policy_network.trainable_params], 0)

            kl_firstfixed = gauss_selfKL_firstfixed(self.policy_network.mu, self.policy_network.logstd) / batch_size
            grads = tf.gradients(kl_firstfixed, self.policy_network.trainable_params)
            shapes = [var_shape(v) for v in self.policy_network.trainable_params]
            self.input_flat_tangent = tf.placeholder(tf.float32, [sum([np.prod(shape) for shape in shapes])])
            self.input_theta = tf.placeholder(tf.float32, [sum([np.prod(shape) for shape in shapes])])
            start = 0
            tangents = []
            self.set_from_flat = []
            for (shape, v) in zip(shapes, self.policy_network.trainable_params):
                size = np.prod(shape)
                tangents.append(tf.reshape(self.input_flat_tangent[start:(start + size)], shape))
                self.set_from_flat.append(tf.assign(v, tf.reshape(self.input_theta[start:start + size], shape)))
                start += size
            gvp = tf.reduce_sum([tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)])
            self.fvp = flatgrad(gvp, self.policy_network.trainable_params)

            


    def get_action_stochastic(self, input_state):
        sess = tf.get_default_session()
        mu, std = sess.run([self.policy_network.mu, self.policy_network.logstd], {self.input_state : np.array([input_state])})
        return mu[0], std[0]

    def get_expected_value(self, input_state):
        sess = tf.get_default_session()
        value = sess.run(self.value_network.layer_output, {self.input_state : np.array([input_state])})
        return value[0][0]

    def optimize_value_batch(self, input_state, input_value):
        sess = tf.get_default_session()
        _, loss = sess.run([self.value_train, self.value_loss], {self.input_state : input_state, self.input_value : input_value} )
        return loss

    def optimize_policy_batch(self, input_state, input_action, input_advantage, input_old_mu, input_old_logstd):
        sess = tf.get_default_session()
        input_list = { self.input_state : input_state, self.input_action : input_action, self.input_advantage : input_advantage,
            self.input_old_mu : input_old_mu, self.input_old_logstd : input_old_logstd }
        
        new_dist, old_dist = sess.run([self.new_dist, self.old_dist], input_list)
        g, thprev = sess.run([self.pg, self.gf], input_list)
        stepdir = self.conjugate_gradient(sess, -g, input_list)
        shs = 0.5 * stepdir.dot( self.fisher_vector_product(sess, stepdir, input_list) )
        lm = np.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm
        negative_g_dot_steppdir = -g.dot(stepdir)

        _, old_loss, new_loss = self.linesearch(sess, thprev, input_list, fullstep, negative_g_dot_steppdir/ lm)

        return old_loss, new_loss, np.sqrt(np.mean(fullstep**2))

    def fisher_vector_product(self, sess, p, input_orig_list):
        input_list = input_orig_list.copy()
        input_list[self.input_flat_tangent] = p
        return sess.run(self.fvp, input_list) + p * self.cg_damping

    def conjugate_gradient(self, sess, b, input_orig_list, cg_iters=10, residual_tol=1e-10):
        
        p = b.copy()
        r = b.copy()
        x = np.zeros_like(b)
        rdotr = r.dot(r)
        for i in range(cg_iters):
            z = self.fisher_vector_product(sess, p, input_orig_list)
            v = rdotr / p.dot(z)
            x += v * p
            r -= v * z
            newrdotr = r.dot(r)
            mu = newrdotr / rdotr
            p = r + mu * p
            rdotr = newrdotr
            if rdotr < residual_tol:
                break
        return x

    def linesearch(self, sess, x, input_list, fullstep, expected_improve_rate):
        print("Line Search Start")
        accept_ratio = .1
        max_backtracks = 10

        sess.run(self.set_from_flat, {self.input_theta : x})
        fval = sess.run(self.surr_loss, input_list)
        print("Original Loss : " + str(fval))
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep

            sess.run(self.set_from_flat, {self.input_theta : xnew})
            newfval = sess.run(self.surr_loss, input_list)
            print("Try " + str(stepfrac) + " New Loss : " + str(newfval))
            if np.isnan(newfval):
                break
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            if ratio > accept_ratio and actual_improve > 0:
                return xnew, fval, newfval
        print("Line Search Failed")
        sess.run(self.set_from_flat, {self.input_theta : x})
        return x, fval, None

def gauss_selfKL_firstfixed(mu, logstd):
    mu1, logstd1 = tf.stop_gradient(mu), tf.stop_gradient(logstd) 
    mu2, logstd2 = mu, logstd

    return gauss_KL(mu1, logstd1, mu2, logstd2)

def gauss_log_prob(mu, logstd, x):
    var = tf.exp(logstd * 2)
    gp = -tf.square(x - mu) / (2 * var) - 0.5*tf.log(tf.constant(2*np.pi)) - logstd
    return  tf.reduce_mean(gp, 1, keepdims=True)

def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = tf.exp(logstd1 * 2)
    var2 = tf.exp(logstd2 * 2)

    kl = tf.reduce_mean(logstd1 - logstd2  + (var1 + tf.square(mu1 - mu2)) / (2 * var2 ) - 0.5)
    return kl

def gauss_ent(mu, logstd):
    h = tf.reduce_mean(logstd + tf.constant(0.5*np.log(2*np.pi*np.e), tf.float32))
    return h

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(grad, [np.prod(var_shape(v))]) for (v, grad) in zip(var_list, grads)], 0)


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    return out