from networks import *


class PolicyVNetwork(Network):

    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient
        compute and apply ops, network parameter synchronization ops, and
        summary ops. """

        super(PolicyVNetwork, self).__init__(conf)

        self.entropy_regularisation_strength = conf['entropy_regularisation_strength']

        with tf.device(conf['device']):
            with tf.name_scope(self.name):

                self.critic_target_ph = tf.placeholder(
                    "float32", [None], name='target')
                self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')

                # Final actor layer
                layer_name = 'actor_output'
                _, _, self.output_layer_pi = softmax(layer_name, self.output, self.num_actions)
                # Final critic layer
                _, _, self.output_layer_v = fc('critic_output', self.output, 1, activation="linear")

                # Avoiding log(0) by adding a very small quantity (1e-30) to output.
                self.log_output_layer_pi = tf.log(tf.add(self.output_layer_pi, tf.constant(1e-30)),
                                                  name=layer_name + '_log_policy')

                # Entropy: sum_a (-p_a ln p_a)
                self.output_layer_entropy = tf.reduce_sum(tf.multiply(
                    tf.constant(-1.0),
                    tf.multiply(self.output_layer_pi, self.log_output_layer_pi)), reduction_indices=1)

                self.output_layer_v = tf.reshape(self.output_layer_v, [-1])

                # Advantage critic
                self.critic_loss = tf.subtract(self.critic_target_ph, self.output_layer_v)

                log_output_selected_action = tf.reduce_sum(
                    tf.multiply(self.log_output_layer_pi, self.selected_action_ph),
                    reduction_indices=1)

                self.actor_objective_advantage_term = tf.multiply(log_output_selected_action, self.adv_actor_ph)
                self.actor_objective_entropy_term = tf.multiply(self.entropy_regularisation_strength, self.output_layer_entropy)

                self.actor_objective_mean = tf.reduce_mean(tf.multiply(tf.constant(-1.0),
                                                                       tf.add(self.actor_objective_advantage_term, self.actor_objective_entropy_term)),
                                                           name='mean_actor_objective')

                self.critic_loss_mean = tf.reduce_mean(tf.scalar_mul(0.25, tf.pow(self.critic_loss, 2)), name='mean_critic_loss')

                # Loss scaling is used because the learning rate was initially runed tuned to be used with
                # max_local_steps = 5 and summing over timesteps, which is now replaced with the mean.
                self.loss = tf.scalar_mul(self.loss_scaling, self.actor_objective_mean + self.critic_loss_mean)

class NIPSPolicyVNetwork(PolicyVNetwork, NIPSNetwork):
    pass


class NaturePolicyVNetwork(PolicyVNetwork, NatureNetwork):
    pass
