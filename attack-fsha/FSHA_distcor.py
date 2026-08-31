import tensorflow as tf
import defense
from FSHA import *

class FSHA_dc(FSHA):
    
    @tf.function
    def train_step(self, x_private, x_public, label_private, label_public):

        with tf.GradientTape(persistent=True) as tape:

            #### Virtually, ON THE CLIENT SIDE:
            # clients' smashed data
            z_private_raw = self.f(x_private, training=True)
            z_private = self.defend_features(z_private_raw, training=True)
            def_loss = tf.constant(0.0, dtype=x_private.dtype)
            
            if 'alpha1' in self.hparams:
                w_dc = self.hparams['alpha1']
                print("With client \alpha_1: %f" % w_dc)
                def_loss = defense.dist_corr(x_private, z_private_raw) * w_dc    
            ####################################


            #### SERVER-SIDE:
            # map to data space (for evaluation and style loss)
            rec_x_private = self.decoder(z_private, training=True)
            ## adversarial loss (f's output must similar be to \tilde{f}'s output):
            adv_private_logits = self.D(z_private)
            if self.hparams['WGAN']:
                print("Use WGAN loss")
                f_loss = tf.reduce_mean(adv_private_logits)
            else:
                f_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(adv_private_logits), adv_private_logits, from_logits=True))
            ##
            
            if 'alpha2' in self.hparams:
                w_loss = self.hparams['alpha2']
                print("Scale adversarial loss server %f" % w_loss)
                f_loss *= w_loss
            f_loss += self._wcc_regularizer(z_private, label_private)
            
            adv_f_loss = f_loss

            z_public_raw = self.tilde_f(x_public, training=True)
            z_public = self.defend_features(z_public_raw, training=True)

            # invertibility loss
            rec_x_public = self.decoder(z_public, training=True)
            public_rec_loss = distance_data_loss(x_public, rec_x_public)
            tilde_f_loss = public_rec_loss
                    
            # discriminator on attacker's feature-space
            adv_public_logits = self.D(z_public, training=True)
            if self.hparams['WGAN']:
                loss_discr_true = tf.reduce_mean( adv_public_logits )
                loss_discr_fake = -tf.reduce_mean( adv_private_logits)
                # discriminator's loss
                D_loss = loss_discr_true + loss_discr_fake
            else:
                loss_discr_true = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(adv_public_logits), adv_public_logits, from_logits=True))
                loss_discr_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(adv_private_logits), adv_private_logits, from_logits=True))
                # discriminator's loss
                D_loss = (loss_discr_true + loss_discr_fake) / 2

            if 'gradient_penalty' in self.hparams:
                print("Use GP")
                w = float(self.hparams['gradient_penalty'])
                D_gradient_penalty = self.gradient_penalty(z_private, z_public)
                D_loss += D_gradient_penalty * w

            ##################################################################
            ## attack validation #####################
            loss_c_verification = distance_data(x_private, rec_x_private)
            ############################################
            ##################################################################

        # train attacker's autoencoder on public data
        var = self._attacker_autoencoder_variables()
        self._apply_gradients(tape, tilde_f_loss, var, self.optimizer1)

        # train discriminator
        var = self.D.trainable_variables
        self._apply_gradients(tape, D_loss, var, self.optimizer2)
        
        # train client's network privacy loss
        var = self.f.trainable_variables
        self._apply_gradients(tape, def_loss, var, self.optimizer0)

        # train client's network 
        var = self._client_trainable_variables()
        self._apply_gradients(tape, f_loss, var, self.optimizer0)

        return adv_f_loss, tilde_f_loss, D_loss, loss_c_verification
