import tensorflow as tf
import numpy as np
try:
    import tqdm
except ImportError:
    class _TqdmFallback:
        @staticmethod
        def tqdm(iterator, *args, **kwargs):
            del args, kwargs
            return iterator
    tqdm = _TqdmFallback()
import architectures, bottlenecks

def distance_data_loss(a,b):
    l = tf.losses.MeanSquaredError()
    return l(a, b)

def distance_data(a,b):
    l = tf.losses.MeanSquaredError()
    return l(a, b)

class FSHA:
    
    def loadBiasNetwork(self, make_decoder, z_shape, channels, input_shape=None):
        if len(z_shape) == 1:
            return architectures.decoder_vector(z_shape, target_shape=input_shape, channels=channels)
        return make_decoder(z_shape, channels=channels)
        
    def __init__(self, xpriv, xpub, id_setup, batch_size, hparams):
            input_shape = xpriv.element_spec[0].shape
            input_shape_tuple = tuple(int(v) for v in input_shape)
            
            self.hparams = hparams

            # setup dataset
            self.client_dataset = xpriv.batch(batch_size, drop_remainder=True).repeat(-1)
            self.attacker_dataset = xpub.batch(batch_size, drop_remainder=True).repeat(-1)
            self.batch_size = batch_size

            ## setup models
            make_f, make_tilde_f, make_decoder, make_D = architectures.SETUPS[id_setup]

            self.f = make_f(input_shape)
            self.tilde_f = make_tilde_f(input_shape)

            assert self.f.output.shape.as_list()[1:] == self.tilde_f.output.shape.as_list()[1:]
            self.raw_z_shape = tuple(self.tilde_f.output.shape.as_list()[1:])

            self.bottleneck_mode = hparams.get('bottleneck_mode', 'none')
            self.compression_ratio = float(hparams.get('compression_ratio', 1.0))
            self.bottleneck_k = hparams.get('bottleneck_k', hparams.get('k', None))
            self.wcc_lambda = float(hparams.get('wcc_lambda', 0.0))
            self._warned_missing_wcc_labels = False
            self.block_size = int(hparams.get('block_size', 2))

            self.bottleneck = bottlenecks.make_bottleneck(
                self.bottleneck_mode,
                self.raw_z_shape,
                compression_ratio=self.compression_ratio,
                bottleneck_k=self.bottleneck_k,
                block_size=self.block_size,
            )
            z_shape = self.bottleneck.output_shape_

            # DP on the transmitted payload: per-sample L2 clip (bounds sensitivity) then
            # calibrated Gaussian noise (std = noise_multiplier * clip). Applied in
            # defend_features so the server AND the attacker receive a noised payload; the
            # clip is calibrated to the median payload L2 (see calibrate_dp), matching the
            # FORA/UnSplit recipe. dp_noise_multiplier<=0 => DP off.
            self.dp_noise_multiplier = float(hparams.get('dp_noise_multiplier', 0.0))
            self.dp_clip = tf.Variable(
                float(hparams.get('dp_clip', 0.0)), trainable=False, dtype=tf.float32, name="dp_clip"
            )

            if len(z_shape) == 1:
                self.D = architectures.discriminator_vector(z_shape)
            else:
                self.D = make_D(z_shape)
            self.decoder = self.loadBiasNetwork(
                make_decoder,
                z_shape,
                channels=input_shape_tuple[-1],
                input_shape=input_shape_tuple,
            )

            if self.wcc_lambda > 0.0 and not self.bottleneck.supports_wcc:
                print("[BOTTLENECK] wcc_lambda=%s ignored for mode=%s; WCC is enabled only for fixed_random." % (
                    self.wcc_lambda,
                    self.bottleneck_mode,
                ))

            print("[BOTTLENECK] mode=%s raw_shape=%s defended_shape=%s compression_ratio=%0.4f k=%s wcc_lambda=%0.4f" % (
                self.bottleneck_mode,
                self.raw_z_shape,
                z_shape,
                self.bottleneck.effective_compression_ratio,
                self.bottleneck.k,
                self.wcc_lambda,
            ))

            # setup optimizers
            self.optimizer0 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_f'])
            self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_tilde'])
            self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_D'])



    @staticmethod
    def addNoise(x, alpha):
        return x + tf.random.normal(x.shape) * alpha

    def defend_features(self, z, training=False):
        payload = self.bottleneck(z, training=training)
        return self._apply_dp(payload)

    def _apply_dp(self, payload):
        # Python-level guard: resolved at trace time, so no DP op is traced when DP is off.
        if self.dp_noise_multiplier <= 0.0:
            return payload
        flat = tf.reshape(payload, [tf.shape(payload)[0], -1])
        norms = tf.norm(flat, axis=1, keepdims=True)
        clip = tf.cast(self.dp_clip, flat.dtype)
        factor = tf.minimum(clip / (norms + 1e-8), 1.0)
        flat = flat * factor
        flat = flat + tf.random.normal(tf.shape(flat), dtype=flat.dtype) * (
            tf.cast(self.dp_noise_multiplier, flat.dtype) * clip
        )
        return tf.reshape(flat, tf.shape(payload))

    def calibrate_dp(self, num_batches=8):
        """Set the DP clip to the median per-sample payload L2 over a few client batches."""
        if self.dp_noise_multiplier <= 0.0:
            return
        norms = []
        for x_private, _label in self.client_dataset.take(num_batches):
            payload = self.bottleneck(self.f(x_private, training=False), training=False)
            flat = tf.reshape(payload, [tf.shape(payload)[0], -1])
            norms.append(tf.norm(flat, axis=1).numpy())
        med = float(np.median(np.concatenate(norms)))
        self.dp_clip.assign(med)
        print("[DP] calibrated clip=%0.4f (median payload L2); noise_mult=%0.3f -> noise_std=%0.5f" % (
            med, self.dp_noise_multiplier, self.dp_noise_multiplier * med))

    def _client_trainable_variables(self):
        return self.f.trainable_variables + self.bottleneck.trainable_variables

    def _attacker_autoencoder_variables(self):
        # The malicious server updates only its own shadow encoder and decoder. The client's
        # bottleneck is client-side state: the server reaches it only through the gradient it
        # returns across the cut (f_loss, see _client_trainable_variables), never directly.
        return self.tilde_f.trainable_variables + self.decoder.trainable_variables

    def _apply_gradients(self, tape, loss, variables, optimizer):
        gradients = tape.gradient(loss, variables)
        grads_and_vars = [(g, v) for g, v in zip(gradients, variables) if g is not None]
        if grads_and_vars:
            optimizer.apply_gradients(grads_and_vars)

    def _wcc_regularizer(self, z_private, label_private):
        if self.wcc_lambda <= 0.0 or not self.bottleneck.supports_wcc:
            return tf.constant(0.0, dtype=z_private.dtype)
        if label_private is None:
            if not self._warned_missing_wcc_labels:
                print("[BOTTLENECK] WCC requested but labels are unavailable; disabling WCC for this path.")
                self._warned_missing_wcc_labels = True
            return tf.constant(0.0, dtype=z_private.dtype)
        return tf.cast(self.wcc_lambda, z_private.dtype) * bottlenecks.within_class_compaction_loss(z_private, label_private)

    @tf.function
    def train_step(self, x_private, x_public, label_private, label_public):

        with tf.GradientTape(persistent=True) as tape:

            #### Virtually, ON THE CLIENT SIDE:
            # clients' smashed data
            z_private_raw = self.f(x_private, training=True)
            z_private = self.defend_features(z_private_raw, training=True)
            ####################################


            #### SERVER-SIDE:
            # map to data space (for evaluation and style loss)
            rec_x_private = self.decoder(z_private, training=True)
            ## adversarial loss (f's output must similar be to \tilde{f}'s output):
            adv_private_logits = self.D(z_private, training=True)
            if self.hparams['WGAN']:
                print("Use WGAN loss")
                f_loss = tf.reduce_mean(adv_private_logits)
            else:
                f_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(adv_private_logits), adv_private_logits, from_logits=True))
            ##
            f_loss += self._wcc_regularizer(z_private, label_private)

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


        # train client's network 
        var = self._client_trainable_variables()
        self._apply_gradients(tape, f_loss, var, self.optimizer0)

        # train attacker's autoencoder on public data
        var = self._attacker_autoencoder_variables()
        self._apply_gradients(tape, tilde_f_loss, var, self.optimizer1)

        # train discriminator
        var = self.D.trainable_variables
        self._apply_gradients(tape, D_loss, var, self.optimizer2)


        return f_loss, tilde_f_loss, D_loss, loss_c_verification


    def gradient_penalty(self, x, x_gen):
        epsilon_shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
        epsilon = tf.random.uniform(epsilon_shape, 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.D(x_hat, training=True)
        gradients = t.gradient(d_hat, x_hat)
        axes = list(range(1, len(gradients.shape)))
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=axes) + 1e-12)
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer
    
    
    @tf.function
    def score(self, x_private, label_private):
        z_private_raw = self.f(x_private, training=False)
        z_private = self.defend_features(z_private_raw, training=False)
        tilde_x_private = self.decoder(z_private, training=False)
        
        err = tf.reduce_mean( distance_data(x_private, tilde_x_private) )
        
        return err
    
    def scoreAttack(self, dataset):
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        scorelog = 0
        i = 0
        for x_private, label_private in tqdm.tqdm(dataset):
            scorelog += self.score(x_private, label_private).numpy()
            i += 1
             
        return scorelog / i

    def attack(self, x_private):
        # smashed data sent from the client:
        z_private_raw = self.f(x_private, training=False)
        z_private = self.defend_features(z_private_raw, training=False)
        # recover private data from smashed data
        tilde_x_private = self.decoder(z_private, training=False)

        z_private_control_raw = self.tilde_f(x_private, training=False)
        z_private_control = self.defend_features(z_private_control_raw, training=False)
        control = self.decoder(z_private_control, training=False)
        return tilde_x_private.numpy(), control.numpy()


    def __call__(self, iterations, log_frequency=500, verbose=False, progress_bar=True):

        n = int(iterations / log_frequency)
        LOG = np.zeros((n, 4))

        iterator = zip(self.client_dataset.take(iterations), self.attacker_dataset.take(iterations))
        if progress_bar:
            iterator = tqdm.tqdm(iterator , total=iterations)

        i, j = 0, 0
        print("RUNNING...")
        for (x_private, label_private), (x_public, label_public) in iterator:
            log = self.train_step(x_private, x_public, label_private, label_public)

            if i == 0:
                VAL = log[3]
            else:
                VAL += log[3] / log_frequency

            if  i % log_frequency == 0:
                LOG[j] = log

                if verbose:
                    print("log--%02d%%-%07d] validation: %0.4f" % ( int(i/iterations*100) ,i, VAL) )

                VAL = 0
                j += 1


            i += 1
        return LOG

#----------------------------------------------------------------------------------------------------------------------


class FSHA_binary_property(FSHA):
    
    def loadBiasNetwork(self, make_decoder, z_shape, channels, input_shape=None):
        del channels, input_shape
        class_num = self.hparams.get("class_num", 1)
        return make_decoder(z_shape, class_num)
    
    def binary_accuracy(self, label, logits):
    
        if self.hparams.get('class_num', 1) == 1:
            p = tf.nn.sigmoid(logits)
            predicted = tf.cast( (p > .5), tf.float32)
        else:
            p = tf.nn.softmax(logits)
            predicted = tf.argmax(p, 1)

        correct_prediction = tf.equal(label, predicted)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    def classification_loss(self, label, logits):
        if self.hparams.get('class_num', 1) == 1:
            return tf.reduce_mean(tf.keras.losses.binary_crossentropy(label, logits, from_logits=True))
        else:
            return tf.reduce_mean( tf.keras.losses.sparse_categorical_crossentropy(label, logits, from_logits=True) )
        

    @tf.function
    def train_step(self, x_private, x_public, label_private, label_public):

        with tf.GradientTape(persistent=True) as tape:

            #### Virtually, ON THE CLIENT SIDE:
            # clients' smashed data
            z_private_raw = self.f(x_private, training=True)
            z_private = self.defend_features(z_private_raw, training=True)
            ####################################

            #### SERVER-SIDE:
            # map to data space (for evaluation and style loss)
            clss_private_logits = self.decoder(z_private, training=True)
            ## adversarial loss (f's output must be similar to \tilde{f}'s output):
            adv_private_logits = self.D(z_private, training=True)
            if self.hparams['WGAN']:
                print("Use WGAN loss")
                f_loss = tf.reduce_mean(adv_private_logits)
            else:
                f_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(adv_private_logits), adv_private_logits, from_logits=True))
            ##
            f_loss += self._wcc_regularizer(z_private, label_private)

            # attacker's classifier
            z_public_raw = self.tilde_f(x_public, training=True)
            z_public = self.defend_features(z_public_raw, training=True)
            clss_public_logits = self.decoder(z_public, training=True)

            # classificatio loss
            public_classification_loss = self.classification_loss(label_public, clss_public_logits)
            public_classification_accuracy = self.binary_accuracy(label_public, clss_public_logits)
            tilde_f_loss = public_classification_loss

            # discriminator on attacker's feature-space
            adv_public_logits = self.D(z_public, training=True)
            if self.hparams['WGAN']:
                loss_discr_true = tf.reduce_mean(adv_public_logits)
                loss_discr_fake = -tf.reduce_mean(adv_private_logits)
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
            private_classification_accuracy = self.binary_accuracy(label_private, clss_private_logits)
            ############################################
            ##################################################################


        var = self._client_trainable_variables()
        self._apply_gradients(tape, f_loss, var, self.optimizer0)

        var = self._attacker_autoencoder_variables()
        self._apply_gradients(tape, tilde_f_loss, var, self.optimizer1)

        var = self.D.trainable_variables
        self._apply_gradients(tape, D_loss, var, self.optimizer2)


        return f_loss, tilde_f_loss, D_loss, private_classification_accuracy, public_classification_accuracy
