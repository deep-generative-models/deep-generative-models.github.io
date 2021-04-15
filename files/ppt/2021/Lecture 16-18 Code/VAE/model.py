import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Dense, Lambda, Dropout, Reshape, Conv2d, DeConv2d, Flatten


class VAE(tl.models.Model):
    def __init__(self, batch_size, origin_units, hidden_units, latent_units):
        super(VAE, self).__init__()
        self.latent_units = latent_units
        self.encoder = self.get_encoder(batch_size, origin_units, hidden_units, latent_units).as_layer()
        self.decoder = self.get_decoder(batch_size, origin_units, hidden_units, latent_units).as_layer()

    def get_encoder(self, batch_size, origin_units, hidden_units, latent_units):
        init = tf.initializers.he_uniform()
        ni = Input((batch_size, origin_units))
        nn = Dense(hidden_units, act=tf.nn.relu, W_init=init, b_init=init)(ni)  # replace tanh with currently more used relu

        mean = Dense(latent_units, W_init=init, b_init=init)(nn)
        log_sigma = Dense(latent_units, W_init=init, b_init=init)(nn)

        def sample(data):
            mean, log_sigma = data
            stddev = 0.5 * tf.exp(log_sigma)  # Here may be a little different with the paper, but is ok
            out = mean + stddev * tf.random.normal(mean.shape)
            return out

        z = Lambda(sample)([mean, log_sigma])
        return tl.models.Model(inputs=ni, outputs=[z, mean, log_sigma])

    def get_decoder(self, batch_size, origin_units, hidden_units, latent_units):
        init = tf.initializers.he_uniform()
        ni = Input((batch_size, latent_units))
        nn = Dense(hidden_units, act=tf.nn.relu, W_init=init, b_init=init)(ni)  # replace tanh with currently more used relu
        no = Dense(origin_units, act=tf.nn.tanh, W_init=init, b_init=init)(nn)
        return tl.models.Model(inputs=ni, outputs=no)

    def forward(self, batch_imgs):
        z, mean, log_sigma = self.encoder(batch_imgs)
        imgs = self.decoder(z)
        return imgs, mean, log_sigma

    def generate(self, z_in):
        assert z_in.shape[-1] == self.latent_units

        return self.decoder(z_in)


def get_vae(batch_size, origin_units, hidden_units, latent_units):
    k = 700 # a weight used for the reconstruction loss
    return k, VAE(batch_size, origin_units, hidden_units, latent_units)


class DC_VAE(tl.models.Model):
    """
    Can also build with Conv and Deconv rather than only with Dense layers
    """
    def __init__(self, batch_size, origin_units, hidden_units, latent_units):
        super(DC_VAE, self).__init__()
        self.latent_units = latent_units
        self.encoder = self.get_encoder(batch_size, origin_units, hidden_units, latent_units).as_layer()
        self.decoder = self.get_decoder(batch_size, origin_units, hidden_units, latent_units).as_layer()

    def get_encoder(self, batch_size, origin_units, hidden_units, latent_units):
        lrelu = lambda x: tf.nn.leaky_relu(x, 0.3)
        ni = Input((batch_size, origin_units))
        nn = Reshape((batch_size, 28, 28, 1))(ni)
        nn = Conv2d(hidden_units, (4, 4), (2, 2), act=lrelu)(nn)
        nn = Dropout(keep=0.8)(nn)
        nn = Conv2d(hidden_units, (4, 4), (2, 2), act=lrelu)(nn)
        nn = Dropout(keep=0.8)(nn)
        nn = Conv2d(hidden_units, (4, 4), (1, 1), act=lrelu)(nn)
        nn = Dropout(keep=0.8)(nn)
        nn = Flatten()(nn)
        mean = Dense(latent_units)(nn)
        log_sigma = Dense(latent_units)(nn)

        def sample(data):
            mean, log_sigma = data
            epsilon = tf.random.normal(mean.shape)
            stddev = tf.exp(0.5 * log_sigma)
            out = mean + stddev * epsilon
            return out

        z = Lambda(sample)([mean, log_sigma])
        return tl.models.Model(inputs=ni, outputs=[z, mean, log_sigma])

    def get_decoder(self, batch_size, origin_units, hidden_units, latent_units):
        lrelu = lambda x: tf.nn.leaky_relu(x, 0.3)
        ni = Input((batch_size, latent_units))
        nn = Dense(n_units=24, act=lrelu)(ni)
        nn = Dense(n_units=49, act=lrelu)(nn)
        nn = Reshape((batch_size, 7, 7, 1))(nn)
        nn = DeConv2d(hidden_units, (4, 4), (2, 2), act=tf.nn.relu)(nn)
        nn = Dropout(keep=0.8)(nn)
        nn = DeConv2d(hidden_units, (4, 4), (1, 1), act=tf.nn.relu)(nn)
        nn = Dropout(keep=0.8)(nn)
        nn = DeConv2d(hidden_units, (4, 4), (1, 1), act=tf.nn.relu)(nn)
        nn = Flatten()(nn)
        nn = Dense(784, act=tf.nn.tanh)(nn)
        return tl.models.Model(inputs=ni, outputs=nn)

    def forward(self, batch_imgs):
        z, mean, log_sigma = self.encoder(batch_imgs)
        imgs = self.decoder(z)
        return imgs, mean, log_sigma

    def generate(self, z_in):
        assert z_in.shape[-1] == self.latent_units

        return self.decoder(z_in)


def get_dcvae(batch_size, origin_units, hidden_units, latent_units):
    k = 2e4 # the same as the k in get_vae()
    # However, the value k may be too small, according to the experiments with vae above, which makes the
    # reconstruction loss converge before the latent loss converge to 0. So not suggested to use
    return k, DC_VAE(batch_size, origin_units, hidden_units, latent_units)


def KL_loss(mu, log_sigma):
    loss = -0.5 * tf.reduce_sum(1 + log_sigma - tf.exp(log_sigma) - mu**2)
    loss = tf.reduce_sum(loss)
    return loss


def recon_loss(x, y, k):
    # loss = k * tf.losses.binary_crossentropy(x, y) # k given by the methods above
    # loss = tf.reduce_sum(loss)
    loss = tl.cost.mean_squared_error(x, y)
    return loss


if __name__ == '__main__':
    vae = get_vae(64, 784, 400, 20)
    print(vae.trainable_weights)
