import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from glob import glob
from data import get_mnist, flags
from model import get_vae, get_dcvae, KL_loss, recon_loss

num_tiles = int(np.sqrt(flags.sample_size))


def train():
    images, len_instance = get_mnist(flags.batch_size)
    k, vae = get_vae(flags.batch_size, flags.original_dim, flags.hidden_dim, flags.z_dim)

    vae.train()

    vae_optimizer = tf.optimizers.Adam(flags.lr)

    n_step_epoch = int(len_instance // flags.batch_size)

    for epoch in range(flags.n_epoch):
        for step, batch_images in enumerate(images):
            if batch_images.shape[0] != flags.batch_size:  # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            last_batch = batch_images

            with tf.GradientTape(persistent=True) as tape:
                reconstr_img, n_mean, n_log_sigma = vae(batch_images)

                reconstr_loss = recon_loss(batch_images, reconstr_img, k)
                latent_loss = KL_loss(n_mean, n_log_sigma)

                loss = tf.add(reconstr_loss, latent_loss)

            grad = tape.gradient(loss, vae.trainable_weights)
            vae_optimizer.apply_gradients(zip(grad, vae.trainable_weights))
            del tape

            print("Epoch: [{}/{}] [{}/{}] took: {:.3f}, reconstr_loss: {:.5f}, latent_loss: {:.5f}".format(epoch, \
                                                                                               flags.n_epoch, step,
                                                                                               n_step_epoch,
                                                                                               time.time() - step_time,
                                                                                               reconstr_loss, latent_loss))

        if np.mod(epoch, flags.save_every_epoch) == 0:
            vae.save_weights('{}/VAE.npz'.format(flags.checkpoint_dir), format='npz')
            vae.eval()
            tmp_result, tmp_mean, tmp_logvar = vae(last_batch) # get the reconstruction results
            z = np.random.normal(0.0, 1.0, (flags.sample_size, flags.z_dim)).astype(np.float32)
            gen_result = vae.generate(z) # get a randomly generated results
            vae.train()
            tl.visualize.save_images(tmp_result.numpy().reshape([-1, 28, 28, 1]), [num_tiles, num_tiles],
                                     '{}/train_{:02d}.png'.format(flags.sample_dir, epoch))
            tl.visualize.save_images(gen_result.numpy().reshape([-1, 28, 28, 1]), [num_tiles, num_tiles],
                                     '{}/generate_{:02d}.png'.format(flags.sample_dir, epoch))


if __name__ == '__main__':
    train()
