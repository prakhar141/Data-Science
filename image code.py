from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Define input image dimensions
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

# Given input of noise (latent) vector, the Generator produces an image.
def build_generator():

    noise_shape = (100,)  # 1D array of size 100 (latent vector / noise)

    model = Sequential()
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)  # Generated image

    return Model(noise, img)

# Given an input image, the Discriminator outputs the likelihood of the image being real.
def build_discriminator():

    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

# Now that we have constructed our two models, it’s time to pit them against each other.
# We do this by defining a training function, loading the data set, re-scaling our training images, and setting the ground truths.
def train(epochs, batch_size=128, save_interval=50):

    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    # Add channels dimension
    X_train = np.expand_dims(X_train, axis=3)

    half_batch = int(batch_size / 2)

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of real images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (half_batch, 100))

        # Generate a half batch of fake images
        gen_imgs = generator.predict(noise)

        # Train the discriminator on real and fake images, separately
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

        # Average loss from real and fake images
        d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])  # Only use the loss value (d_loss_real[0] and d_loss_fake[0])
        d_acc = 0.5 * (d_loss_real[1] + d_loss_fake[1])  # Use the accuracy value (d_loss_real[1] and d_loss_fake[1])

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, 100))  # Create noise vectors as input for generator
        valid_y = np.array([1] * batch_size)  # Array of ones (valid labels)

        # Train the generator with noise as input
        g_loss = combined.train_on_batch(noise, valid_y)

        # Print the progress
        #print(f"{epoch} [D loss: {d_loss:.6f}, acc.: {100*d_acc:.2f}%] [G loss: {g_loss:.6f}]")
        # Print the progress
        #print(f"{epoch} [D loss: {d_loss:.6f}, acc.: {100 * d_acc:.2f}%] [G loss: {g_loss:.6f}]")


        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_imgs(epoch)


# When the specific sample_interval is hit, we call the sample_image function. Which looks as follows.
def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"/content/drive/MyDrive/images/mnist_{epoch}.png")
    plt.close()

# Optimizer for easy use later on
optimizer = Adam(0.0002, 0.5)  # Learning rate and momentum.

# Build and compile the discriminator first
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

# Build and compile the generator
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

# This builds the Generator and defines the input noise.
z = Input(shape=(100,))  # Our random input to the generator
img = generator(z)

# This ensures that when we combine our networks, we only train the Generator.
discriminator.trainable = False

# This specifies that our Discriminator will take the images generated by our Generator
# and true dataset and set its output to a parameter called valid, which will indicate
# whether the input is real or not.
valid = discriminator(img)  # Validity check on the generated image

# Here we combined the models and also set our loss function and optimizer.
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# Start training
train(epochs=100, batch_size=32, save_interval=10)

# Save the generator model for future use
generator.save('/content/drive/MyDrive/images/generator_model.h5')
