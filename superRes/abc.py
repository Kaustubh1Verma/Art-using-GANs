from __future__ import print_function, division

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#import tensorflow 
#tensorflow.logging.set_verbosity(tensorflow.logging.WARN)
from tensorflow.python.keras import backend as k
import scipy
from scipy.misc import imsave
from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.python.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.applications import VGG19
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf
import datetime
import cv2
import sys
from dataloader_new import DataLoader
import numpy as np
import os
from tiramisu import Tiramisu
from tensorflow.python.keras.applications.vgg19 import preprocess_input

def feature_extract(x):
    print(x)
    fc2_features = model_extractfeatures.predict(x,steps=1)
    return fc2_features


def preprocess(img):
    print(img)
    #print(k.get_value(img))
    resized_images = tf.image.resize_images(img, (224, 224))
    print(resized_images)
    #k.resize_images(img,height_factor=224,width_factor=224,data_format="channels_last")
    #print(img.shape)
    #x = image.img_to_array(img[0])
    #x = np.expand_dims(x, axis=0)
    #print(img)
    x = preprocess_input(resized_images)
    return x

HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
   x   = k.abs(y_true - y_pred)
   x   = k.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  k.sum(x)

def total_loss(y_true, y_pred):
    f1=preprocess(y_true)
    f2=preprocess(y_pred)
    fx1=feature_extract(f1)
    fx2=feature_extract(f2)
    loss1 = tf.reduce_mean(tf.squared_difference(fx1, fx2))
    loss2=smoothL1(y_true,y_pred)
    return k.eval(loss1)+k.eval(loss2)

###################################################################3




class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'facades'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        #self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])
 
       
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', smoothL1],
                              loss_weights=[2, 100],
                              optimizer=optimizer)
        
        ################# Perceptual loss and L1 loss ######################
        self.vggmodel=VGG19(weights="vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False)
        #print(vggmodel.get_layer('block4_pool'))
        #print(self.combined.output[1])
        #print(vggmodel.get_layer('block4_pool').output)
        #lossOut = vggmodel(inputs=self.combined.output[1], output = vggmodel.get_layer('block4_pool').output)
        lossOut = self.vggmodel(inputs=self.combined.output[1])

        self.vggmodel.trainable = False
        for l in self.vggmodel.layers:
            l.trainable = False

        self.vgg_combined = Model(inputs=self.combined.input, outputs=lossOut)
        self.vgg_combined.compile(loss='mse',optimizer='adam',loss_weights=[10])
         
        valid.trainable = False

    def build_generator(self):

        layer_per_block = [4, 4, 4, 4, 4, 15, 4, 4, 4, 4, 4]


        tiramisu = Tiramisu(layer_per_block)
        tiramisu.summary()


        #d0 = Input(shape=self.img_shape)

        return tiramisu

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
               
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss=self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                
                full_vgg = self.vggmodel.predict(imgs_A)
                 
                vgg_loss = self.vgg_combined.train_on_batch([imgs_A, imgs_B], full_vgg)

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)


            self.combined.save_weights("Weights/"+str(epoch)+".h5")  







    def img_to_frame(self,imgA,imgB,fakeA):
        no_images = imgA.shape[0]
        img_height = imgA.shape[1]
        img_width = imgA.shape[2]
        pad = 20
        title_pad=20
        pad_top = pad+title_pad
        frame=np.zeros((no_images*(img_height+pad_top),no_images*(img_width+pad),3))
        count=0
        gen_imgs = np.concatenate([imgB, fakeA, imgA])
        gen_imgs = 0.5 * gen_imgs + 0.5
        titles = ['Input', 'Generated', 'Original']
        for r in range(no_images):
            for c in range(no_images):
                im = gen_imgs[count]
                count=count+1
                y0 = r*(img_height+pad_top) + pad//2
                x0 = c*(img_width+pad) + pad//2
                # print(frame[y0:y0+img_height,x0:x0+img_width,:].shape)
                frame[y0:y0+img_height,x0:x0+img_width,:] = im*255
                frame = cv2.putText(frame, titles[r], (x0, y0-title_pad//4), cv2.FONT_HERSHEY_COMPLEX, .5, (255,255,255))
        return frame




    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        os.makedirs('images/dehazed', exist_ok=True)
        os.makedirs('images/haze', exist_ok=True)
        os.makedirs('images/original',exist_ok=True)
        r, c = 3, 3
         
     
        imgs_A, imgs_B, or_A, or_B = self.data_loader.load_data(batch_size=3, is_testing=True)
   
        fake_A = self.generator.predict(imgs_B)

        cv2.imwrite("images/dehazed"+"/"+"Img:"+str(epoch)+"_"+str(batch_i)+".jpg",(fake_A[0]*0.5+0.5)*255) 
        cv2.imwrite("images/haze"+"/"+"Img:"+str(epoch)+"_"+str(batch_i)+".jpg",(or_B[0]*0.5+0.5)*255)
        cv2.imwrite("images/original"+"/"+"Img:"+str(epoch)+"_"+str(batch_i)+".jpg",(or_A[0]*0.5+0.5)*255)
        
        frame=self.img_to_frame(imgs_A,imgs_B,fake_A)
	
        cv2.imwrite("images/"+self.dataset_name+"/"+"Img:"+str(epoch)+"_"+str(batch_i)+".png",frame)
        #imsave("images/"+self.dataset_name+"/"+"Scipy:Img:"+str(epoch)+"_"+str(batch_i)+".png",frame )



if __name__ == '__main__':
    gan = Pix2Pix()
gan.train(epochs=200, batch_size=1, sample_interval=200)
