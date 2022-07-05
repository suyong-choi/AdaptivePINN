import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import numpy as np
from diffutils import *
from gtree import *
import pickle
from diffutils import *


def weightfn(x):
    return 1.0+tf.math.log1p(1000.0*x)

def weightfn2(x):
    #return tf.math.pow(5.0*(x-1.0), 2) + 0.1*tf.math.log1p(5.0+(x-1.0))
    return tf.math.pow(x, 2)

class APINN:
    def __init__(self):
        self.interiorpointslist = []
        self.BCpointslist = []
        self.BCphilist = []
        self.learning_rate = 0.01
        pass

    def initialize(self):
        self.BCpoints = np.vstack(self.BCpointslist)
        self.BCphi = np.vstack(self.BCphilist)
        self.interiorpoints = np.vstack(self.interiorpointslist)
        self.optimizer = tf.optimizers.Adam(learning_rate = self.learning_rate)
        self.lamoptimizer = tf.optimizers.Adam(learning_rate = 0.1*self.learning_rate)
        self.recalc()
        pass

    def recalc(self):
        self.BCpoints_size = self.BCpoints.shape[0]
        self.BCpoints_x = np.expand_dims(self.BCpoints[:, 0], axis=1)
        self.BCpoints_y = np.expand_dims(self.BCpoints[:, 1], axis=1)
        self.BClam = np.ones((self.BCpoints_size, 1), dtype=np.float32) # adaptive weights
        self.BCindices = np.arange(self.BCpoints_size, dtype=np.int32)

        self.interiorpoints_size = self.interiorpoints.shape[0]
        self.interior_x = np.expand_dims(self.interiorpoints[:, 0], axis=1)
        self.interior_y = np.expand_dims(self.interiorpoints[:, 1], axis=1)
        self.interiorlam = 0.1*np.ones((self.interiorpoints_size, 1), dtype=np.float32) # adaptive weights
        self.interiorindices = np.arange(self.interiorpoints_size, dtype=np.int32)
        pass



    def add_Dirichlet_BC(self, points, potentials):
        self.BCpointslist.append(points)
        self.BCphilist.append(potentials)
        pass

    def add_interior_points(self, points):
        self.interiorpointslist.append(points)
        pass

    def buildNN(self):

        # potential function
        tin = tfk.layers.Input((2,))
        h1 = tfk.layers.Dense(256, activation='swish')(tin)
        #h1 = tfk.layers.BatchNormalization()(h1)
        h1 = tfk.layers.Dense(512, activation='swish')(h1)
        x_out = tfk.layers.Dense(1, activation=None)(h1)
        self.phi_xy = tfk.Model(tin, x_out)

        # weighting function for BC loss, inspired from arXiv:2009.04544
        tin2 = tfk.layers.Input((2,))
        h21 = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(tin2)
        h21 = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(h21)
        out2 = 1.0+10.0*tfk.layers.Dense(1, activation='sigmoid')(0.01*h21)
        self.lamBC = tfk.Model(tin2, out2)

        # weighting function for internal points loss
        tin3 = tfk.layers.Input((2,))
        h31 = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(tin3)
        h31 = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(h31)
        out3 = 1.0+10.0*tfk.layers.Dense(1, activation='sigmoid')(0.01*h31)
        self.lamint = tfk.Model(tin3, out3)

        pass

    def updatelam(self, BCindices, interiorindices, gradlamBC, gradlamint):
        self.BClam[BCindices] +=  self.learning_rate * gradlamBC
        self.interiorlam[interiorindices] += self.learning_rate * gradlamint
        pass

    @tf.function 
    def train_step_BC(self, batch_size=-1, updatelambda = False):
        if batch_size>0:
            self.BCindices = np.random.permutation(self.BCpoints_size)[0:batch_size]
            BCpoints = self.BCpoints[self.BCindices]
            BCphi = self.BCphi[self.BCindices]
        else:
            self.BCindices = np.arange(self.BCpoints_size, dtype=np.int32)
            BCpoints = self.BCpoints
            BCphi = self.BCphi

        with tf.GradientTape(persistent=True) as tape:
            L_bc = tf.abs(self.phi_xy(BCpoints) - BCphi) 
            L = tf.reduce_mean(L_bc) 
        
        grad = tape.gradient(L, self.phi_xy.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.phi_xy.trainable_variables))
        gradlamBC = 0.0
        gradlamint = 0.0

        return L, gradlamBC, gradlamint

    @tf.function
    def train_step(self, batch_size=-1, updatelambda = False):
        if batch_size>0:
            self.BCindices = np.random.permutation(self.BCpoints_size)[0:batch_size]
            BCpoints = self.BCpoints[self.BCindices]
            BCphi = self.BCphi[self.BCindices]
            BClam = self.BClam[self.BCindices]

            self.interiorindices = np.random.permutation(self.interiorpoints_size)[0:batch_size]
            interior_x =self.interior_x[self.interiorindices]
            interior_y =self.interior_y[self.interiorindices]
            interiorlam= self.interiorlam[self.interiorindices]
        else:
            BCpoints = self.BCpoints
            BCphi = self.BCphi
            BClam = self.BClam

            interior_x =self.interior_x
            interior_y =self.interior_y
            interiorlam = self.interiorlam

        # convert numpy to tf tensor in order to use differentiate
        BClam_tftensor = tf.convert_to_tensor(BClam)
        interiorlam_tftensor = tf.convert_to_tensor(interiorlam)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(BClam_tftensor)
            tape.watch(interiorlam_tftensor)
            interiorweight = weightfn2(interiorlam_tftensor)
            BCweight = weightfn2(BClam_tftensor)

            #L_phi =  tf.reduce_sum(interiorweight * tf.square(Laplacian(self.phi_xy, interior_x, interior_y))) / tf.reduce_sum(interiorweight)
            #L_bc =  tf.reduce_sum(BCweight * tf.square(self.phi_xy(BCpoints) - BCphi) ) / tf.reduce_sum(BCweight)
            L_phi =  tf.reduce_mean(interiorweight * tf.square(Laplacian(self.phi_xy, interior_x, interior_y))) 
            L_bc =  tf.reduce_mean(BCweight * tf.square(self.phi_xy(BCpoints) - BCphi) ) 
            L = L_bc +L_phi
            negL = -L
        
        grad = tape.gradient(L, self.phi_xy.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.phi_xy.trainable_variables))
        if updatelambda:
            gradlamBC = tape.gradient(L, BClam_tftensor)
            gradlamint = tape.gradient(L, interiorlam_tftensor)
        else:
            gradlamBC = 0.0
            gradlamint = 0.0

        return L, gradlamBC, gradlamint

    # works Ok 200k steps
    @tf.function
    def train_step_nnweights(self, batch_size=-1, updatelambda = False):
        if batch_size>0:
            self.BCindices = np.random.permutation(self.BCpoints_size)[0:batch_size]
            BCpoints = self.BCpoints[self.BCindices]
            BCphi = self.BCphi[self.BCindices]

            self.interiorindices = np.random.permutation(self.interiorpoints_size)[0:batch_size]
            interior_x =self.interior_x[self.interiorindices]
            interior_y =self.interior_y[self.interiorindices]
            interiorpoints = self.interiorpoints[self.interiorindices]
        else:
            self.BCindices = np.arange(self.BCpoints_size, dtype=np.int32)
            BCpoints = self.BCpoints
            BCphi = self.BCphi

            self.interiorindices = np.arange(self.interiorpoints_size, dtype=np.int32)
            interior_x =self.interior_x
            interior_y =self.interior_y
            interiorpoints = self.interiorpoints

        with tf.GradientTape(persistent=True) as tape:
            lamint = self.lamint(interiorpoints)
            L_phi = tf.reduce_sum(lamint * tf.square(Laplacian(self.phi_xy, interior_x, interior_y))) / tf.reduce_sum(lamint)
            lamBC = self.lamBC(BCpoints)
            L_bc = tf.reduce_sum(lamBC * tf.square(self.phi_xy(BCpoints) - BCphi)) / tf.reduce_sum(lamBC)
            L = L_bc + L_phi
            negL = -L
        
        grad = tape.gradient(L, self.phi_xy.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.phi_xy.trainable_variables))
        if updatelambda:
            gradlamBC = tape.gradient(negL, self.lamBC.trainable_variables)
            gradlamint = tape.gradient(negL, self.lamint.trainable_variables)
            self.lamoptimizer.apply_gradients(zip(gradlamint, self.lamint.trainable_variables))
            self.lamoptimizer.apply_gradients(zip(gradlamBC, self.lamBC.trainable_variables))
        else:
            gradlamBC = 0.0
            gradlamint = 0.0

        return L, gradlamBC, gradlamint

    def train(self, nsteps, batch_size, trainforBC=False):
        self.train_hist=[]
        trainstep = self.train_step
        #trainstep = self.train_step_nnweights
        
        if trainforBC:
            trainstep = self.train_step_BC

        for istep in range(nsteps):
            loss, gradlamBC, gradlamint =trainstep(batch_size, True)
            self.updatelam(self.BCindices, self.interiorindices, gradlamBC, gradlamint)
            if istep % 10 ==0:
                print(f'{istep}: {loss.numpy()}')
                self.train_hist.append(loss.numpy())

    def save(self, savedir="poisson_pinn/"):
        tf.keras.models.save_model(self.phi_xy, savedir+'phi_xy')
        tf.keras.models.save_model(self.lamBC, savedir+'lamBC')
        tf.keras.models.save_model(self.lamint, savedir+'lamint')
        file1 = open(savedir+'BC.pkl', 'wb')
        pickle.dump((self.BCpoints, self.BCphi, self.BClam), file1)
        file1.close()
        file2 = open(savedir+'interior.pkl', 'wb')
        pickle.dump((self.interiorpoints, self.interiorlam), file2)
        file2.close
        pass

    def load(self, savedir="poisson_pinn/"):
        self.phi_xy = tf.keras.models.load_model(savedir+'phi_xy')
        self.lamBC = tf.keras.models.load_model(savedir+'lamBC')
        self.lamint = tf.keras.models.load_model(savedir+'lamint')
        file1 = open(savedir+'BC.pkl', 'rb')
        self.BCpoints, self.BCphi, self.BClam = pickle.load(file1)
        file2 = open(savedir+'interior.pkl', 'rb')
        self.interiorpoints, self.interiorlam = pickle.load(file2)
        self.recalc()
        pass

