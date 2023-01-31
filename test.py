# import tensorflow as tf
# import tensorflow.keras as keras
from utils import *
import numpy as np


class DV():

    def __init__(self, model, task='cifar10'):

        self.model = model
        self.matrix = np.zeros((10, 40, 32, 32, 3))

    

        # print(np.argmax(self.model.predict(self.matrix[3]),1))

    def decision(self, x):
        x = np.clip(x, 0, 1)
        if len(x.shape) == 3:

            x = np.expand_dims(x, axis=0)

        else:

            x = x

        return np.argmax(self.model.predict(x), 1)

    # def scale_predict(self, scale, x, x_sub,l):
    #
    #     t = 1/scale
    #
    #     for i in range(scale):

    def dv_predict(self, x):

        label = self.decision(x)

        for l in range(10):

            print(f"LABEL:{l}\n")

            if label == l:

                continue

            else:

                x_sub = self.matrix[l]

                out_images, highs = self.binary_search_batch(x_sub, np.repeat(x, repeats=40, axis=0),
                                                             label, label_o=l)

                # print(highs)
                hit_t = np.array(
                    [self.approximate_gradient(np.expand_dims(out_images[i], axis=0), 1000, 0.01, label, label_o=l)[0]
                     for i in
                     range(len(out_images))])

                hit_o = np.array(
                    [self.approximate_gradient(np.expand_dims(out_images[i], axis=0), 1000, 0.01, label, label_o=l)[1]
                     for i in
                     range(len(out_images))])

                print(hit_t)
                print(hit_o)
                print(1 - (hit_t + hit_o))
                print("###############\n")

    def predict_v2(self, x, scale=500):

        label = self.decision(x)

        print(label)

        decisions = np.ones((40, scale))

        x_original = self.matrix[label][0]

        for i in range(40):
            decisions[i] = np.array(self.grid_search(np.expand_dims(x_original[i], axis=0),
                                                     x,
                                                     scale=scale))

        print(np.mean(decisions == label, axis=1))

    def grid_search(self, original_images, perturbated_images, scale):

        step = 1 / scale

        evals = np.empty((scale, original_images.shape[1], original_images.shape[2], 3))

        for i in range(scale):
            evals[i] = (original_images * (1 - step * scale) + perturbated_images * step * scale)[0]

        return self.decision(evals)

    def binary_search_batch(self, original_image, perturbed_images, label_t, label_o):
        """ Binary search to approach the boundar. """
        thresholds = 1 / (224 * 224 * 3 * 10)
        lows = np.zeros(len(perturbed_images))
        highs = np.ones(len(perturbed_images))

        # Call recursive function.
        while np.max((highs - lows) / thresholds) > 1:
            # projection to mids.

            mids = (highs + lows) / 2.0

            mid_images = np.array([(1 - mids[i]) * original_image[i] + mids[i] * perturbed_images[i] for i in
                                   range(len(perturbed_images))])
            # Update highs and lows based on model decisions.
            decisions = self.decision(mid_images)

            # You should be careful here.
            lows = np.where(decisions != label_t, mids, lows)

            highs = np.where(decisions == label_t, mids, highs)

        out_images = np.array([(1 - highs[i]) * original_image[i] +
                               highs[i] * perturbed_images[i] for i in range(len(perturbed_images))])
        print(highs)
        return out_images, highs

    def approximate_gradient(self, sample, num_evals, delta, label_t, label_o):
        np.random.seed(1995)
        # Generate random vectors.
        sample = np.repeat(sample, num_evals, axis=0)
        noise_shape = [num_evals, 32, 32, 3]

        rv = np.random.random(size=noise_shape)

        rv = rv / np.sqrt(np.sum(rv ** 2, axis=(1, 2, 3), keepdims=True))

        perturbed = np.clip(sample + delta * rv, 0, 1)

        # rv = (perturbed - sample) / delta

          # query the model.
        decisions = self.decision(perturbed)

        return [np.mean(decisions == label_t), np.mean(decisions == label_o)]

    def extract(self, x, label, size=10):

        noise = np.random.uniform(low=-1, high=1, size=(size, x.shape[1], x.shape[2], 3))

        evals = np.clip(np.repeat(x, size, axis=0) + 0.05 * noise, 0, 1)

        decisions = self.decision(evals)

        return noise[decisions != label]

    def out_of_predict(self, x, ground_truth, file
                       ):

        evals = np.empty((len(x), 11, x.shape[-2], x.shape[-2], 3))
        decisions = np.empty((len(x), 11))
        source_label = self.decision(x)

        for i in range(1, 12):
            evals[:, i - 1] = np.clip(i*x, 0, 1)
            decisions[:, i - 1] = self.decision(evals[:, i - 1])

        a = decisions[decisions[:, 0] == np.reshape(ground_truth,(len(ground_truth)))]
        #a = decisions
        print(a)
        np.save(f"saved_np/{file}",
                a)

      
    def out_of_predict_v2(self, x):

        soure_label = self.decision(x)
    
        noise = -self.extract(x, soure_label)

        if len(noise) == 0:
            return self.out_of_predict(x)

        evals = np.empty((4 * len(noise), 32, 32, 3))

        for h in range(0, len(noise)):

            for i in range(2, 6):
                evals[h * 4 + (i - 2)] = np.clip((i - 1) * noise[h] + x[0], 0, 1)

        # print(np.mean(np.mean(np.reshape(self.decision(evals)==soure_label,(len(noise),4)),axis=1)<=0.2))
        print(np.reshape(self.decision(evals) == soure_label, (len(noise), 4)))


def process(file_data):
    a = np.load(file_data)
   
    score = np.empty(len(a))

    for i in range(len(a)):
        score[i] = np.mean(a[i] == a[i][0])

    return score

def process_few(file_data,file):

    a = np.load(file_data)
   
    mean = np.load(f"saved_np/{file}/mean.npy")
    var = np.load(f"saved_np/{file}/var.npy")
    mi = np.load(f"saved_np/{file}/min.npy")
    ma = np.load(f"saved_np/{file}/max.npy")
    print(mean) 
    #few = ((ma+mi)*0.5)/(10*np.sqrt(var)+0.000001)
    #mean[0] = 0.1
    few = mean/(10*np.sqrt(var)+0.000001)
    # print(few)
    # few[0] = 0.06
    # few = mean
    #few = mean       
    # print(var)

    score = np.empty(len(a))

    for i in range(len(a)):
        score[i] = np.abs(np.mean(a[i] == a[i][0]) -few[int(a[i][0])])
    return score

def few_shot(file_data,class_num=200,file='WaNet'):

    a = np.load(file_data)
    #a = a[:,:9]

    score_class = np.zeros(class_num)

    score_var = np.zeros(class_num)
    score_min = np.zeros(class_num)
    score_max = np.zeros(class_num)
    for i in range(class_num):


        dat = a[a[:,0]==i]
       # print(dat)
        var = []
        max_arr = []
        min_arr = []
        for h in range(len(dat)):

            score_class[i]+= np.mean(dat[h]==dat[h][0])
            var.append(np.mean(dat[h]==dat[h][0]))
        score_class[i] = np.mean(np.array(var))
        score_var[i] = np.var(np.array(var))
        score_max[i] = np.max(np.array(var))
        score_min[i] = np.min(np.array(var))
  
    np.save(f"saved_np/{file}/mean.npy",score_class)
    np.save(f"saved_np/{file}/var.npy",score_var)
    np.save(f"saved_np/{file}/max.npy",score_max)
    np.save(f"saved_np/{file}/min.npy",score_min)
    print(score_min)
    print(score_var)
    return score_class





if __name__ == '__main__':

	# run few_shot function to implement data-limited defense
    #few_shot("saved_np/WaNet/tiny_benign.npy",file='saved_np/WaNet/')
    
    AUROC_Score(process("saved_np/WaNet/tiny_bd.npy"),
               process("saved_np/WaNet/tiny_benign.npy"), "tiny_imagenet")
  
