from time import time
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from perlin_noise import PerlinNoise


class HazeGenerator:
    def __init__(self, mapping_option):
        self.mapping_option = mapping_option

    def draw_samples(self, scale, height, width):
        return np.random.normal(loc=0, scale=scale, size=(height, width))

    def generate_intensity_map_coarse(self, image, intensity_mean, scale, interpolation):
            height_intensity, width_intensity = (8, 8)
            intensity = intensity_mean + self.draw_samples(scale=scale, height=height_intensity, width=width_intensity)
            intensity = cv2.resize(intensity, (image.shape[1], image.shape[0]), interpolation=interpolation)
            return intensity
    def normalize(self, a):
        b = (a - np.min(a))/np.ptp(a)
        return b

    def gen_perlin_noise_map(self, img, octaves):
        w, h = img.shape[0], img.shape[1]
        list_noise = []
        for rank in octaves:
            list_noise.append(PerlinNoise(octaves=rank))
        pic = []
        for i in range(w):
            row = []
            for j in range(h):
                noise = 0
                for k in range (len(list_noise)):
                    noise += 1/(k+1) * list_noise[k]([i/w, j/h])
                row.append(noise)
            pic.append(row)
        return np.array(pic)

    def gen_intensity_map_fine(self, img, intensity_mean, octaves):
        intensity_details = self.gen_perlin_noise_map(img, octaves=octaves)
        return (intensity_mean * 2 * intensity_details - 0.5)/2.5

    def generate_alpha_mask(self, img, alpha_min, octaves, alpha_multiplier, sparsity, density_multiplier):
            alpha_generator = self.gen_perlin_noise_map(img, octaves=octaves)
            alpha = alpha_min + (alpha_multiplier * alpha_generator)
            alpha = (alpha ** sparsity) * density_multiplier
            alpha = np.nan_to_num(alpha)
            alpha = np.clip(alpha, 0.0, 1.0)
            return alpha


    def generate_haze(self, img, level):
            mapper = self.mapping_option[level]
            intensity_mean = np.random.rand() * (mapper['intensity_mean'][1] - mapper['intensity_mean'][0]) + mapper['intensity_mean'][0]
            octaves = mapper['octaves']
            density_multiplier = np.random.rand() * (mapper['density_multiplier'][1] - mapper['density_multiplier'][0]) + mapper['density_multiplier'][0]
            alpha_multiplier = np.random.rand() * (mapper['alpha_multiplier'][1] - mapper['alpha_multiplier'][0]) + mapper['alpha_multiplier'][0]
            sparsity = np.random.rand() * (mapper['sparsity'][1] - mapper['sparsity'][0]) + mapper['sparsity'][0]
            scale = np.random.rand() * (mapper['scale'][1] - mapper['scale'][0]) + mapper['scale'][0]
            interpolation = mapper['interpolation']
            alpha_min = np.random.rand() * (mapper['alpha_min'][1] - mapper['alpha_min'][0]) + mapper['alpha_min'][0]
            
            intensity_coarse = self.generate_intensity_map_coarse(img, intensity_mean=intensity_mean, scale=scale, interpolation=interpolation)
            intensity_fine = self.gen_intensity_map_fine(img, intensity_mean=intensity_mean, octaves=octaves)
            intensity = intensity_coarse + intensity_fine
            alpha = self.generate_alpha_mask(img, octaves=octaves, alpha_min=alpha_min, alpha_multiplier=alpha_multiplier, sparsity=sparsity, density_multiplier=density_multiplier)
            alpha = alpha[..., np.newaxis]
            intensity = intensity[..., np.newaxis]
            res = np.clip((1 - alpha) * img.astype(alpha.dtype) + alpha * intensity.astype(alpha.dtype), 0, 255).astype(np.uint8)
            return res


mapping_option = {
    'level1': {'intensity_mean':[230, 240], 'octaves':[1, 1], 
               'density_multiplier':[0.7, 0.9], 'alpha_multiplier':[0.2, 0.4], 
               'sparsity':[1.5, 1.7], 'scale':[0.8, 1.0], 
               'interpolation':cv2.INTER_CUBIC, 'alpha_min':[0.5, 0.7]},
    'level2': {'intensity_mean':[230, 240], 'octaves':[1, 4], 
               'density_multiplier':[1.0, 1.2], 'alpha_multiplier':[0.4, 0.6], 
               'sparsity':[1.7, 1.9], 'scale':[1.0, 1.2], 
               'interpolation':cv2.INTER_CUBIC, 'alpha_min':[0.5, 0.7]},
    'level3': {'intensity_mean':[230, 240], 'octaves':[1, 4, 6], 
               'density_multiplier':[1.2, 1.4], 'alpha_multiplier':[0.6, 0.8], 
               'sparsity':[1.7, 1.9], 'scale':[1.2, 1.4], 
               'interpolation':cv2.INTER_CUBIC, 'alpha_min':[0.5, 0.7]},
}

import time
s = time.time()
img = cv2.imread('51.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(1,4,1)
plt.imshow(img)

hazeGen = HazeGenerator(mapping_option)

res1 = hazeGen.generate_haze(img, level='level1')
plt.subplot(1,4,2)
plt.imshow(res1)

res2 = hazeGen.generate_haze(img, level='level2')
plt.subplot(1,4,3)
plt.imshow(res2)

res3 = hazeGen.generate_haze(img, level='level3')
plt.subplot(1,4,4)
plt.imshow(res3)

plt.show()
