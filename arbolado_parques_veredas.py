# -*- coding: utf-8 -*-
"""
@author: Melisa Di Giacomo
"""

#%% Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
import seaborn as sns
from scipy.stats import ttest_ind


#%% Dowload databases from website
def download(t_url, filename):
    response = urlopen(t_url)
    data = response.read()
    txt_str = str(data)
    lines = txt_str.split("\\n")
    des_url = filename
    fx = open(des_url,"w")
    for line in lines:
        fx.write(line+ "\n")
    fx.close()

# Download arbolado-publico-lineal-2017-2018
# Dataset of Buenos Aires trees on sidewalks
uurl1 = "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/arbolado-publico-lineal/arbolado-publico-lineal-2017-2018.csv"
filename1 = "./arbolado-publico-lineal-2017-2018.csv"
download(uurl1, filename1)

# Download arbolado-en-espacios-verdes
# Dataset of Buenos Aires trees on parks
uurl2 = "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/arbolado-en-espacios-verdes/arbolado-en-espacios-verdes.csv"
filename2 = "./arbolado-en-espacios-verdes.csv"
download(uurl2, filename2)


#%% Functions
def especie_seleccionada(especie_vereda, especie_parque):
    '''Based on the scientific name of the species of a Buenos Aires tree, it
    returns a DataFrame with the diameters and environment.
    
    Pre: especie_vereda is the scientific name in the dataset
         'arbolado-publico-lineal-2017-2018.csv'. Environment: 'VEREDA'.
         especie_parque is the scientific name in the dataset
         'arbolado-en-espacios-verdes.csv'. Environment: 'PARQUE'.
    Pos: DataFrame with the diameter and the environment
         of the species of interest.
    '''

    df_veredas = pd.read_csv('./arbolado-publico-lineal-2017-2018.csv', header = 0)
    df_parques = pd.read_csv('./arbolado-en-espacios-verdes.csv', header = 0)
    
    # Selected columns
    cols_sel_veredas = ['diametro_altura_pecho']
    cols_sel_parques = ['diametro']
    
    # Selected species
    especie_seleccionar_veredas = [especie_vereda]
    especie_seleccionar_parques = [especie_parque]
    
    # Select columns and species and copy the dataset
    df_especie_veredas= df_veredas[df_veredas['nombre_cientifico'].isin(especie_seleccionar_veredas)][cols_sel_veredas].copy()
    df_especie_parques= df_parques[df_parques['nombre_cie'].isin(especie_seleccionar_parques)][cols_sel_parques].copy()
    
    # Rename columns because they have different names in the two datasets
    df_especie_veredas = df_especie_veredas.rename(columns = {'diametro_altura_pecho':'diametro'})
    
    # Add column environment ("ambiente") in each dataset
    df_especie_veredas["ambiente"]= "vereda"
    df_especie_parques["ambiente"]= "parque"
    
    # Concat the two datasets
    df_especie = pd.concat([df_especie_veredas, df_especie_parques])

    return df_especie


def boxplot_diametros(df_especie):
    '''Based on the DataFrame of the species of interest, it
    returns a boxplot of the diameter per environment (PARQUE and VEREDA).
    
    Pre: df_especie is a DataFrame that contains the diameters and
         environment of the species of interest.
    Pos: boxplot of the diameters per environment (PARQUE and VEREDA).
    '''
    # Boxplot with Seaborn
    plt.clf()
    sns.boxplot(y='diametro', x='ambiente', 
                     data=df_especie, 
                     width=0.5,
                     palette="Set2")
    plt.title("Boxplot diameters per environment (VEREDA and PARQUE)")
    return plt.show()


def ttest(df_especie):
    '''Calculate the T-test for the means of two independent samples of scores.
    
    Pre: df_especie is a DataFrame that contains the diameters and
         environment of the species of interest.
    Pos: returns the  t-statistic and the p value.
    '''
    # Variables to compare
    vereda = df_especie.where(df_especie.ambiente== 'vereda').dropna()['diametro']
    parque = df_especie.where(df_especie.ambiente== 'parque').dropna()['diametro']
    
    # t test
    t_stat, p = ttest_ind(vereda, parque)
    print(f't={t_stat}, p={p}')


def pairplot(especies_parque):
    '''Based on the scientific name of different species of tress in Buenos 
    Aires parks, it returns a pairplot of heights and diameters of the species
    of interest.
    
    Pre: especies_parque are the scientific names of the species of interest.
    Pos: pairplot of heights and diameters of the species of interest.
    '''
    df_parques= pd.read_csv('./arbolado-en-espacios-verdes.csv', header = 0)

    # Selected columns
    cols_sel=['nombre_cie', 'altura_tot', 'diametro']

    # Selected species
    especies_sel = especies_parque

    # Select columns and species and copy the dataset
    df_especie_parques= df_parques[df_parques['nombre_cie'].isin(especies_sel)][cols_sel].copy()

    #Pairplot
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    sns.pairplot(data = df_especie_parques[cols_sel], hue = 'nombre_cie')
    return plt.show()


def mostcommon(n):
    '''n most common trees in Buenos Aires parks.
    Pre: n is the number of most common trees selected.
    Pos: Series of n most common trees in Buenos Aires parks.'''

    df_parques = pd.read_csv('./arbolado-en-espacios-verdes.csv', header = 0)

    # Quantify species and get 10 most common
    arbol = df_parques['nombre_com'].value_counts()
    arbol_mostcommon = arbol.head(n)
    return arbol_mostcommon


# Class BubbleChart
class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        """
        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center')

#%% Most common tree in Buenos Aires parks

# 10 most common
arbol_10_mostcommon = mostcommon(10)

# Set n=10 pastels colors
pastels = ['#98D4BB', '#E5DB9C', '#A15D98', '#E5B3BB', '#9AD9DB', '#F9968B', '#F4C815', '#C54B6C', '#CCD4BF', '#218B82']

# Plot
bubble_chart = BubbleChart(area=arbol_10_mostcommon, bubble_spacing=0.1)
bubble_chart.collapse()

fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
bubble_chart.plot(ax, arbol_10_mostcommon.index, pastels)
ax.axis("off")
ax.relim()
ax.autoscale_view()
ax.set_title('Most common trees in Buenos Aires parks', fontsize=15)
plt.show()


#%% Most common Species: Eucalyptus sp.

df_eucalyptus = especie_seleccionada('Eucalyptus sp.', 'Eucalyptus sp.')
# Boxplot
boxplot_diametro_eucalyptus = boxplot_diametros(df_eucalyptus)
# T-test
comparisson = ttest(df_eucalyptus)


#%% 3 most common species in Buenos Aires parks: Eucalyptus, Tipa and Jacarandá

especies_seleccionadas = ['Eucalyptus sp.', 'Tipuana Tipu', 'Jacarandá mimosifolia']
pairplot(especies_seleccionadas)