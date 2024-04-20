import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import pyplot as plt
N_SAMPLES = 700
PERCENT_TRAIN = 0.8


class Plotter:

    def __init__(self, regularization, poly_degree, print_images=False):
        self.reg = regularization
        self.POLY_DEGREE = poly_degree
        self.print_images = print_images
        self.rng = np.random.RandomState(seed=10)

    def init_figure(self, title):
        figure = go.Figure()
        camera = dict(eye=dict(x=1, y=-1.9, z=0.8), up=dict(x=0, y=0, z=1))
        figure.update_layout(title=title, scene=dict(xaxis_title=
            'Feature 1', yaxis_title='Feature 2', zaxis_title='Y', camera=
            camera), scene_aspectmode='cube', height=700, width=800,
            autosize=True)
        return figure

    def print_figure(self, figure, title):
        fig_title = title.replace(' ', '_')
        path = f'outputs/{fig_title}.png'
        figure.write_image(path)
        img = mpimg.imread(path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def create_data(self):
        rng = self.rng
        true_weight = rng.rand(self.POLY_DEGREE ** 2 + 2, 1)
        x_feature1 = np.linspace(-5, 5, N_SAMPLES)
        x_feature2 = np.linspace(-3, 3, N_SAMPLES)
        x_all = np.stack((x_feature1, x_feature2), axis=1)
        reg = self.reg
        x_all_feat = reg.construct_polynomial_feats(x_all, self.POLY_DEGREE)
        x_cart_flat = []
        for i in range(x_all_feat.shape[0]):
            point = x_all_feat[i]
            x1 = point[:, 0]
            x2 = point[:, 1]
            x1_end = x1[-1]
            x2_end = x2[-1]
            x1 = x1[:-1]
            x2 = x2[:-1]
            x3 = np.asarray([[(m * n) for m in x1] for n in x2])
            x3_flat = list(np.reshape(x3, x3.shape[0] ** 2))
            x3_flat.append(x1_end)
            x3_flat.append(x2_end)
            x3_flat = np.asarray(x3_flat)
            x_cart_flat.append(x3_flat)
        x_cart_flat = np.asarray(x_cart_flat)
        x_cart_flat = (x_cart_flat - np.mean(x_cart_flat)) / np.std(x_cart_flat
            )
        x_all_feat = np.copy(x_cart_flat)
        p = np.reshape(np.dot(x_cart_flat, true_weight), (N_SAMPLES,))
        y_noise = rng.randn(x_all_feat.shape[0], 1)
        y_all = np.dot(x_cart_flat, true_weight) + y_noise
        print('x_all: ', x_all.shape[0], ' (rows/samples) ', x_all.shape[1],
            ' (columns/features)', sep='')
        print('y_all: ', y_all.shape[0], ' (rows/samples) ', y_all.shape[1],
            ' (columns/features)', sep='')
        return x_all, y_all, p, x_all_feat

    def split_data(self, x_all, y_all):
        rng = self.rng
        all_indices = rng.permutation(N_SAMPLES)
        train_indices = all_indices[:round(N_SAMPLES * PERCENT_TRAIN)]
        test_indices = all_indices[round(N_SAMPLES * PERCENT_TRAIN):]
        xtrain = x_all[train_indices]
        ytrain = y_all[train_indices]
        xtest = x_all[test_indices]
        ytest = y_all[test_indices]
        return xtrain, ytrain, xtest, ytest, train_indices, test_indices

    def plot_all_data(self, x_all, y_all, p):
        df = pd.DataFrame({'feature1': x_all[:, 0], 'feature2': x_all[:, 1],
            'y': np.squeeze(y_all), 'best_fit': np.squeeze(p)})
        title = 'All Simulated Datapoints'
        fig = self.init_figure(title)
        fig.add_scatter3d(x=df['feature1'], y=df['feature2'], z=df['y'],
            mode='markers', marker=dict(color='blue', size=8, opacity=0.12),
            name='Data Points')
        fig.add_scatter3d(x=df['feature1'], y=df['feature2'], z=df[
            'best_fit'], mode='lines', line=dict(color='red', width=7),
            name='Line of Best Fit')
        config = {'scrollZoom': True}
        fig.show(config=config)
        if self.print_images:
            self.print_figure(title)

    def plot_split_data(self, xtrain, xtest, ytrain, ytest):
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        train_df = pd.DataFrame({'feature1': xtrain[:, 0], 'feature2':
            xtrain[:, 1], 'y': ytrain, 'label': 'Training'})
        test_df = pd.DataFrame({'feature1': xtest[:, 0], 'feature2': xtest[
            :, 1], 'y': ytest, 'label': 'Testing'})
        title = 'Data Set Split'
        fig = self.init_figure(title)
        fig.add_scatter3d(x=train_df['feature1'], y=train_df['feature2'], z
            =train_df['y'], mode='markers', marker=dict(color='yellow',
            size=2, opacity=0.75), name='Training')
        fig.add_scatter3d(x=test_df['feature1'], y=test_df['feature2'], z=
            test_df['y'], mode='markers', marker=dict(color='red', size=2,
            opacity=0.75), name='Testing')
        fig.show()
        if self.print_images:
            self.print_figure(title)

    def plot_linear_closed(self, xtrain, xtest, ytrain, ytest, x_all, y_pred):
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        train_df = pd.DataFrame({'feature1': xtrain[:, 0], 'feature2':
            xtrain[:, 1], 'y': ytrain, 'label': 'Training'})
        test_df = pd.DataFrame({'feature1': xtest[:, 0], 'feature2': xtest[
            :, 1], 'y': ytest, 'label': 'Testing'})
        pred_df = pd.DataFrame({'feature1': x_all[:, 0], 'feature2': x_all[
            :, 1], 'Trendline': np.squeeze(y_pred)})
        title = 'Linear (Closed)'
        fig = self.init_figure(title)
        fig.add_scatter3d(x=train_df['feature1'], y=train_df['feature2'], z
            =train_df['y'], mode='markers', marker=dict(color='yellow',
            size=2, opacity=0.75), name='Training')
        fig.add_scatter3d(x=test_df['feature1'], y=test_df['feature2'], z=
            test_df['y'], mode='markers', marker=dict(color='red', size=2,
            opacity=0.75), name='Testing')
        fig.add_scatter3d(x=pred_df['feature1'], y=pred_df['feature2'], z=
            pred_df['Trendline'], mode='lines', line=dict(color='red',
            width=7), name='Trendline')
        fig.show()
        if self.print_images:
            self.print_figure(title)

    def plot_linear_gd(self, xtrain, xtest, ytrain, ytest, x_all, y_pred):
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        train_df = pd.DataFrame({'feature1': xtrain[:, 0], 'feature2':
            xtrain[:, 1], 'y': ytrain, 'label': 'Training'})
        test_df = pd.DataFrame({'feature1': xtest[:, 0], 'feature2': xtest[
            :, 1], 'y': ytest, 'label': 'Testing'})
        pred_df = pd.DataFrame({'feature1': x_all[:, 0], 'feature2': x_all[
            :, 1], 'Trendline': np.squeeze(y_pred)})
        title = 'Linear (GD)'
        fig = self.init_figure(title)
        fig.add_scatter3d(x=train_df['feature1'], y=train_df['feature2'], z
            =train_df['y'], mode='markers', marker=dict(color='yellow',
            size=2, opacity=0.75), name='Training')
        fig.add_scatter3d(x=test_df['feature1'], y=test_df['feature2'], z=
            test_df['y'], mode='markers', marker=dict(color='red', size=2,
            opacity=0.75), name='Testing')
        fig.add_scatter3d(x=pred_df['feature1'], y=pred_df['feature2'], z=
            pred_df['Trendline'], mode='lines', line=dict(color='red',
            width=7), name='Trendline')
        fig.show()
        if self.print_images:
            self.print_figure(title)

    def plot_linear_gd_tuninglr(self, xtrain, xtest, ytrain, ytest, x_all,
        x_all_feat, learning_rates, weights):
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        train_df = pd.DataFrame({'feature1': xtrain[:, 0], 'feature2':
            xtrain[:, 1], 'y': ytrain, 'label': 'Training'})
        test_df = pd.DataFrame({'feature1': xtest[:, 0], 'feature2': xtest[
            :, 1], 'y': ytest, 'label': 'Testing'})
        title = 'Tuning Linear (GD)'
        fig = self.init_figure(title)
        fig.add_scatter3d(x=train_df['feature1'], y=train_df['feature2'], z
            =train_df['y'], mode='markers', marker=dict(color='yellow',
            size=2, opacity=0.75), name='Training')
        fig.add_scatter3d(x=test_df['feature1'], y=test_df['feature2'], z=
            test_df['y'], mode='markers', marker=dict(color='red', size=2,
            opacity=0.75), name='Testing')
        colors = ['green', 'blue', 'pink']
        for ii in range(len(learning_rates)):
            y_pred = self.reg.predict(x_all_feat, weights[ii])
            y_pred = np.reshape(y_pred, (y_pred.size,))
            pred_df = pd.DataFrame({'feature1': x_all[:, 0], 'feature2':
                x_all[:, 1], 'Trendline': np.squeeze(y_pred)})
            fig.add_scatter3d(x=pred_df['feature1'], y=pred_df['feature2'],
                z=pred_df['Trendline'], mode='lines', line=dict(color=
                colors[ii], width=7), name='Trendline LR=' + str(
                learning_rates[ii]))
        fig.show()
        if self.print_images:
            self.print_figure(title)

    def plot_10_samples(self, x_all, y_all_noisy, sub_train, y_pred, title):
        samples_df = pd.DataFrame({'feature1': x_all[sub_train, 0],
            'feature2': x_all[sub_train, 1], 'y': np.squeeze(y_all_noisy[
            sub_train]), 'label': 'Samples'})
        pred_df = pd.DataFrame({'feature1': x_all[:, 0], 'feature2': x_all[
            :, 1], 'Trendline': np.squeeze(y_pred)})
        fig = self.init_figure(title)
        fig.update_layout(scene=dict(zaxis=dict(range=[-500, 500])))
        fig.add_scatter3d(x=samples_df['feature1'], y=samples_df['feature2'
            ], z=samples_df['y'], mode='markers', marker=dict(color='red',
            opacity=0.75, size=6, symbol='x', line=dict(width=1, color=
            'red')), name='Samples')
        fig.add_scatter3d(x=pred_df['feature1'], y=pred_df['feature2'], z=
            pred_df['Trendline'], mode='lines', line=dict(color='blue',
            width=7), name='Trendline')
        fig.show()
        if self.print_images:
            self.print_figure(title)
