# Bayesian Methods for Image Restoration

## Overview
This project implements Bayesian statistical methods for image restoration from degraded/distorted images. We assume that we observe image $y = (y_{i,j})_{i=1,...,N, j=1,...,M}$ instead of the original image $x = (x_{i,j})_{i=1,...,N, j=1,...,M}$, where the original has been corrupted or degraded.

## Problem Definition
In this context, $y_{i,j}$ represents the attribute of pixel $(i,j)$, such as grayscale level on a scale from 0 to $n$, or color encoded in RGB or CMYK format.

Our goal is to obtain the best possible reconstruction of image $x$ from the observed image $y$.

## Applications
1. Restoration of old, faded photographs
2. Processing digital camera images from CCD/CMOS sensors that contain visible noise (Digital cameras count photons hitting individual sensors, resulting in raw images with noise that require algorithms to remove this noise and produce a "real" image)

## Implementation Requirements
This project requires implementing image restoration models based on Bayesian statistics methods for two prior distributions describing the distribution of the real image:

### 1. Potts Model
$$-\log \mathbb{P}(X = x) = c + \beta \sum_{(i,j),(k,l):(i,j)\sim(k,l)} (1 - \delta(x_{i,j}, x_{k,l}))$$

where $(i,j) \sim (k,l)$ indicates that pixel $(i,j)$ is adjacent to $(k,l)$, $\delta(a,b) = 1_{\{a=b\}}$, and $c$ is a normalizing constant.

### 2. Truncated Quadratic Energy Function (preserving contrast)
$$-\log \mathbb{P}(X = x) = c + \sum_{(i,j),(k,l):(i,j)\sim(k,l)} \max\{\lambda^2(x_{i,j} - x_{k,l})^2, \alpha\}$$

where $\lambda, \alpha$ are parameters, and $c$ is a normalizing constant.

## Technical Details
The project specifically aims to compute two estimators of the recovered image:
1. MAP (maximum a posteriori)
2. MMS (minimum mean square)

We assume that noise is additive, i.i.d., and follows a normal distribution $N(0, \sigma^2)$.

From an algorithmic perspective, this requires implementing simulated annealing and the Gibbs sampler.

