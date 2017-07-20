# Hybrid Stereo Matching

## Overview

This framework for stereo matching couples several computer vision algorithms into a customizable workflow
for purely event- or mixed event- and frame-based stereo matching. It is devised to take an input from a DVS
or DAVIS vision sensors and produce an output of asynchronous depth-resolved stream of events, and in the 
case of hybrid stereo matching -- additional sequence of synchronous depth maps.
 
The event-based stereo matching relies on a neuromorphic computing platform, such as SpiNNaker, while the 
frame-based algorithms run on normal CPUs. 

## Requirements

The hardware requirements consist of:
- SpiNN-5 board (or multiple)
- ordinary PC, (tested on 8 core, 8 GB RAM, Ubuntu 16.04)

The code is in Python and for the necessary packages see `requirements.txt`.

## Experiments

There are two major types of experiments that are currently supported:

* SNN simulaiton on a SpiNNaker machine to compute an asynchronous stream of depth events. 
    - this can be done offline with spikes recording
    - or online with live output (live input is not supported yet)
* SNN simulation with frame-based matching
    - if in offline mode, first the SNN simulation is run and then the frame-based matching
    - using the live SNN output, the frame-based matching can run in parallel to the spiking simulation


#### Data
There are three hybrid experimental datasets, for which retina events and frames are available. 
Additionally there are three more datasets for which only events are present. The evaluation of the SNN on them as well 
as a detailed description of the recording and ground-truth data can be found in [1]

#### Running custom experiments:

The easiest way to prepare a custom experiment is to write a configuration file and put it in `experiments/configs`. 
Then run `main.py` with the path to the new yaml-based config file. For a reference configuration 
see `experiments/configs/template.yaml`. 

##### Hybrid mode

##### SNN-only mode

