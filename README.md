# Hybrid Stereo Matching

### Overview

This framework for stereo matching couples several computer vision algorithms into a customizable workflow
for purely event- or mixed event- and frame-based stereo matching. It is devised to take an input from a DVS [ref]
or DAVIS [ref] vision sensors and produce an output of asynchronous depth-resolved stream of events and in the 
case of hybrid stereo matching -- additional sequence of synchronous depth maps.
 
The event-based stereo matching relies on a neuromorphing computing platform, such as SpiNNaker [ref], while the 
frame-based algorithms run on normal CPUs. 

### Requirements

- SpiNNaker 
- ordinary PC, tested on (8 core, 8 GB RAM, Ubuntu 16.04)
- see requirements.txt for the necessary packages

### Experimental setup

##### Reproducing the experiments from [ref]:
TODO

##### Launching a custom experiment:
TODO

