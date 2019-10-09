# selfiebooth
Raspberry Pi 4 selfie booth.

## About
The Selfiebooth captures selfies (based on pre-trained HAAR cascades) and saves the images so that they can be fed to a machine learning algorithm. The project currently includes two options for the algorithm: LBPH (OpenCV) and SelfieNet, which is a shallow convolutional neural network using Keras/Tensorflow.  

## Usage
Install using `git clone`. Make sure you have a webcam connected to your RasPi (or other device you are using.)

TODO: Add requirements.txt.

Capture selfies of persons using:
```
python selfie-generator.py –-name nameoftheperson
```

The program will display the webcam image in a window. A face found by HAAR detector is highlighted using a green rectangle; make sure this rentangle doesn't pick any other areas from the image area than your face. It is does, change composition or remove unnecessary items from the image area. After this, press space bar to start capturing selfies. Pressing space again will stop recording, pressing 'Q' will quit the software. Make sure you capture selfies of at least 3 different persons before continuing to training the network; otherwise you will get error messages.

Train the network using:
```
python train.py --conf conf/selfienet.conf
```

The program will train SelfieNet on a Rasbperry Pi 4 for around 12 minutes. If you want, you can also train the LBPH using lbph.conf as a parameters. The models exist in separate files, so you can train both.

Finally, you can launch the selfiebooth that will recognize people in the image area:
```
python selfiebooth --conf conf/selfienet.conf
```

The project has been tested with 21 people under a fairly controlled environment; test results were surprisingly good considering the shallowness of the network. Adding another CONV layer (and ACT and POOL) to the SelfieNet improves the results, but of course increases the time it takes to train the network. My suggestion is to use rsync/scp to move data between Raspberry and a server running Tensorflow-GPU. Tesla V100 trains the same network a lot faster (12 minutes vs. 5 seconds).

## Thank you
This is a project based on PyImageSearch photo booth and courses from the same site, and thus, thank you, Adrian!
