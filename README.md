# GPT-53.1M
A Generative Pre-trained Transformer model trained on the OpenWebText dataset (size: 40gb).  <br>
The model has 53.1 million trainable parameters. See more details from below. <br><br>
NOTE: This project is not currently "finished" and will be updated later. <br>
There is still a lot to do with hyperparameters, tokenization methods, text generating methods, fine-tuning. <br>
<br>
Generates somewhat natural human-like text✅
<br>
Unable to post the state-dict for the model because it is too large to upload here. (219MB)


# Outputs:
I am using "My name is Vex and I'm " as the initial "prompt" for the model. Currently the model has trained for 200 000 training steps and it's <b>loss</b> sitting comfortable at around 1.1.<br>
Note that some of the outputs seem like they are cut out. This is because the amount of tokens generated is limited.
<br><br>
Tokens generated:  87 <br>
<i>My name is Vex and I'm </i>still on the howl of a girl. Some of this is a arrest gift for m 

Tokens generated:  87 <br>
<i>My name is Vex and I'm </i>Feminist. Like it’s the club, I am still like, it's phasing out. 

Tokens generated:  87 <br>
<i>My name is Vex and I'm </i>obvious I’ve got it to my end with my mind before I get some con 

Tokens generated:  87 <br>
<i>My name is Vex and I'm </i>more likely to pull yourself down to the west-side Ocean Club in 

Tokens generated:  87 <br>
<i>My name is Vex and I'm </i>not sure what they about doing, schools Alternative History III  

# Training and testing graph
The model has been training for ~200k training steps which is ~20h of GPU time, which is quite low. <br>
Visual representation of the loss can be seen below in graphs. <br>
Red line: Training loss <br>
Blue line: Testing loss <br>
<br>
![0-10k training_with_current_best_model_LOSS_TRACK_red_TRAINING__Blue_TESTING_LOSS](https://github.com/asuzi/GPT-53.1M/assets/61744031/48e16d7e-ec93-4d9f-8303-8f73a838acc1)
0 - 10 000 steps
<br>
![100k-200k training loss track](https://github.com/asuzi/GPT-53.1M/assets/61744031/a06e5c07-7a1f-4eff-8233-b0d7eec3b279)
100 000 - 200 000 steps
<br>

# Architecture
The model architecture is the same as described as "original GPT model", in https://en.wikipedia.org/wiki/Generative_pre-trained_transformer
![image](https://github.com/asuzi/GPT-53.1M/assets/61744031/54849745-494f-4781-b729-1bae2ee7a6c5)

<br>

# Information about the model
Currently the model uses ~11gb video ram while training on the GPU. <br>
With torchinfo.summary() we can check various information about the model.
![image](https://github.com/asuzi/GPT-53.1M-PRIVATE/assets/61744031/6f20f208-4cc5-4fdb-be67-878ca44574d4)


