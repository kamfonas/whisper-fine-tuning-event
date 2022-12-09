sanchit-gandhi — Today at 10:55 AM
Thanks for sharing your results @farsipal! I'll summarise them briefly here:

1) No fine-tuning: 31% WER (this is from Whisper paper on CV9, would be similar for CV11)
2) Fine-tune and freeze encoder: 25% WER
3) Fine-tune full model: 20% WER

I think this highlights what you get by training the encoder vs freezing it: the acoustic side of the model is able to better adapt to your source audio when you train it 
Personally I've only tried freezing the encoder with the medium model and on English. Here, I made the following observations:
1. The model converges faster when you freeze the encoder vs not (if you're training for a small number of steps, better to freeze)
2. If you train for long enough, not freezing the encoder almost always wins (if you're training for more steps, better not to freeze)
3. Freezing the encoder gets you much bigger batch sizes and a bit faster training
But I think with multilingual you'll get a lot of benefit from training the encoder, especially with smaller models. Here, model capacity plays a big role. For the small model, we don't have many model parameters, but lots of languages we need to learn. By fine-tuning the encoder, we can train the model to 'forget' some of the languages it doesn't need to know, and improve on the language we care about! 
With Hindi, I couldn't get it to work really freezing the encoder, the performance was nowhere near as good as training the full model. For Greek, it looks like freezing the encoder gets you halfway there, but really you need to fine-tune the full model
Also I think there's an inefficiency currently with our training script: when we disable gradients for the encoder, we still use gradient checkpointing. Gradient checkpointing slows down the forward pass at the expense of more memory efficient gradient computations, but since we don't need these gradients we can turn off checkpointing for the encoder! So we can anticipate a little bit more speed up when freezing the encoder in the optimised script
farsipal — Today at 11:04 AM
I agree. Although I didn't measure the clock time the duration was significantly less than half the unfrozen run.
Is there a way to, say... only freeze the lower half of the encoder layers via the api in place?
Are we restricted to using gradient checkpointing? Can we try forward and cache? 
sanchit-gandhi — Today at 11:19 AM
the duration was significantly less than half the unfrozen run
Okay that's pretty significant! We can speed this up even more by fixing the gradient checkpointing inefficiency!

only freeze the lower half of the encoder layers via the api in place?
Going into that level of detail will require you to change the modelling code in Transformers. You want to change this line: https://github.com/huggingface/transformers/blob/bf9a5882a7125a6050aaad0f52257f07df062d6a/src/transformers/models/whisper/modeling_whisper.py#L680
It needs to be:
if self.gradient_checkpointing and self.training and idx < NUM_LAYERS_TO_FREEZE:

where we set the variable NUM_LAYERS_TO_FREEZE as the number of encoder layers to freeze

Are you comfortable making changes to Transformers that are reflected in your environment? I can walk you through how to do this if not!

Are we restricted to using gradient checkpointing? Can we try forward and cache?
Not at all! We use gradient checkpointing because it facilitates much larger batch sizes. You can disable it in the training_args (gradient_checkpointing=False). Then just make sure that you also set model.config.use_cache=True before we instantiate the Trainer. You'll likely need to lower your per_device_train_batch_size by doing this. 
farsipal — Today at 11:42 AM
I am trying to use the python script. I found the sequencing to be a bit different and I was not sure if I needed to change anything regarding the pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=model_args.config_name, task=data_args.task)  (not sure if I got the arguments right)
farsipal — Today at 11:45 AM
I would appreciate any help I can get on modifying transformer code. I have looked into this before when trying to add a rank-based metric for quetsion-answering with squad but was not brave enough to do it yet 
Drishti — Today at 12:03 PM
Noted. Thanks for the clarification..
sanchit-gandhi — Today at 12:38 PM
Okay here's a rapid guide for setting it up! What we're going to do is:
1. Uninstall transformers from pip
2. Clone transformers from GitHub
3. cd into the transformers repo
4. Add upstream transformers so we can git pull
5. Install transformers from source (pip install transformers -e .)

pip uninstall transformers
cd ~/
git clone https://github.com/huggingface/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
pip install transformers -e .
 
Now you can view the modelling file:
vim ~/transformers/models/whisper/modeling_whisper.py

Making changes here will be reflected in your environment


sanchit-gandhi — Today at 12:38 PM wrote: 

Okay here's a rapid guide for setting it up! What we're going to do is:
1. Uninstall transformers from pip
2. Clone transformers from GitHub
3. cd into the transformers repo
4. Add upstream transformers so we can git pull
5. Install transformers from source (pip install transformers -e .)

```
pip uninstall transformers 
cd ~/ 
git clone https://github.com/huggingface/transformers.git 
cd transformers 
git remote add upstream https://github.com/huggingface/transformers.git 
pip install transformers -e . 
``` 
Now you can view the modelling file:
```
vim ~/transformers/models/whisper/modeling_whisper.py
```
Making changes here will be reflected in your environment

=========================================================================

farsipal — Dec 6 2022 at 5:53 PM

I am trying to run the small model using the latest python script for another frozen encoder run for translation from greek (el), but I think there are some things we can improve:

There is an error caused by the deprecated call (around liine 420) to model.freeze_feature_encoder() which I commented out.

There is no parameter to control caching, which seems to default to True, and it clashes with the gradient checkpointing (which is implicit for this model). At every step, it logs a warning about the incompatibility and that it changed use_cache to False. As a matter of fact, I don't see a way to change gradient checkpointing either  (at least from the args). We should be able to have an argument to enable/disable gradient_checkpointing and depending on it invoke enable/disable_gradient_checkpointing() on the model. Another option that would be nice to have is whether to stream or not. 

Maybe these parameters exist somewhere and I don't see them in the documentation or the code. So I only added the model_args.use_cache to avoid the delay caused by the warnings and will wait till tomorrow to add the enable_gradient_checkpointing and the streaming option. 

The run started around 5:30PM EST. I will let it run overnight and use a physical connection to avoid the wifi suspension when going to sleep.

-----------------------------------------------------------------------------------------------------

sanchit-gandhi — Dec 7 2022 at 6:50 AM

Here is your suggestion:

if self.gradient_checkpointing and self.training and idx < NUM_LAYERS_TO_FREEZE:

Sorry ignore this suggestion! I read through and misinterpreted what we were trying to achieve. There's no need to change this line - it won't change the freezing, only the gradient checkpointing. So we can leave it as is

What you're doing is correct - we want to incrementally freeze the encoder layers. You can test this, but it will look something like this (L610):
```
def _freeze_parameters(self):
    # freeze the conv 
    for param in self.conv1.parameters():
        param.requires_grad = False
    for param in self.conv2.parameters():
        param.requires_grad = False
    for param in self.embed_positions.parameters():
        param.requires_grad = False

    # freeze the first encoder layers
    for layer_idx in range(NUM_LAYERS_TO_FREEZE):
        for param in self.layers[layer_idx].parameters():
            param.requires_grad = False
```
This will freeze the first NUM_LAYERS_TO_FREEZE layers of the encoder
You can adapt this script for interleaving CV11 and FLEURS: https://github.com/huggingface/community-events/blob/main/whisper-fine-tuning-event/interleave_streaming_datasets.ipynb
You simply need to swap the dataset language code from "es" to "el", and remove the VoxPopuli and MultiLingual LibriSpeech parts:

```
dataset_names = ["mozilla-foundation/common_voice_11_0", "google/fleurs"]
dataset_config_names = ["es", "el_gr"]
text_column_names = ["sentence", "transcription"]
```

There is an error caused by the deprecated call (around liine 420) to model.freeze_feature_encoder() which I commented out.

We set this to "False" in our training arguments. 

sanchit-gandhi — Today at 7:04 AM
See the section in the README for using the Python script! https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event#python-script

It's got some helpful tips and start-to-finish instructions for how you can use these resources 🤗

Essentially, what you want to do is create a file called run.sh that has all the arguments that we pass to the training script. You can see an example of creating this file run.sh in the README. It has all the arguments that control our dataset, language, checkpoint and training args like gradient checkpointing, batch size, gradient accumulation steps, etc

There is an error caused by the deprecated call (around liine 420) to model.freeze_feature_encoder() which I commented out.
We set this to "False" in our training arguments, see the README:
--freeze_feature_encoder="False" \

it logs a warning about the incompatibility and that it changed use_cache to False
Will push a PR to fix this today, essentially when we load the config we can do:
```
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
    use_cache=False if training_args.gradient_checkpointing else True,
)
 ```
I don't see a way to change gradient checkpointing either  (at least from the args)
Gradient checkpointing is set by the arg --gradient_checkpointing. We set it to True in the example on the README. If you want to disable it, set the following in your training arguments file run.sh 
```
--gradient_checkpointing="False" \
```
(see README and docs on Seq2SeqTrainingArguments: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments.gradient_checkpointing)

We should be able to have an argument to enable/disable gradient_checkpointing
We have this! See above 🙂

Another option that would be nice to have is whether to stream or not. 

Very true! It would be nice to have a script where we can enable / disable streaming mode. Unfortunately, the two modes require slightly different loading logic, so we can't have them under one script for now. I'll add a script for non-streaming mode shortly: it will use exactly the same arguments as the streaming mode one, so you can swap them out 1-for-1. This is a good idea for the future though!
I hope that answers your questions @farsipal! In summary, you can control the training args with your file run.sh, including things like gradient checkpointing etc 🙂 Check out the README for more detailed instructions! https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event#python-script Best of luck with your training runs! Keep us posted with any results!

===================================================================================

 used your notebook https://huggingface.co/farsipal/whisper-small-el/blob/main/fine-tune-whisper-streaming-cf11-el-v2.ipynb
after I wrote to you, I decided to compare your notebook and the original.
model.config.forced_decoder_ids = None
 Set None like here and now it works. Do you know is it ok?
fine-tune-whisper-streaming-cf11-el-v2.ipynb · farsipal/whisper-sma...
fine-tune-whisper-streaming-cf11-el-v2.ipynb · farsipal/whisper-sma...
sanchit-gandhi — Today at 5:51 AM
Hey @Aigiz! 
model.config.forced_decoder_ids = None
This is correct! The forced_decoder_ids specify certain tokens that are 'forced' at the beginning of the generation process. The probabilities of these tokens are set to 1 in the decoding process. We use the 'forced' ids for inference with the pre-trained model to control the language and the task (transcribe or translate). 

For fine-tuning, we don't need to set any forced ids -> we train the model to predict the correct language and task tokens, so the fine-tuned model knows the correct language and task ids to set by itself 🙂 

Hope that makes sense!

========================================================================================

🤗 Updates from the HF team
1. Do make sure to check if your model shows up on our leaderboard: https://huggingface.co/spaces/autoevaluate/leaderboards?dataset=mozilla-foundation%2Fcommon_voice_11_0&only_verified=0&task=automatic-speech-recognition&config=-unspecified-&split=-unspecified-&metric=wer - Let's race to beat the SoTA!!
2. We heard that evaluation was a bottleneck for you all so we created a script: https://github.com/huggingface/community-events/blob/main/whisper-fine-tuning-event/run_eval_whisper_streaming.py
All you need to do to run your experiments is this:
python run_eval_whisper_streaming.py --model_id="openai/whisper-tiny.en" --config="en" --device=0

Scroll down to the end of the script to checkout the parameters that you can play with. Happy evaluating!! ❤️
3. Don't forget about our GPU giveaways for your spaces demos: https://discord.com/channels/879548962464493619/1050010333076537384/1050735976856690708
We want to help you showcase your hardwork better 🤗 

☁️ GPU on Lambda updates
1. Don't forget to shut down your GPUs when not in use. Give them a little rest.
2. Participate in the lambda cloud credits giveaway by tweeting your models & demos. All you need to do is tag @huggingface & @lambdaapi. Don't miss out on the chance to win 300 A100 GPU hours!!

====================================================================================

VB — Dec 9 at 6:29 AM
@ml-4-audio - Some really exciting news!! We have decided to grant GPU upgrades for you to show off your fine-tuned models! You can use our official demo: https://huggingface.co/spaces/whisper-event/whisper-demo to retrofit it for your model.

Here's what you'd need to do to claim it:
1. Go to the whisper-demo space,
2. Cliick on the three vertical dots on the right side  Duplicate this Space
3. In the new space go to app.py and update your MODEL_NAME and lang
4. Then head over to settings and click on Apply for a community grant - You can tag both reach-vb & sanchit-gandhi in your request.

This all won't take more than 5 minutes. We'll be granting GPUs to those whose models are more closer to the SoTA (in their chosen language)! 🔥

P.S. Don't forget to post them here in this channel :)) 
Whisper Demo - a Hugging Face Space by whisper-event

========================================================================================

jilp — Yesterday at 4:44 PM
Hi, I created scripts for speeding up the environment setup for Lambda cloud instances. Here's how to use them:

1. Initialize the Lambda instance and launch the Cloud IDE.

2. In Jupyter Lab, click on "Upload Files" and upload the two scripts install_ffmpeg_env.py and install_venv_env.py.

3. SSH into the GPU from your local machine and run this command from your local device: python install_ffmpeg_env.py.

4. You'll know the first script is finished when your virtual environment name is at the start of your command line.

5. Run this command python install_venv_env.py and your environment will be ready.

I hope they save you time and effort!
import subprocess

env_name = input("Enter the name of your virtual environment: ")

commands = [
    "sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4",
    "sudo apt update",
    "sudo apt install -y ffmpeg",
    "sudo apt-get install git-lfs",
    f"python3 -m venv {env_name}",
    f"echo \"source ~/{env_name}/bin/activate\" >> ~/.bashrc",
    "bash"
]

for command in commands:
    subprocess.run(command, shell=True)
Collapse
install_ffmpeg_env.py
1 KB
import subprocess
import os

# clone the community-events repository
subprocess.run(["git", "clone", "https://github.com/huggingface/community-events.git"])

# install the required packages
os.chdir("community-events/whisper-fine-tuning-event")
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# configure git credential helper
subprocess.run(["git", "config", "--global", "credential.helper", "store"])

# login to huggingface cli
subprocess.run(["huggingface-cli", "login"])

# install 🤗 libraries
subprocess.run(["pip", "install", "--quiet", "datasets", "git+https://github.com/huggingface/transformers",\
    "evaluate", "huggingface_hub", "jiwer", "bitsandbytes", "accelerate"])

Note: placed these scripts in setup folder

=========================================================================================
