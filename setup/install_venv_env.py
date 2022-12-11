import subprocess
import os

# clone a copy of the community-events repository
subprocess.run(["git", "clone", "https://github.com/kamfonas/whisper-fine-tuning-event.git"])

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
