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