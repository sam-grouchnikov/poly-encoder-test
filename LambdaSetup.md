## Executing a training loop using Lambda Cloud GPUs (Windows Powershell)

### 1. Generate a new SSH key
```commandline
ssh-keygen -t rsa
```

### 2. Save key to chosen path
```commandline
path-to-desired-file (click enter for default path)
```
Optional: enter passphrase to access SSH key

### 3. Add key to Lambda Cloud
1. Go to Lambda Cloud dashboard
2. Go to SSH Keys -> Click add new
3. Open the public key in terminal
4. Copy **PUBLIC** key into the pop-up
5. Name the key appropriately (remember the name for future use)

### 4. Launch Lambda Cloud instance
1. Go to Lambda Cloud dahsboard
2. Go to instances
3. Click launch instance
4. Select GPU type, region, SSH key, etc.

### 5. Connect local machine to cloud instance
~~~commandline 
ssh -i "path\to\private\key" ubuntu@instance ip
~~~
Note: Instance IP generated once instance is fully booted, appears on instances dashboard in Lambda Cloud. Must use **PRIVATE** key in this step.

### From this point on, commands should use Linux formatting

### 6. Update system packages
~~~commandline
sudo apt update && sudo apt upgrade -y
~~~

### 7. Clone GitHub Project
~~~commandline
git clone github.com/your-username/repo-name.git
cd repo-name
~~~
Repository must either be public, or user must log into GitHub with credentials

### 8. Set up Python and Venv
~~~commandline
sudo apt install python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
~~~

### 9. (Optional) Configure Weights and Biases
~~~commandline
pip install wandb
wandb login
~~~
Must have WandB account. Go to WandB website, generate an API key, and enter it in the terminal when prompted (this is how WandB login works)


### 10. Run code
~~~commandline
python file-to-run.py
~~~

### Important: Terminate instance on website when done