# HTCondor + Hadoop + Python

## Job working directory

after testing, condor job user is **nobody**

hence, make working directory for **nobody**

create working directory

```shell
sudo mkdir /home/nobody
```

change ownership of this directory

```shell
sudo chown nobody:nogroup /home/nobody/
```

change authority of this directory, to let everyone can read(4), write(2), execute(1)

```shell
sudo chmod 777 /home/nobody/
```

And define the job working directory at the first line of sub file.

## Job definition

### Pre-condition 

To prevent configure python runtime environment each time, Every instances have to install pip venv.

```shell
sudo apt install python3-pip -y
```

```shell
sudo apt install python3.10-venv -y
```

Note: python3.10-venv is determined by python version, which installed in the linux.

### File Tree Structure

arrival.sub

----> setup.sh



setup.sh

--> requirements.txt

--> main.py



main.py

--> json_parse.py

-->config.json

## HTCondor Submission

```shell
condor_submit arrival.sub
```

