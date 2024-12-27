# HTCondor + Hadoop + Python

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

