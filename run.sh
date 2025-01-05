python3 fixmatch.py --config_yml /home/student02/hieu/Projects/SSSS/Configs/cf_fugc.yml --exp fixmatch
python3 train_sup.py --config_yml /home/student02/hieu/Projects/SSSS/Configs/cf_fugc.yml --exp train_sup
python3 cps.py --config_yml /home/student02/hieu/Projects/SSSS/Configs/cf_fugc.yml --exp cps


tmux new -s my_session

tmux attach-session -t my_session
tmux attach-session -t my_session1