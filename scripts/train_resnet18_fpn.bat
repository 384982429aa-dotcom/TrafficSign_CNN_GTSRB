@echo off
cd /d E:\class_work\TrafficSign_CNN_GTSRB
python src\training\train.py --data-dir E:\ML\GTSRB --img-size 64 --batch-size 64 --epochs 30
pause
