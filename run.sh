python train_classifier.py --epoch=1 --only_self=True --only_official=False --lr=3e-4
python train_classifier.py --epoch=6 --weight_path=checkpoints/se_resnext101_32x4d/backup/model_best.pth --only_self=False --only_official=True --lr=2e-5
