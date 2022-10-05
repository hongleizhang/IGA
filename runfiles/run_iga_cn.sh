<<<<<<< HEAD
nohup python -u main.py new -d ../data_path/coco -b 32 -m 90 -r 30 --use-mc --noise "crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()" --name iga_cn> iga_cn_nohup.out 2>&1 &
=======
nohup python -u main.py new -d ../data_path/coco -b 32 -m 30 --noise "crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()" --name iga_cn> iga_cn_nohup.out 2>&1 &
>>>>>>> 20040681ef050522b5ce942e262ef55a0f48643e
