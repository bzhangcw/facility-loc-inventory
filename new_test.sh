# 1 2 3 7 5; 13 14 15
python rounding_ray.py --conf_label 1 --pick_instance 13 &>1_13.log
python rounding_ray.py --conf_label 1 --pick_instance 14 &>1_14.log
python rounding_ray.py --conf_label 1 --pick_instance 15 &>1_15.log

python rounding_ray.py --conf_label 2 --pick_instance 13 &>2_13.log
python rounding_ray.py --conf_label 2 --pick_instance 14 &>2_14.log
python rounding_ray.py --conf_label 2 --pick_instance 15 &>2_15.log

python rounding_ray.py --conf_label 3 --pick_instance 13 &>3_13.log
python rounding_ray.py --conf_label 3 --pick_instance 14 &>3_14.log
python rounding_ray.py --conf_label 3 --pick_instance 15 &>3_15.log

python rounding_ray.py --conf_label 7 --pick_instance 13 &>7_13.log
python rounding_ray.py --conf_label 7 --pick_instance 14 &>7_14.log
python rounding_ray.py --conf_label 7 --pick_instance 15 &>7_15.log

python rounding_ray.py --conf_label 5 --pick_instance 13 &>5_13.log
python rounding_ray.py --conf_label 5 --pick_instance 14 &>5_14.log
python rounding_ray.py --conf_label 5 --pick_instance 15 &>5_15.log
