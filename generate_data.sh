# pick_instance: 5/7/8/9
# conf_label: 1/2/3/5/6/8
# data_now:1219
# backorder : 0


python template_run.py --template_choose 'us' --demand_type 1 &> us_1.log 
python template_run.py --template_choose 'us' --demand_type 2 &> us_2.log 
python template_run.py --template_choose 'us' --demand_type 3 &> us_3.log 
python template_run.py --template_choose 'us' --demand_type 4 &> us_4.log 
python template_run.py --template_choose 'us' --demand_type 5 &> us_5.log 

python template_run.py --template_choose 'sechina' --demand_type 1 &> sechina_1.log 
python template_run.py --template_choose 'sechina' --demand_type 2 &> sechina_2.log 
python template_run.py --template_choose 'sechina' --demand_type 3 &> sechina_3.log 
python template_run.py --template_choose 'sechina' --demand_type 4 &> sechina_4.log 
python template_run.py --template_choose 'sechina' --demand_type 5 &> sechina_5.log 