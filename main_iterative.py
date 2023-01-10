from data_process import data_construct
from phase_one_model import PhaseOne
from phase_two_model import PhaseTwo

model_dir = '../模型数据/'
phase_one_dir = 'data/phase_one/'
phase_two_dir = 'data/phase_two/'

model_data = data_construct(model_dir)
model_data.T = model_data.T
phase_one = PhaseOne(model_data, phase_one_dir)
phase_one.build()
phase_one.run()
phase_two = PhaseTwo(model_data, phase_two_dir)
phase_two.build()
phase_two.set_state()
phase_two.run()

