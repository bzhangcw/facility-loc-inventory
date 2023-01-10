from data_process import data_construct
from phase_one_model import PhaseOne
from phase_two_model import PhaseTwo
from utils import show_profiling, DEFAULT_ALG_PARAMS

model_dir = '../模型数据v1/'
phase_one_dir = 'data/phase_one/'
phase_two_dir = 'data/phase_two/'

DEFAULT_ALG_PARAMS.show()

model_data = data_construct(model_dir)
phase_one = PhaseOne(model_data, phase_one_dir)
if DEFAULT_ALG_PARAMS.phase1_resolve:
    phase_one.build()
    phase_one.run()

phase_two = PhaseTwo(model_data, phase_two_dir)
phase_two.load_phasei_from_local(phase_one.solution_dir)
phase_two.build()
phase_two.set_state()
phase_two.run()

show_profiling()
