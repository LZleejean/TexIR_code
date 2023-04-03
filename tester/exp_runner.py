import sys
import argparse
import GPUtil
import torch


from tester.test_editing import MatEditingRunner
from tester.test_novel import NovelViewRunner
from tester.test_relighting import RelightingRunner
from tester.test_error import MatErrorRunner

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='')
    parser.add_argument('--exps_folder_name', type=str, default='exps')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--teststage', type=str, default='IRF', help='')
    
    parser.add_argument('--frame_skip', type=int, default=1, help='skip frame when training')
    # parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--max_niter', type=int, default=200001, help='max number of iterations to train for')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, 
                        maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    runder_dict = {
        'Editing': MatEditingRunner,
        'View': NovelViewRunner,
        'Relighting': RelightingRunner,
        'Error': MatErrorRunner
    }

    trainrunner = runder_dict[opt.teststage](conf=opt.conf,
                                            exps_folder_name=opt.exps_folder_name,
                                            expname=opt.expname,
                                            frame_skip=opt.frame_skip,
                                            max_niters=opt.max_niter,
                                            is_continue=True,
                                            timestamp=opt.timestamp,
                                            checkpoint=opt.checkpoint,
                                            gpu_index=gpu
                                            )

    trainrunner.run()