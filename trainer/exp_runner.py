import sys
sys.path.append("../TexIR_code")
import argparse
import GPUtil
import torch

from trainer.train_irf import IRFTrainRunner
from trainer.train_material import MatTrainRunner
from trainer.train_irrf import IRRFTrainRunner
from trainer.train_pil import PILTrainRunner

from trainer.generate_ir_texture import IrrTextureRunner

from trainer.train_material_invrender import MatInvTrainRunner
from trainer.train_material_neilf import MatNeilfTrainRunner
from trainer.train_material_recMLP import MatRecMLPTrainRunner

from trainer.train_material_syn import MatTrainSynRunner
from trainer.train_material_recMLP_syn import MatRecMLPTrainSynRunner
from trainer.train_material_neilf_syn import MatNeilfTrainSynRunner
from trainer.train_material_invrender_syn import MatInvTrainSynRunner


torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='')
    parser.add_argument('--exps_folder_name', type=str, default='exps')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--trainstage', type=str, default='IRF', help='')
    
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
        'IRF': IRFTrainRunner,
        'Mat': MatTrainRunner,
        'IRRF': IRRFTrainRunner,
        'PIL': PILTrainRunner,
        'Inv': MatInvTrainRunner,
        'Neilf': MatNeilfTrainRunner,
        'IrrT': IrrTextureRunner,
        'RecMLP': MatRecMLPTrainRunner,
        'MatSyn': MatTrainSynRunner,
        'RecMLPSyn': MatRecMLPTrainSynRunner,
        'NeilfSyn': MatNeilfTrainSynRunner,
        'InvSyn': MatInvTrainSynRunner
    }

    trainrunner = runder_dict[opt.trainstage](conf=opt.conf,
                                            exps_folder_name=opt.exps_folder_name,
                                            expname=opt.expname,
                                            frame_skip=opt.frame_skip,
                                            max_niters=opt.max_niter,
                                            is_continue=opt.is_continue,
                                            timestamp=opt.timestamp,
                                            checkpoint=opt.checkpoint,
                                            gpu_index=gpu
                                            )

    trainrunner.run()