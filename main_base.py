import warnings

warnings.filterwarnings("ignore")
import wandb
import os
import sys
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import math

from ops.dataset import TSNDataSet
from ops.models_ada import TSN_Ada
#from ops.models_ada_runtime import TSN_Ada
#from ops.models_amd import TSN_Amd
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map, Recorder

from tensorboardX import SummaryWriter
from ops.my_logger import Logger

from ops.sal_rank_loss import cal_sal_rank_loss

from ops.net_flops_table import get_gflops_params, feat_dim_dict
from ops.amd_net_flops_table import amd_get_gflops_params 
from ops.utils import get_mobv2_new_sd

from os.path import join as ospj

import pdb

def amd_load_to_sd(model_dict, model_path, module_name, fc_name, resolution, apple_to_apple=False):
    if ".pth" in model_path:
        print("done loading\t%s\t(res:%3d) from\t%s" % ("%-25s" % module_name, resolution, model_path))
        sd = torch.load(model_path)['state_dict']
        if "module.block_cnn_dict.base.1.bias" in sd:
            print("Directly upload")
            return sd
        
        if apple_to_apple:
            del_keys = []
            if args.remove_all_base_0:
                for key in sd:
                    if "module.base_model_list.0" in key or "new_fc_list.0" in key or "linear." in key:
                        del_keys.append(key)

            if args.no_weights_from_linear:
                for key in sd:
                    if "linear." in key:
                        del_keys.append(key)

            for key in list(set(del_keys)):
                del sd[key]

            return sd

        replace_dict = []
        nowhere_ks = []
        notfind_ks = []

        for k, v in sd.items():  # TODO(yue) base_model->base_model_list.i
            new_k = k.replace("base_model", module_name)
            new_k = new_k.replace("new_fc", fc_name)
            if new_k in model_dict:
                replace_dict.append((k, new_k))
            else:
                nowhere_ks.append(k)
        for new_k, v in model_dict.items():
            if module_name in new_k:
                k = new_k.replace(module_name, "base_model")
                if k not in sd:
                    notfind_ks.append(k)
            if fc_name in new_k:
                k = new_k.replace(fc_name, "new_fc")
                if k not in sd:
                    notfind_ks.append(k)
        if len(nowhere_ks) != 0:
            print("Vars not in ada network, but are in pretrained weights\n" + ("\n%s NEW  " % module_name).join(
                nowhere_ks))
        if len(notfind_ks) != 0:
            print("Vars not in pretrained weights, but are needed in ada network\n" + ("\n%s LACK " % module_name).join(
                notfind_ks))
        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)

        if "lite_backbone" in module_name:
            # TODO not loading new_fc in this case, because we are using hidden_dim
            if args.frame_independent == False:
                del sd["module.lite_fc.weight"]
                del sd["module.lite_fc.bias"]
        return {k: v for k, v in sd.items() if k in model_dict}
    else:
        print("skip loading\t%s\t(res:%3d) from\t%s" % ("%-25s" % module_name, resolution, model_path))
        return {}
    



def load_to_sd(model_dict, model_path, module_name, fc_name, resolution, apple_to_apple=False):
    if ".pth" in model_path:
        print("done loading\t%s\t(res:%3d) from\t%s" % ("%-25s" % module_name, resolution, model_path))
        sd = torch.load(model_path)['state_dict']
        new_version_detected = False
        for k in sd:
            if "lite_backbone.features.1.conv.4." in k:
                new_version_detected = True
                break
        if new_version_detected:
            sd = get_mobv2_new_sd(sd, reverse=True)

        if apple_to_apple:
            del_keys = []
            if args.remove_all_base_0:
                for key in sd:
                    if "module.base_model_list.0" in key or "new_fc_list.0" in key or "linear." in key:
                        del_keys.append(key)

            if args.no_weights_from_linear:
                for key in sd:
                    if "linear." in key:
                        del_keys.append(key)

            for key in list(set(del_keys)):
                del sd[key]

            return sd

        replace_dict = []
        nowhere_ks = []
        notfind_ks = []

        for k, v in sd.items():  # TODO(yue) base_model->base_model_list.i
            new_k = k.replace("base_model", module_name)
            new_k = new_k.replace("new_fc", fc_name)
            if new_k in model_dict:
                replace_dict.append((k, new_k))
            else:
                nowhere_ks.append(k)
        for new_k, v in model_dict.items():
            if module_name in new_k:
                k = new_k.replace(module_name, "base_model")
                if k not in sd:
                    notfind_ks.append(k)
            if fc_name in new_k:
                k = new_k.replace(fc_name, "new_fc")
                if k not in sd:
                    notfind_ks.append(k)
        if len(nowhere_ks) != 0:
            print("Vars not in ada network, but are in pretrained weights\n" + ("\n%s NEW  " % module_name).join(
                nowhere_ks))
        if len(notfind_ks) != 0:
            print("Vars not in pretrained weights, but are needed in ada network\n" + ("\n%s LACK " % module_name).join(
                notfind_ks))
        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)

        if "lite_backbone" in module_name:
            # TODO not loading new_fc in this case, because we are using hidden_dim
            if args.frame_independent == False:
                del sd["module.lite_fc.weight"]
                del sd["module.lite_fc.bias"]
        return {k: v for k, v in sd.items() if k in model_dict}
    else:
        print("skip loading\t%s\t(res:%3d) from\t%s" % ("%-25s" % module_name, resolution, model_path))
        return {}
    

def main():
    t_start = time.time()
    global args, best_prec1, num_class, use_ada_framework  # , model
    wandb.init(
        project="arnet-reproduce",
        name="AMD "+"aff"+ str(args.accuracy_weight) + " eff"+ str(args.efficency_weight) + "b" + str(args.batch_size) + "-" + str(args.pe_at_rnn),
        entity="video_channel",
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)
    set_random_seed(args.random_seed)
    use_ada_framework = args.ada_reso_skip or args.ada_depth_skip and args.offline_lstm_last == False and args.offline_lstm_all == False and args.real_scsampler == False

    if args.ablation:
        logger = None
    else:
        if not test_mode:
            logger = Logger()
            sys.stdout = logger
        else:
            logger = None

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.data_dir)

    #===
    #args.val_list = args.train_list
    #===

    if args.ada_reso_skip:
        if len(args.ada_crop_list) == 0:
            args.ada_crop_list = [1 for _ in args.reso_list]

    if use_ada_framework:
        if args.ada_reso_skip :
            init_gflops_table()
        elif args.ada_depth_skip:
            amd_init_gflops_table()

    if args.ada_depth_skip:
        if args.runtime:
            from ops.models_amd_runtime import TSN_Amd
        elif args.resolution_list :
            from ops.models_amd_cnn_once_reso import TSN_Amd
        else:
            from ops.models_amd_cnn_once import TSN_Amd

        model = TSN_Amd(num_class, args.num_segments,
                    base_model=args.arch,
                    consensus_type=args.consensus_type,
                    dropout=args.dropout,
                    partial_bn=not args.no_partialbn,
                    pretrain=args.pretrain,
                    fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                    args=args)

    
    else:
        model = TSN_Ada(num_class, args.num_segments,
                        base_model=args.arch,
                        consensus_type=args.consensus_type,
                        dropout=args.dropout,
                        partial_bn=not args.no_partialbn,
                        pretrain=args.pretrain,
                        fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                        args=args)
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    
    # TODO(yue) freeze some params in the policy + lstm layers
    if args.freeze_policy:
        for name, param in model.module.named_parameters():
            if "lite_fc" in name or "lite_backbone" in name or "rnn" in name or "linear" in name:
                param.requires_grad = False

    if args.freeze_backbone:
        for name, param in model.module.named_parameters():
            if "base_model" in name:
                param.requires_grad = False
    if len(args.frozen_list) > 0:
        for name, param in model.module.named_parameters():
            for keyword in args.frozen_list:
                if keyword[0] == "*":
                    if keyword[-1] == "*":  # TODO middle
                        if keyword[1:-1] in name:
                            param.requires_grad = False
                            print(keyword, "->", name, "frozen")
                    else:  # TODO suffix
                        if name.endswith(keyword[1:]):
                            param.requires_grad = False
                            print(keyword, "->", name, "frozen")
                elif keyword[-1] == "*":  # TODO prefix
                    if name.startswith(keyword[:-1]):
                        param.requires_grad = False
                        print(keyword, "->", name, "frozen")
                else:  # TODO exact word
                    if name == keyword:
                        param.requires_grad = False
                        print(keyword, "->", name, "frozen")
        print("=" * 80)
        for name, param in model.module.named_parameters():
            print(param.requires_grad, "\t", name)

        print("=" * 80)
        for name, param in model.module.named_parameters():
            print(param.requires_grad, "\t", name)

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> f/ine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

        
     # TODO(yue) ada_model loading process
    if args.ada_depth_skip:
        if test_mode:
            print("Test mode load from pretrained model AMD")
            the_model_path = args.test_from
            if ".pth.tar" not in the_model_path:
                the_model_path = ospj(the_model_path, "models", "ckpt.best.pth.tar")
            model_dict = model.state_dict()
            sd = amd_load_to_sd(model_dict, the_model_path, "foo", "bar", -1, apple_to_apple=True)
            model_dict.update(sd)
            model.load_state_dict(model_dict)
        elif args.base_pretrained_from != "":
            print("Adaptively load from pretrained whole AMD")
            model_dict = model.state_dict()
            sd = amd_load_to_sd(model_dict, args.base_pretrained_from, "foo", "bar", -1, apple_to_apple=True)

            model_dict.update(sd)
            model.load_state_dict(model_dict)

        elif len(args.model_paths) != 0:
            print("Adaptively load from model_path_list AMD")
            model_dict = model.state_dict()
            # TODO(yue) backbones
            for i, tmp_path in enumerate(args.model_paths):
                base_model_index = i
                new_i = i
                
                sd = amd_load_to_sd(model_dict, tmp_path, "base_model_list.%d" % base_model_index, "new_fc_list.%d" % new_i, 224)
                
                model_dict.update(sd)
            model.load_state_dict(model_dict)
            
    # TODO(yue) ada_model loading process
    elif args.ada_reso_skip:
        if test_mode:
            print("Test mode load from pretrained model")
            the_model_path = args.test_from
            if ".pth.tar" not in the_model_path:
                the_model_path = ospj(the_model_path, "models", "ckpt.best.pth.tar")
            model_dict = model.state_dict()
            sd = load_to_sd(model_dict, the_model_path, "foo", "bar", -1, apple_to_apple=True)
            model_dict.update(sd)
            model.load_state_dict(model_dict)
        elif args.base_pretrained_from != "":
            print("Adaptively load from pretrained whole")
            model_dict = model.state_dict()
            sd = load_to_sd(model_dict, args.base_pretrained_from, "foo", "bar", -1, apple_to_apple=True)

            model_dict.update(sd)
            model.load_state_dict(model_dict, strict=False)

        elif len(args.model_paths) != 0:
            print("Adaptively load from model_path_list")
            model_dict = model.state_dict()
            # TODO(yue) policy net
            sd = load_to_sd(model_dict, args.policy_path, "lite_backbone", "lite_fc",
                            args.reso_list[args.policy_input_offset])
            model_dict.update(sd)
            # TODO(yue) backbones
            prev_path = None
            for i, tmp_path in enumerate(args.model_paths):
                base_model_index = i
                new_i = i
                
#                 if i == 0 :
#                     sd = load_to_sd(model_dict, tmp_path, "base_model_list.%d" % base_model_index, "new_fc_list.%d" % new_i, args.reso_list[i])
                if prev_path == tmp_path:
                    continue
                
                sd = load_to_sd(model_dict, tmp_path, "base_model_list.%d" % base_model_index, "new_fc_list.%d" % new_i, args.reso_list[i])
                prev_path = tmp_path
                model_dict.update(sd)
            model.load_state_dict(model_dict)

    
    else:
        if test_mode:
            the_model_path = args.test_from
            if ".pth.tar" not in the_model_path:
                the_model_path = ospj(the_model_path, "models", "ckpt.best.pth.tar")
            model_dict = model.state_dict()
            sd = load_to_sd(model_dict, the_model_path, "foo", "bar", -1, apple_to_apple=True)
            model_dict.update(sd)
            model.load_state_dict(model_dict)

    if args.ada_reso_skip == False and args.base_pretrained_from != "":
        print("Baseline: load from pretrained model")
        model_dict = model.state_dict()
        sd = load_to_sd(model_dict, args.base_pretrained_from, "base_model", "new_fc", 224)

        if args.ignore_new_fc_weight:
            print("@ IGNORE NEW FC WEIGHT !!!")
            del sd["module.new_fc.weight"]
            del sd["module.new_fc.bias"]

        model_dict.update(sd)
        model.load_state_dict(model_dict)

    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)
    data_length = 1
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       normalize,
                   ]), dense_sample=args.dense_sample,
                   dataset=args.dataset,
                   partial_fcvid_eval=args.partial_fcvid_eval,
                   partial_ratio=args.partial_ratio,
                   ada_reso_skip=args.ada_reso_skip,
                   reso_list=args.reso_list,
                   random_crop=args.random_crop,
                   center_crop=args.center_crop,
                   ada_crop_list=args.ada_crop_list,
                   rescale_to=args.rescale_to,
                   policy_input_offset=args.policy_input_offset,
                   save_meta=args.save_meta),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       normalize,
                   ]), dense_sample=args.dense_sample,
                   dataset=args.dataset,
                   partial_fcvid_eval=args.partial_fcvid_eval,
                   partial_ratio=args.partial_ratio,
                   ada_reso_skip=args.ada_reso_skip,
                   reso_list=args.reso_list,
                   random_crop=args.random_crop,
                   center_crop=args.center_crop,
                   ada_crop_list=args.ada_crop_list,
                   rescale_to=args.rescale_to,
                   policy_input_offset=args.policy_input_offset,
                   save_meta=args.save_meta
                   ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    if not test_mode:
        exp_full_path = setup_log_directory(logger, args.log_dir, args.exp_header)
    else:
        exp_full_path = None
    
    
    if not args.ablation:
        if not test_mode:
            with open(os.path.join(exp_full_path, 'args.txt'), 'w') as f:
                f.write(str(args))
            tf_writer = SummaryWriter(log_dir=exp_full_path)
        else:
            tf_writer = None
    else:
        tf_writer = None
    
    if args.evaluate:
        validate(val_loader, model, criterion, 0, logger, exp_full_path, tf_writer)
        return



    # TODO(yue)
    map_record = Recorder()
    mmap_record = Recorder()
    prec_record = Recorder()
    best_train_usage_str = None
    best_val_usage_str = None

    wandb.watch(model)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if not args.skip_training:
            set_random_seed(args.random_seed + epoch)
            adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
            train_usage_str = train(train_loader, model, criterion, optimizer, epoch, logger, exp_full_path, tf_writer)
        else:
            train_usage_str = "No training usage stats (Eval Mode)"

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            set_random_seed(args.random_seed)
            if args.runtime:
                mAP, mmAP, prec1, val_usage_str, val_gflops = runtime_validate(val_loader, model, criterion, epoch, logger,exp_full_path, tf_writer)
            else:
                mAP, mmAP, prec1, val_usage_str, val_gflops = validate(val_loader, model, criterion, epoch, logger,exp_full_path, tf_writer)
            # remember best prec@1 and save checkpoint
            map_record.update(mAP)
            mmap_record.update(mmAP)
            prec_record.update(prec1)

            if mmap_record.is_current_best():
                best_train_usage_str = train_usage_str
                best_val_usage_str = val_usage_str

            print('Best mAP: %.3f (epoch=%d)\t\tBest mmAP: %.3f(epoch=%d)\t\tBest Prec@1: %.3f (epoch=%d)' % (
                map_record.best_val, map_record.best_at,
                mmap_record.best_val, mmap_record.best_at,
                prec_record.best_val, prec_record.best_at))

            if args.skip_training:
                break

            if (not args.ablation) and (not test_mode):
                tf_writer.add_scalar('acc/test_top1_best', prec_record.best_val, epoch)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': prec_record.best_val,
#                 }, mmap_record.is_current_best(), exp_full_path)
                }, mmap_record.is_current_best(), exp_full_path, epoch+1)
    if use_ada_framework and not test_mode:
        print("Best train usage:")
        print(best_train_usage_str)
        print()
        print("Best val usage:")
        print(best_val_usage_str)

    print("Finished in %.4f seconds\n" % (time.time() - t_start))


def set_random_seed(the_seed):
    if args.random_seed >= 0:
        np.random.seed(the_seed)
        torch.manual_seed(the_seed)
        torch.cuda.manual_seed(the_seed)
        torch.cuda.manual_seed_all(the_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(the_seed)

def amd_init_gflops_table():
    global gflops_table
    gflops_table = {}
    default_gflops_table = {}
    seg_len = -1
    resolution = args.rescale_to
    resol_list = args.resolution_list    
    for resol in resol_list:
        _gflops_table = {}
        _default_gflops_table = {}
        """get gflops of block even it not using"""
        default_block_list = ["base", "conv_2", "conv_3", "conv_4", "conv_5"]
        default_case_list = ["cnn", "rnn"]
        resize = int(resol)
       
        _default_gflops_table[str(args.arch) + "base"] = \
                    amd_get_gflops_params(args.arch, "base", num_class, resolution=resize, case="cnn", seg_len=seg_len)[0]
        _default_gflops_table[str(args.arch) + "base" + "fc"] = \
                    amd_get_gflops_params(args.arch, "base_fc", num_class, resolution=resize, case="cnn", seg_len=seg_len)[0]
        for _block in default_block_list:
            for _case in default_case_list:
                _default_gflops_table[str(args.arch) + _block + _case] = \
                    amd_get_gflops_params(args.arch, _block, num_class, resolution=resize, case=_case, hidden_dim = args.hidden_dim if _case is "rnn" else None, seg_len=seg_len)[0]
        
        print(_default_gflops_table)

        """add gflops of unusing block to using block"""
        start = 0
        for using_block in args.block_rnn_list :
            _gflops_table[str(args.arch) + using_block + "rnn"] = _default_gflops_table[str(args.arch) + using_block + "rnn"]
            _gflops_table[str(args.arch) + using_block + "cnn"] = 0
            index = default_block_list.index(using_block)
            for j in range(start, index+1):
                if j is 0:
                    _gflops_table[str(args.arch) + using_block + "cnn"] = _default_gflops_table[str(args.arch) + "base"]
                else:
                    _gflops_table[str(args.arch) + using_block + "cnn"] += _default_gflops_table[str(args.arch) + default_block_list[j] + "cnn"]
            start = index+1
        
        """get gflops of all pass block"""
        _gflops_table[str(args.arch) + "basefc"] = _default_gflops_table[str(args.arch) + "basefc"] 
        for last_block in range(start, len(default_block_list)):
            name = default_block_list[last_block]
            if name is not "base":
                _gflops_table[str(args.arch) + "basefc"] += _default_gflops_table[str(args.arch) + name + "cnn"] 

            
        print("gflops_table: from base to ")
        for k in _gflops_table:
            print("%-20s: %.4f GFLOPS" % (k, _gflops_table[k]))

        gflops_table[resol] = _gflops_table


def amd_get_gflops_t_tt_vector(resolution):
    gflops_vec = []
    t_vec = []
    tt_vec = []
    
    _gflops_table = gflops_table[str(resolution)]
    if all([arch_name not in args.arch for arch_name in ["resnet", "mobilenet", "efficientnet", "res3d", "csn"]]):
        exit("We can only handle resnet/mobilenet/efficientnet/res3d/csn as backbone, when computing FLOPS")

    for using_block in args.block_rnn_list:
        gflops_lstm = _gflops_table[str(args.arch) + str(using_block) + "rnn"]
        the_flops = _gflops_table[str(args.arch) + str(using_block) + "cnn"] + gflops_lstm
        gflops_vec.append(the_flops)
        t_vec.append(1.)
        tt_vec.append(1.)
    
    the_flops = _gflops_table[str(args.arch) + "basefc"]
    gflops_vec.append(the_flops)
    t_vec.append(1.)
    tt_vec.append(1.)
  
    return gflops_vec, t_vec, tt_vec #ex : (conv_2 skip, conv_3 skip, conv_4 skip, conv_5 skip, all_pass)


def amd_cal_eff(r_, all_policy_r):
    each_losses = []
    # TODO r N * T * (#which block exit, conv2/ conv_3/ conv_4/ conv_5/all)
    # r_loss : pass conv_2/ conv_3/ conv_4/ conv_5/ all
    gflops_vec, t_vec, tt_vec = amd_get_gflops_t_tt_vector(args.rescale_to)
    t_vec = torch.tensor(t_vec).cuda()
    ''' 
    for i in range(1, len(gflops_vec)):
        gflops_vec[i-1] = gflops_vec[i]
    gflops_vec[-1] = 0.1
    '''
    
    for i in range(1, len(gflops_vec)):
        gflops_vec[i] += gflops_vec[i-1]
    total_gflops = gflops_vec[-1]


    for i in range(len(gflops_vec)):
        gflops_vec[i] = total_gflops - gflops_vec[i]
    gflops_vec[-1] += 0.00001
    
    #uni_gflops = np.sum(gflops_vec)/r.shape[2]
    if args.use_gflops_loss:
        r_loss = torch.tensor(gflops_vec).cuda()
     #   r_loss = torch.tensor(np.multiply(gflops_vec,[6,5,4,3,2,1])).cuda()
     #    r_loss = torch.tensor([uni_gflops, uni_gflops, uni_gflops,uni_gflops, uni_gflops, uni_gflops, uni_gflops]).cuda()[:r.shape[2]]
    else:
        r_loss = torch.tensor([4., 2., 1., 0.5, 0.25]).cuda()[:r.shape[2]]
    '''
    b_, t_, c_ = r_.shape
    r_last = (r_[:,:,-1] < 1).float()
    r_last = r_last.unsqueeze(-1).expand(b_, t_, c_)
    r_ = r_last * r_
    '''
    #loss = torch.sum(torch.mean(r_[:,:,:-1], dim=[0, 1]) * r_loss[1:])
    loss = torch.sum(torch.mean(r_, dim=[0, 1]) * r_loss)
    each_losses.append(loss.detach().cpu().item())
    

    # TODO(yue) uniform loss
    if args.uniform_loss_weight > 1e-5:
#         if_policy_backbone = 1 if args.policy_also_backbone else 0
#         num_pred = len(args.backbone_list)
        policy_dim = len(args.block_rnn_list)
        reso_skip_vec = torch.zeros(policy_dim).cuda()

        if args.skip_twice:
            uniform_loss = 0
            check_passed_block_bool = all_policy_r[:,:,:,-1] > 0.5
            check_passed_block = torch.tensor(check_passed_block_bool, dtype=torch.float).cuda()
            passed_block = check_passed_block.unsqueeze(-1).expand(-1,-1,-1,3) * all_policy_r  #all_policy_r : B, T, K, A (A = skip/ skip_twice/ pass)
            
            for k_i in range(all_policy_r.shape[-2]):
                reso_skip_vec = torch.zeros(policy_dim).cuda()
                for c_i in range(all_policy_r.shape[-1]):
                    reso_skip_vec[c_i] = torch.sum(passed_block[:,:,k_i,c_i])
                    
                reso_skip_vec = reso_skip_vec / torch.sum(reso_skip_vec).clamp(min=1e-6)
                
                if args.uniform_cross_entropy:  # TODO cross-entropy+ logN
                    uniform_loss = torch.sum(
                        torch.tensor([x * torch.log(torch.clamp_min(x, 1e-6)) for x in reso_skip_vec])) + torch.log(
                        torch.tensor(1.0 * len(reso_skip_vec)))
                    uniform_loss += uniform_loss * args.uniform_loss_weight
                else:  # TODO L2 norm
                    usage_bias = reso_skip_vec - torch.mean(reso_skip_vec)
                    uniform_loss += torch.norm(usage_bias, p=2) * args.uniform_loss_weight                

        else:
            # TODO
            offset = 0
            for b_i in range(policy_dim):
                reso_skip_vec[b_i] = torch.sum(r[:, :, b_i]) - torch.sum(r[:, :, b_i+1])

            reso_skip_vec = reso_skip_vec / torch.sum(reso_skip_vec)
            reso_skip_vec = 1 - reso_skip_vec #?
            if args.uniform_cross_entropy:  # TODO cross-entropy+ logN
                uniform_loss = torch.sum(
                    torch.tensor([x * torch.log(torch.clamp_min(x, 1e-6)) for x in reso_skip_vec])) + torch.log(
                    torch.tensor(1.0 * len(reso_skip_vec)))
                uniform_loss = uniform_loss * args.uniform_loss_weight
            else:  # TODO L2 norm
                usage_bias = reso_skip_vec - torch.mean(reso_skip_vec)
                uniform_loss = torch.norm(usage_bias, p=2) * args.uniform_loss_weight
                
        loss = loss + uniform_loss
        each_losses.append(uniform_loss.detach().cpu().item())

    # TODO(yue) high-reso punish loss
    if args.head_loss_weight > 1e-5:
        head_usage = torch.mean(r[:, :, 0])
        usage_threshold = 0.2
        head_loss = (head_usage - usage_threshold) * (head_usage - usage_threshold) * args.head_loss_weight
        loss = loss + head_loss
        each_losses.append(head_loss.detach().cpu().item())

    # TODO(yue) frames loss
    if args.frames_loss_weight > 1e-5:
        num_frames = torch.mean(torch.mean(r, dim=[0, 1]) * t_vec)
        frames_loss = num_frames * num_frames * args.frames_loss_weight
        loss = loss + frames_loss
        each_losses.append(frames_loss.detach().cpu().item())

    return loss, each_losses    

def amd_cal_kld(output, r, base_outs):
    class diff_tanh_KLD(torch.autograd.Function):
            
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            tanh = torch.nn.Tanh()
            return 1 - (1-tanh(input))*(1+tanh(input)).clamp(min=1e-6)

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            tanh = torch.nn.Tanh()
            return (grad_output.clone()*(2)*tanh(input)*(1-tanh(input))*(1+tanh(input))).clamp(min=1e-6)
    
    cossim = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
#     pred_frame_outs = torch.tensor(base_outs) * torch.mean(r[:,:,,1:],dim=[2]).unsqueeze(-1)
    r_mean = torch.mean(r[:,:,1:], dim=[2]).unsqueeze(-1) 
#     KLD_criterion = torch.nn.KLDivLoss(reduction='batchmean')
#     DTK_criterion = diff_tanh_KLD().apply
#     kld_loss = KLD_criterion(pred_frame_outs,torch.tensor(output).unsqueeze(1).expand(-1, 16, -1))
#     diff_tanh_kld_loss = DTK_criterion(kld_loss)

    relu = torch.nn.ReLU()
    cossim_outs = cossim(torch.tensor(output).unsqueeze(1).expand(-1, 16, -1), base_outs).unsqueeze(-1)
#     sim_base_outs = torch.sum(r_mean * relu(cossim_outs) * base_outs, dim=[1]) #-1~1 -> 0~1 
    sim_base_outs = torch.sum(r_mean * (cossim_outs+1)/2 * base_outs, dim=[1]) #-1~1 -> 0~1 

#     take_bool = r_mean > 0
#     t_tensor = torch.sum(torch.tensor(take_bool, dtype=torch.float).cuda(), dim=[1])
#     total_tensor = torch.sum(r_mean * relu(cossim_outs), dim=[1]).clamp(min=1e-6)
    total_tensor = torch.sum(r_mean * (cossim_outs+1)/2, dim=[1]).clamp(min=1e-6)

    return sim_base_outs/total_tensor
       
        
        

def init_gflops_table():
    global gflops_table
    gflops_table = {}
    params_table = {}
    seg_len = -1
    
    for i, backbone in enumerate(args.backbone_list):
        gflops_table[backbone + str(args.reso_list[i])] = \
            get_gflops_params(backbone, args.reso_list[i], num_class, seg_len)[0]
        params_table[backbone + str(args.reso_list[i])] = \
            get_gflops_params(backbone, args.reso_list[i], num_class, seg_len)[1]
    gflops_table["policy"] = \
        get_gflops_params(args.policy_backbone, args.reso_list[args.policy_input_offset], num_class, seg_len)[0]
    params_table["policy"] = \
        get_gflops_params(args.policy_backbone, args.reso_list[args.policy_input_offset], num_class, seg_len)[1]
    gflops_table["lstm"] = 2 * (feat_dim_dict[args.policy_backbone] ** 2) / 1000000000

    print("gflops_table: ")
    for k in gflops_table:
        print("%-20s: %.4f GFLOPS" % (k, gflops_table[k]))
        
    print("params_table: ")
    for k in params_table:
        print("%-20s: %.4f params" % (k, params_table[k]))


def get_gflops_t_tt_vector():
    gflops_vec = []
    t_vec = []
    tt_vec = []

    for i, backbone in enumerate(args.backbone_list):
        if all([arch_name not in backbone for arch_name in ["resnet", "mobilenet", "efficientnet", "res3d", "csn"]]):
            exit("We can only handle resnet/mobilenet/efficientnet/res3d/csn as backbone, when computing FLOPS")

        for crop_i in range(args.ada_crop_list[i]):
            the_flops = gflops_table[backbone + str(args.reso_list[i])]
            gflops_vec.append(the_flops)
            t_vec.append(1.)
            tt_vec.append(1.)

    if args.policy_also_backbone:
        gflops_vec.append(0)
        t_vec.append(1.)
        tt_vec.append(1.)

    for i, _ in enumerate(args.skip_list):
        t_vec.append(1. if args.skip_list[i] == 1 else 1. / args.skip_list[i])
        tt_vec.append(0)
        gflops_vec.append(0)

    return gflops_vec, t_vec, tt_vec


def cal_eff(r):
    each_losses = []
    # TODO r N * T * (#reso+#policy+#skips)
    gflops_vec, t_vec, tt_vec = get_gflops_t_tt_vector()
    t_vec = torch.tensor(t_vec).cuda()
    if args.use_gflops_loss:
        r_loss = torch.tensor(gflops_vec).cuda()
    else:
        r_loss = torch.tensor([4., 2., 1., 0.5, 0.25, 0.125, 0.0625, 0.03125]).cuda()[:r.shape[2]]
    
    loss = torch.sum(torch.mean(r, dim=[0, 1]) * r_loss)
    each_losses.append(loss.detach().cpu().item())

    # TODO(yue) uniform loss
    if args.uniform_loss_weight > 1e-5:
        if_policy_backbone = 1 if args.policy_also_backbone else 0
        num_pred = len(args.backbone_list)
        policy_dim = num_pred + if_policy_backbone + len(args.skip_list)

        reso_skip_vec = torch.zeros(policy_dim).cuda()

        # TODO
        offset = 0
        # TODO reso/ada_crops
        for b_i in range(num_pred):
            interval = args.ada_crop_list[b_i]
            reso_skip_vec[b_i] += torch.sum(r[:, :, offset:offset + interval])
            offset = offset + interval

        # TODO mobilenet + skips
        for b_i in range(num_pred, reso_skip_vec.shape[0]):
            reso_skip_vec[b_i] = torch.sum(r[:, :, b_i])

        reso_skip_vec = reso_skip_vec / torch.sum(reso_skip_vec)
        if args.uniform_cross_entropy:  # TODO cross-entropy+ logN
            uniform_loss = torch.sum(
                torch.tensor([x * torch.log(torch.clamp_min(x, 1e-6)) for x in reso_skip_vec])) + torch.log(
                torch.tensor(1.0 * len(reso_skip_vec)))
            uniform_loss = uniform_loss * args.uniform_loss_weight
        else:  # TODO L2 norm
            usage_bias = reso_skip_vec - torch.mean(reso_skip_vec)
            uniform_loss = torch.norm(usage_bias, p=2) * args.uniform_loss_weight
        loss = loss + uniform_loss
        each_losses.append(uniform_loss.detach().cpu().item())

    # TODO(yue) high-reso punish loss
    if args.head_loss_weight > 1e-5:
        head_usage = torch.mean(r[:, :, 0])
        usage_threshold = 0.2
        head_loss = (head_usage - usage_threshold) * (head_usage - usage_threshold) * args.head_loss_weight
        loss = loss + head_loss
        each_losses.append(head_loss.detach().cpu().item())

    # TODO(yue) frames loss
    if args.frames_loss_weight > 1e-5:
        num_frames = torch.mean(torch.mean(r, dim=[0, 1]) * t_vec)
        frames_loss = num_frames * num_frames * args.frames_loss_weight
        loss = loss + frames_loss
        each_losses.append(frames_loss.detach().cpu().item())

    return loss, each_losses


def reverse_onehot(a):
    try:
        if args.ada_depth_skip:
            return np.array(a.sum(axis=1), np.int32)
        else:
            return np.array([np.where(r > 0.5)[0][0] for r in a])
    except Exception as e:
        print("error stack:", e)
        print(a)
        for i, r in enumerate(a):
            print(i, r)
        return None

def confidence_criterion_loss(criterion, all_policy_r, feat_outs, target):
    # all_policy_r B,T,K-1,A
    # feat_outs B,T,(K-1)+1,#class
    policy_gt_loss = 0
    inner_acc_loss = 0
    _feat_outs = F.softmax(feat_outs, dim=-1)
    _target = target[:,0]
    total_cnt = 0.0
    total_acc_cnt = 0.0
    
    batch_size  = feat_outs.shape[0]
    time_length = feat_outs.shape[1]
    layer_cnt   = feat_outs.shape[2]
    
    for b_i in range(feat_outs.shape[0]):
        conf_outs = _feat_outs[b_i,:,:,_target[b_i]]
        diff_conf_l = []
        for k_i in range(1, layer_cnt):
            diff_conf_l.append(conf_outs[:,k_i] - conf_outs[:,k_i-1])
        
        target_pass_bool = torch.stack(diff_conf_l, dim=1) > 0  #T,K-1
        target_policy = torch.tensor(target_pass_bool, dtype=torch.long).cuda()
        
        for k_i in range(layer_cnt-1):
            total_cnt+=1.0
            policy_gt_loss += criterion(all_policy_r[b_i,:,k_i,:], target_policy[:,k_i])
    
    for t_i in range(time_length):
        for k_i in range(layer_cnt-1):
            total_acc_cnt +=1.0
            inner_acc_loss += criterion(feat_outs[:,t_i,k_i,:], _target)

    
    return policy_gt_loss/total_cnt, inner_acc_loss/total_acc_cnt

def indep_confidence_criterion_loss(criterion, all_policy_r, feat_outs, target):
    # all_policy_r B,T,K-1,A
    # feat_outs B,T,(K-1)+1,#class
    policy_gt_loss = 0
    inner_acc_loss = 0
    _target = target[:,0]
    total_cnt = 0.0
    total_acc_cnt = 0.0
    
    batch_size  = feat_outs.shape[0]
    time_length = feat_outs.shape[1]
    layer_cnt   = feat_outs.shape[2]
    
    inner_feat_outs = feat_outs[:, :, :-1, :]
    for k_i in range(layer_cnt-1):
        r_tensor = all_policy_r[:,:,k_i,-1].unsqueeze(-1)
        t_tensor = torch.sum(all_policy_r[:,:,k_i,-1], dim=[1]).unsqueeze(-1).clamp(1)
        inner_feat_out = (inner_feat_outs[:,:,k_i,:] * r_tensor).sum(dim=[1])/t_tensor
        inner_acc_loss += criterion(inner_feat_out, _target)
        
        
    
    return inner_acc_loss/(layer_cnt-1), inner_acc_loss/(layer_cnt-1)

def confidence_criterion_loss_selected(criterion, all_policy_r, feat_outs, target):
    # all_policy_r B,T,K-1,A
    # feat_outs B,T,(K-1)+1,#class
    policy_gt_loss = torch.tensor(0.0).cuda()
    inner_acc_loss = torch.tensor(0.0).cuda()
    _feat_outs = F.softmax(feat_outs, dim=-1)
    _target = target[:,0]
    total_cnt = 0.0
    total_acc_cnt = 0.0
    
    selected = (torch.sum(all_policy_r[:,:,:,-1], dim=[2]) == all_policy_r.shape[2]) #B,T
    
    batch_size  = feat_outs.shape[0]
    time_length = feat_outs.shape[1]
    layer_cnt   = feat_outs.shape[2]
    
    for b_i in range(feat_outs.shape[0]):
        conf_outs = _feat_outs[b_i,:,:,_target[b_i]]
        diff_conf_l = []
        for k_i in range(1, layer_cnt):
            diff_conf_l.append(conf_outs[:,k_i] - conf_outs[:,k_i-1])
        
        target_pass_bool = torch.stack(diff_conf_l, dim=1) > 0  #T,K-1
        target_policy = torch.tensor(target_pass_bool, dtype=torch.long).cuda()
        
        for t_i in range(time_length):
            for k_i in range(layer_cnt-1):
                total_cnt+=1.0
                policy_gt_loss += criterion(all_policy_r[b_i,t_i,k_i,:].unsqueeze(0), target_policy[t_i,k_i].unsqueeze(0))
                if selected[b_i, t_i]:
                    total_acc_cnt+=1.0
                    inner_acc_loss += criterion(feat_outs[b_i,t_i,k_i,:].unsqueeze(0), _target[b_i].unsqueeze(0))
    
    
    return policy_gt_loss/max(total_cnt, 1.0), inner_acc_loss/max(total_acc_cnt, 1.0)

def guide_criterion_loss_selected(criterion, all_policy_r, feat_outs, target, output, epoch):
    # all_policy_r B,T,K-1,A
    # feat_outs B,T,(K-1)+1,#class
    policy_gt_loss = 0
    inner_acc_loss = 0
    _target = target[:,0]
    _feat_outs = F.softmax(feat_outs[:,:,-1,:], dim=-1)

    total_cnt = 0.0
    total_acc_cnt = 0.0
    
    batch_size  = feat_outs.shape[0]
    time_length = feat_outs.shape[1]
    layer_cnt   = feat_outs.shape[2]
    
    ###
    selected = (torch.sum(all_policy_r[:,:,:,-1], dim=[2]) == all_policy_r.shape[2]) #B,T

    
    ###
    exp_decay_factor = np.log(1.0/0.9)/float(60)
    output_val, output_class = torch.max(F.softmax(output, 1),dim=1)#B
    
    ###
    conf_bool_l = []
    for b_i in range(batch_size):
        conf_bool = _feat_outs[b_i,:,output_class[b_i]] > (0.9 * np.exp(exp_decay_factor * epoch))
        conf_t = torch.tensor(conf_bool, dtype=torch.long) #T
        conf_bool_l.append(conf_t)
    
    conf_l_t = torch.stack(conf_bool_l, dim=0).cuda() #B,T
    

    t_f_bool = output_class == _target
    t_f_t = torch.tensor(t_f_bool, dtype=torch.long).cuda() #B, 1
    
    ###
    target_policy = conf_l_t * t_f_t.unsqueeze(-1).repeat(1, time_length) #B, T
    for b_i in range(batch_size):
        for t_i in range(time_length):
            total_cnt += 1.0
            policy_gt_loss += criterion(all_policy_r[b_i, t_i, :, :], target_policy[b_i,t_i].unsqueeze(-1).repeat(layer_cnt-1))

    
    return policy_gt_loss/max(total_cnt, 1.0), policy_gt_loss/max(total_cnt, 1.0)

def early_stop_criterion_loss(criterion, all_policy_r, early_stop_r, feat_outs, target):
    # early_stop_r B, T, 2
    # feat_out B,T,(K-1)+1, #class
    batch_size  = feat_outs.shape[0]
    time_length = feat_outs.shape[1]
    layer_cnt   = feat_outs.shape[2]
    
    threshold = 0.25
    selected_r = all_policy_r[:,:,-1,-1].unsqueeze(-1) # B,T, 1
    selected_r_bool = selected_r > 0.5
    
    take_selected = torch.tensor(selected_r_bool, dtype=torch.long).cuda()
    selected_feat_outs = selected_r * feat_outs[:,:,-1,:] #using last prediction  B, T, #class
    early_stop_gt_loss = 0
    
    answer_sheet = None
    key_max = time_length
    for b_i in range(batch_size):
        compare_pred_dict = {}
        target_var = target[b_i, 0]
        pred = None
        stop_flag_cnt = 2
        for t_i in range(time_length):
            if take_selected[b_i, t_i, 0] == 1:
                compare_pred_dict[t_i] =  F.softmax(torch.sum(selected_feat_outs[b_i,:(t_i+1),:], dim=[0]), dim=-1) # #class
                if compare_pred_dict[t_i][target_var] > 0.99:
                    stop_flag_cnt -= 1
                    if stop_flag_cnt == 0:
#                         print("early_stop_g.t._activate")
                        key_max = t_i
                        break
                
        if key_max is time_length:
            key_max = time_length-1
            answer_sheet = torch.ones(time_length,  dtype=torch.long).cuda()
        else:
#             key_max = max(compare_pred_dict.keys(), key=(lambda k: compare_pred_dict[k][target_var]))
            answer_sheet = torch.cat((torch.ones(key_max,  dtype=torch.long), torch.zeros(1,  dtype=torch.long)), dim=0).cuda()
        
        early_stop_gt_loss += criterion(early_stop_r[b_i, :(key_max+1), :], answer_sheet)/(key_max + 1)

    return early_stop_gt_loss / batch_size
        
def dual_policy_criterion_loss(criterion, base_outs, target, dual_policy_r, similarity_r):
    #dual_policy_r B, T, K, 2', 2 
    #similarity_r  B, T, K, 1
    #base_outs     B, T, C
    #target        B, 3
    
    batch_size, _t, policy_cnt, _ = similarity_r.shape
    
    target_val = target[:, 0].unsqueeze(-1) #B, 1
    output_val = base_outs.max(dim=2)[1]    #B, T(argmax)
    
    correct = target_val == output_val 
    correct = torch.tensor(correct, dtype=torch.long).cuda() # B, T
    
    redundant_policy_r = dual_policy_r[:,:,:,0,:] #B, T, K, 2
    noisy_policy_r = dual_policy_r[:,:,:,1,:]     #B, T, K, 2
    
    
    #noisy_policy pseudo-G.T.
    # False-> skip(0)
    # True -> pass(1)
    noisy_gt = correct.unsqueeze(-1).expand(-1,-1,policy_cnt) #B, T, K
    noisy_p_loss = criterion(noisy_policy_r.contiguous().view(batch_size*_t*policy_cnt, -1), noisy_gt.contiguous().view(batch_size*_t*policy_cnt))
    
    #redundant_policy pseudo-G.T.
    # False-> skip(0)
    # True & False(similarity > thr) -> skip(0)
    # True & True(similarity < thr) -> pass(1)
    sim_thr = 0.8
    redundant_gt = similarity_r < sim_thr 
    redundant_gt = torch.tensor(redundant_gt.squeeze(-1), dtype=torch.long).cuda() #B, T, K, 1 -> B, T, K
    
    redundant_gt = redundant_gt * noisy_gt #B, T, K
    redundant_p_loss = criterion(redundant_policy_r.contiguous().view(batch_size*_t*policy_cnt, -1), redundant_gt.contiguous().view(batch_size*_t*policy_cnt))
    
    return redundant_p_loss, noisy_p_loss

def early_exit_criterion_loss(criterion, exit_r_t, r, feat_outs, target):
    #exit_r_t : B, T, K
    #r : B, T, K 
    #feat_out : B,T,(K-1)+1, #class
    BCE = torch.nn.BCELoss().cuda()
    
    selected = r[:,:,-1]#B,T
        
    batch_size = r.shape[0]
    time_steps = r.shape[1]
    psuedo_gt_list = []
    
    total_cnt = 0
    total_exit_loss = 0.0
    for b_i in range(batch_size):
        local_avg_list = []
        local_output = torch.zeros(feat_outs.shape[-1], dtype=torch.float).cuda()
        local_selected_cnt = 0.0
        psuedo_gt = 0
        for t_i in range(time_steps):
            if selected[b_i,t_i] > 0.5:
                local_selected_cnt +=1
                total_cnt +=1
                local_output += feat_outs[b_i,t_i,-1,:]
             
 #             if local_selected_cnt == 0:
#                 loss = criterion(local_output.unsqueeze(0), target[b_i, 0].unsqueeze(0))
#             else :
                loss = criterion((local_output/local_selected_cnt).unsqueeze(0), target[b_i, 0].unsqueeze(0))
            
#             pdb.set_trace()
                loss_bool = loss < 1e-1
                loss_bool_l_t = torch.tensor(loss_bool, dtype=torch.float).cuda()
                total_exit_loss += BCE(exit_r_t[b_i,t_i], loss_bool_l_t.unsqueeze(-1).repeat(1,1,r.shape[2]-1))        
#                 psuedo_gt_list.append(loss_bool_l_t)
                
#         psuedo_gt_l_t = torch.stack(psuedo_gt_list, dim=0).cuda() #T
#         psuedo_gt_l_t = psuedo_gt_l_t.unsqueeze(-1)
#         total_exit_loss = BCE(exit_r_t[b_i,:,-1], psuedo_gt_l_t)
        


    
#     psuedo_gt_l_t = torch.stack(psuedo_gt_list, dim=0).reshape(batch_size, time_steps).cuda() #B, T
#     psuedo_gt_l_t = psuedo_gt_l_t.unsqueeze(-1).repeat(1,1,r.shape[2]-1) #B,T,K
#     psuedo_gt_l_t = psuedo_gt_l_t.unsqueeze(-1) #B,T,1

#     total_exit_loss = BCE(exit_r_t[:,:,-1], psuedo_gt_l_t)
    
#     return total_exit_loss
    return total_exit_loss/total_cnt

def early_exit_future_criterion_loss(criterion, exit_r_t, r, feat_outs, target):
    #exit_r_t : B, T, K
    #r : B, T, K 
    #feat_out : B,T,(K-1)+1, #class
    BCE = torch.nn.BCELoss().cuda()
    
    selected = r[:,:,-1]#B,T
        
    batch_size = r.shape[0]
    time_steps = r.shape[1]
    psuedo_gt_list = []
    
    total_cnt = 0
    total_exit_loss = 0.0
    for b_i in range(batch_size):
        local_avg_list = []
        local_output = torch.zeros(feat_outs.shape[-1], dtype=torch.float).cuda()
        local_selected_cnt = 0.0
        psuedo_gt = 0
        total_target_class = torch.argmax(F.softmax((selected[b_i].unsqueeze(-1) * feat_outs[b_i, :, -1, :]).sum(dim=0) /max(selected[b_i].sum(), 1)))
        
        for t_i in range(time_steps):
            if selected[b_i,t_i] > 0.5:
                local_selected_cnt +=1
                total_cnt +=1
                local_output += feat_outs[b_i,t_i,-1,:]
             
                target_class = torch.argmax(F.softmax(local_output/local_selected_cnt))
                
                loss_bool = target_class == total_target_class
                loss_bool_l_t = torch.tensor(loss_bool, dtype=torch.float).cuda()
                total_exit_loss += BCE(exit_r_t[b_i,t_i], loss_bool_l_t.unsqueeze(-1).repeat(1,1,r.shape[2]-1))
                

    return total_exit_loss/total_cnt

                
    


def get_criterion_loss(criterion, output, target):
    return criterion(output, target[:, 0])


def kl_categorical(p_logit, q_logit):
    import torch.nn.functional as F
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                         - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)


def compute_acc_eff_loss_with_weights(acc_loss, eff_loss, each_losses, epoch):
    if epoch > args.eff_loss_after:
        acc_weight = args.accuracy_weight
        eff_weight = args.efficency_weight
    else:
        acc_weight = 1.0
        eff_weight = 0.0
    return acc_loss * acc_weight, eff_loss * eff_weight, [x * eff_weight for x in each_losses]


def compute_every_losses(r, all_policy_r, acc_loss, epoch):
    if args.ada_depth_skip :
        eff_loss, each_losses = amd_cal_eff(r, all_policy_r)
        
        
    else:
        eff_loss, each_losses = cal_eff(r)
    acc_loss, eff_loss, each_losses = compute_acc_eff_loss_with_weights(acc_loss, eff_loss, each_losses, epoch)
    return acc_loss, eff_loss, each_losses


def elastic_list_print(l, limit=8):
    if isinstance(l, str):
        return l

    limit = min(limit, len(l))
    l_output = "[%s," % (",".join([str(x) for x in l[:limit // 2]]))
    if l.shape[0] > limit:
        l_output += "..."
    l_output += "%s]" % (",".join([str(x) for x in l[-limit // 2:]]))
    return l_output


def compute_exp_decay_tau(epoch):
    return args.init_tau * np.exp(args.exp_decay_factor * epoch)




def amd_get_policy_usage_str(r_list, skip_twice_r_list, act_dim, reso_r_list):
    gflops_vec_l = []
    for reso in args.resolution_list:
        gflops_vec, t_vec, tt_vec = amd_get_gflops_t_tt_vector(reso)
        gflops_vec_l.append(gflops_vec)
    gflops_vec = gflops_vec_l[args.resolution_list.index(str(args.rescale_to))]
    printed_str = ""
    rs = np.concatenate(r_list, axis=0)
    if skip_twice_r_list :
        st_rs = np.concatenate(skip_twice_r_list, axis=0)
        st_tmp_cnt = [np.sum(st_rs[:, :, iii] == 1) for iii in range(st_rs.shape[2])] 
        prev_st_cnt = st_tmp_cnt[0]
    if reso_r_list is None:
        tmp_cnt = [np.sum(rs[:, :, iii] == 1) for iii in range(rs.shape[2])] #[#all #conv_2 #conv_3 #conv_4 #conv_5]
        tmp_total_cnt = rs.shape[0] * rs.shape[1]
    else:
        tmp_cnt = [np.sum(rs[:, :, iii] == 1) for iii in range(rs.shape[2])] #[#all #conv_2 #conv_3 #conv_4 #conv_5]
        tmp_total_cnt = rs.shape[0] * rs.shape[1]
        reso_rs = np.concatenate(reso_r_list, axis=0) #Test * T * K'
        
        reso_tmp_cnt_l = []
        for reso_i in range(reso_rs.shape[2]):
            gflops_vec = gflops_vec_l[reso_i]
            _rs = np.array([rs[..., i] * reso_rs[...,reso_i] for i in range(rs.shape[2])])
            reso_tmp_cnt = [np.sum(_rs[iii, :, :] == 1) for iii in range(_rs.shape[0])]
            reso_tmp_cnt_l.append(reso_tmp_cnt)
    gflops = 0
    avg_frame_ratio = 0
    avg_pred_ratio = 0

    used_model_list = []
    reso_list = []
    if reso_r_list:
        reso_dim = reso_rs.shape[2]
#     for i in range(len(args.backbone_list)):
#         used_model_list += [args.backbone_list[i]] * args.ada_crop_list[i]
#         reso_list += [args.reso_list[i]] * args.ada_crop_list[i]
    prev_pass_cnt = tmp_total_cnt
    if args.use_early_stop:
        printed_str += "total %d\n" % (tmp_total_cnt)
    for action_i in range(rs.shape[2]):
        if action_i is 0:
            action_str = "pass%d (base) " % (action_i)
        else:
            action_str = "pass%d (%s)" % (action_i, args.block_rnn_list[action_i-1])

        usage_ratio = tmp_cnt[action_i] / tmp_total_cnt
        printed_str += "%-22s: %6d (%.2f%%)" % (action_str, tmp_cnt[action_i], 100 * usage_ratio)
        if reso_r_list:
            for reso_i in range(reso_dim):
                printed_str += "/%5d" % (reso_tmp_cnt_l[reso_i][action_i])
        
        if skip_twice_r_list:
            skip_twice_in_stage = (st_tmp_cnt[action_i] - prev_st_cnt) / (prev_pass_cnt - tmp_cnt[action_i])
            printed_str += "| %6d (%.2f%%)" % ((st_tmp_cnt[action_i] - prev_st_cnt),  100 * skip_twice_in_stage)
            prev_pass_cnt = tmp_cnt[action_i]
            prev_st_cnt = st_tmp_cnt[action_i]
        printed_str += "\n"
        
        if reso_r_list is None:
            gflops += usage_ratio * gflops_vec[action_i]
        else: 
            for reso_i in range(reso_dim):
                reso_usage_ratio = reso_tmp_cnt_l[reso_i][action_i] / tmp_total_cnt
                gflops += reso_usage_ratio * gflops_vec_l[reso_i][action_i]

    avg_frame_ratio = usage_ratio * t_vec[-1]

    num_clips = args.num_segments
    printed_str += "GFLOPS: %.6f  AVG_FRAMES: %.3f " % (gflops, avg_frame_ratio * num_clips)
    if skip_twice_r_list:
        printed_str += "skip_twice : %6d total_skip : %6d" % (st_tmp_cnt[st_rs.shape[2]-1], tmp_total_cnt - tmp_cnt[rs.shape[2]-1]) 
  
    return printed_str, gflops

def get_policy_usage_str(r_list, reso_dim):
    gflops_vec, t_vec, tt_vec = get_gflops_t_tt_vector()
    printed_str = ""
    rs = np.concatenate(r_list, axis=0)

    tmp_cnt = [np.sum(rs[:, :, iii] == 1) for iii in range(rs.shape[2])]

    if args.all_policy:
        tmp_total_cnt = tmp_cnt[0]
    else:
        tmp_total_cnt = sum(tmp_cnt)

    gflops = 0
    avg_frame_ratio = 0
    avg_pred_ratio = 0

    used_model_list = []
    reso_list = []

    for i in range(len(args.backbone_list)):
        used_model_list += [args.backbone_list[i]] * args.ada_crop_list[i]
        reso_list += [args.reso_list[i]] * args.ada_crop_list[i]

    for action_i in range(rs.shape[2]):
        if args.policy_also_backbone and action_i == reso_dim - 1:
            action_str = "m0(%s %dx%d)" % (
                args.policy_backbone, args.reso_list[args.policy_input_offset],
                args.reso_list[args.policy_input_offset])
        elif action_i < reso_dim:
            action_str = "r%d(%7s %dx%d)" % (
                action_i, used_model_list[action_i], reso_list[action_i], reso_list[action_i])
        else:
            action_str = "s%d (skip %d frames)" % (action_i - reso_dim, args.skip_list[action_i - reso_dim])

        usage_ratio = tmp_cnt[action_i] / tmp_total_cnt
        printed_str += "%-22s: %6d (%.2f%%)\n" % (action_str, tmp_cnt[action_i], 100 * usage_ratio)

        gflops += usage_ratio * gflops_vec[action_i]
        avg_frame_ratio += usage_ratio * t_vec[action_i]
        avg_pred_ratio += usage_ratio * tt_vec[action_i]

    num_clips = args.num_segments
    gflops += (gflops_table["policy"] + gflops_table["lstm"]) * avg_frame_ratio
    printed_str += "GFLOPS: %.6f  AVG_FRAMES: %.3f  NUM_PREDS: %.3f" % (
        gflops, avg_frame_ratio * args.num_segments, avg_pred_ratio * num_clips)
    return printed_str, gflops


def extra_each_loss_str(each_terms):
    loss_str_list = ["gf"]
    s = ""
    if args.uniform_loss_weight > 1e-5:
        loss_str_list.append("u")
    if args.head_loss_weight > 1e-5:
        loss_str_list.append("h")
    if args.frames_loss_weight > 1e-5:
        loss_str_list.append("f")
    for i in range(len(loss_str_list)):
        s += " %s:(%.4f)" % (loss_str_list[i], each_terms[i].avg)
    return s


def get_current_temperature(num_epoch):
    if args.exp_decay:
        tau = compute_exp_decay_tau(num_epoch)
    else:
        tau = args.init_tau
    return tau


def get_average_meters(number):
    return [AverageMeter() for _ in range(number)]

def update_weights(epoch, acc, eff):
    if args.use_weight_decay:
#         exp_decay_factor = np.log(0.8/float(acc))/float(args.epochs)
#         acc = acc * np.exp(exp_decay_factor * epoch)

        exp_decay_factor = np.log(float(acc)/0.8)/float(args.epochs)
        acc = 0.8 * np.exp(exp_decay_factor * epoch)
        eff = 1 - acc
    return acc, eff

    


def train(train_loader, model, criterion, optimizer, epoch, logger, exp_full_path, tf_writer):
    batch_time, data_time, losses, top1, top5 = get_average_meters(5)
    tau = 0
    if use_ada_framework:
        tau = get_current_temperature(epoch)
        if args.use_early_stop:
            if args.use_conf_btw_blocks:
                alosses, elosses, inner_alosses, policy_gt_losses, es_gt_losses = get_average_meters(5)
            else:
                alosses, elosses, kld_losses, es_gt_losses = get_average_meters(4)
        else:        
            if args.use_conf_btw_blocks:
                alosses, elosses, inner_alosses, policy_gt_losses, early_exit_losses = get_average_meters(5)
            elif args.use_local_policy_module:
                alosses, elosses, redundant_policy_losses, noisy_policy_losses = get_average_meters(4)
            else: 
                alosses, elosses, kld_losses = get_average_meters(3)
                      
        
                      
        each_terms = get_average_meters(NUM_LOSSES)
        r_list = []
        reso_r_list = []
        kld_loss = 0
        if args.skip_twice:
            skip_twice_r_list = []

    meta_offset = -2 if args.save_meta else 0

    model.module.partialBN(not args.no_partialbn)
    # switch to train mode
    model.train()
    
#    model.eval()
#    model.module.early_stop_decision_block.train()

    end = time.time()
    print("#%s# lr:%.4f\ttau:%.4f" % (
        args.exp_header, optimizer.param_groups[-1]['lr'] * 0.1, tau if use_ada_framework else 0))
    
    accuracy_weight, efficiency_weight = update_weights(epoch, args.accuracy_weight, args.efficency_weight)
    
    accumulation_steps = args.repeat_batch
    total_loss = 0
    for i, input_tuple in enumerate(train_loader):
        data_time.update(time.time() - end)  # TODO(yue) measure data loading time

        target = input_tuple[-1].cuda()
        target_var = torch.autograd.Variable(target)

        input = input_tuple[0]
        if args.ada_reso_skip or args.ada_depth_skip:
            input_var_list = [torch.autograd.Variable(input_item) for input_item in input_tuple[:-1 + meta_offset]]

            if args.real_scsampler:
                output, r, all_policy_r, real_pred, lite_pred = model(input=input_var_list, tau=tau)
                if args.sal_rank_loss:
                    acc_loss = cal_sal_rank_loss(real_pred, lite_pred, target_var)
                else:
                    acc_loss = get_criterion_loss(criterion, lite_pred.mean(dim=1), target_var)
            else:
                if args.use_reinforce:
                    output, r, all_policy_r, r_log_prob, base_outs = model(input=input_var_list, tau=tau)
                elif args.use_conf_btw_blocks or args.use_early_stop:
                    output, r, all_policy_r, feat_outs, early_stop_r, reso_r = model(input=input_var_list, tau=tau)
                elif args.use_local_policy_module:
                    output, r, all_policy_r, base_outs, dual_policy_r, similarity_r = model(input=input_var_list, tau=tau)
                else:
                    output, r, all_policy_r, feat_outs, base_outs, reso_r = model(input=input_var_list, tau=tau)
                
                acc_loss = get_criterion_loss(criterion, output, target_var)

            if use_ada_framework:
                acc_loss, eff_loss, each_losses = compute_every_losses(r, all_policy_r, acc_loss, epoch)
                if args.use_kld_loss:
                    if args.use_conf_btw_blocks or args.use_early_stop:
                        base_outs = feat_outs[:,:,-1,:]
                    kld_loss = args.accuracy_weight * get_criterion_loss(criterion, amd_cal_kld(output, r, base_outs), target_var)
                    kld_losses.update(kld_loss.item(), input.size(0))
                    
                if args.use_conf_btw_blocks:
                    policy_gt_loss, inner_aloss= confidence_criterion_loss(criterion, all_policy_r, feat_outs, target_var)
#                     policy_gt_loss, inner_aloss = guide_criterion_loss_selected(criterion, all_policy_r, feat_outs, target_var, output, epoch)
                    #policy_gt_loss, inner_aloss= confidence_criterion_loss_selected(criterion, all_policy_r, feat_outs, target_var)

                    policy_gt_loss = efficiency_weight * policy_gt_loss
                    inner_aloss = accuracy_weight * inner_aloss
                    inner_alosses.update(inner_aloss.item(), input.size(0))
                    policy_gt_losses.update(policy_gt_loss.item(), input.size(0))
                    
                if args.use_early_exit:
                    early_exit_loss = early_exit_future_criterion_loss(criterion, exit_r_t, r, feat_outs, target_var)
#                     early_exit_loss = efficiency_weight * early_exit_loss
                    early_exit_losses.update(early_exit_loss.item(), input.size(0))
                    
                if args.use_early_stop:
                    early_stop_gt_loss = args.efficency_weight * 10 * early_stop_criterion_loss(criterion, all_policy_r, early_stop_r, feat_outs, target_var)
                    es_gt_losses.update(early_stop_gt_loss.item(), input.size(0))
                    
                if args.use_local_policy_module:
                    redundant_policy_loss, noisy_policy_loss = dual_policy_criterion_loss(criterion, base_outs, target_var, dual_policy_r, similarity_r)
                    redundant_policy_loss = redundant_policy_loss
                    noisy_policy_loss = noisy_policy_loss
                    
                    redundant_policy_losses.update(redundant_policy_loss.item(), input.size(0))
                    noisy_policy_losses.update(noisy_policy_loss.item(), input.size(0))
                    
                    

                if args.use_reinforce and not args.freeze_policy:
                    if args.separated:
                        acc_loss_items = []
                        eff_loss_items = []

                        for b_i in range(output.shape[0]):
                            acc_loss_item = get_criterion_loss(criterion, output[b_i:b_i + 1], target_var[b_i:b_i + 1])
                            acc_loss_item, eff_loss_item, each_losses_item = compute_every_losses(r[b_i:b_i + 1],
                                                                                                  acc_loss_item, epoch)

                            acc_loss_items.append(acc_loss_item)
                            eff_loss_items.append(eff_loss_item)

                        if args.no_baseline:
                            b_acc = 0
                            b_eff = 0
                        else:
                            b_acc = sum(acc_loss_items) / len(acc_loss_items)
                            b_eff = sum(eff_loss_items) / len(eff_loss_items)

                        log_p = torch.mean(r_log_prob, dim=1)

                        acc_loss = sum(acc_loss_items) / len(acc_loss_items)
                        eff_loss = sum(eff_loss_items) / len(eff_loss_items)

                        if args.detach_reward:
                            acc_loss_vec = (torch.stack(acc_loss_items) - b_acc).detach()
                            eff_loss_vec = (torch.stack(eff_loss_items) - b_eff).detach()
                        else:
                            acc_loss_vec = (torch.stack(acc_loss_items) - b_acc)
                            eff_loss_vec = (torch.stack(eff_loss_items) - b_eff)

                        intended_acc_loss = torch.mean(log_p * acc_loss_vec)
                        intended_eff_loss = torch.mean(log_p * eff_loss_vec)

                        each_losses = [0 * each_l for each_l in each_losses]

                    else:
                        sum_log_prob = torch.sum(r_log_prob) / r_log_prob.shape[0] / r_log_prob.shape[1]
                        acc_loss = - sum_log_prob * acc_loss
                        eff_loss = - sum_log_prob * eff_loss
                        each_losses = [-sum_log_prob * each_l for each_l in each_losses]

                    intended_loss = intended_acc_loss + intended_eff_loss

                alosses.update(acc_loss.item(), input.size(0))
                elosses.update(eff_loss.item(), input.size(0))


                for l_i, each_loss in enumerate(each_losses):
                    each_terms[l_i].update(each_loss, input.size(0))


            if args.use_kld_loss:
                loss = acc_loss + eff_loss + kld_loss
            elif args.use_conf_btw_blocks:
                loss = acc_loss + eff_loss + policy_gt_loss + inner_aloss 
#                 loss = acc_loss + eff_loss + inner_aloss 

#                 loss = acc_loss + policy_gt_loss

                if args.use_early_exit:
                    loss += early_exit_loss
            else:
#                 loss = acc_loss
                loss = acc_loss + eff_loss

            
            if args.use_local_policy_module:
                loss = loss + redundant_policy_loss + noisy_policy_loss
            elif args.use_early_stop:
                loss = loss + early_stop_gt_loss
#                 loss =  early_stop_gt_loss

        else:
            input_var = torch.autograd.Variable(input)
            output = model(input=[input_var])
            loss = get_criterion_loss(criterion, output, target_var)
        
         # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        if args.use_reinforce and not args.freeze_policy:
            intended_loss.backward()
        else:
            loss = loss / accumulation_steps
            loss.backward()
        if (i+1) % accumulation_steps == 0:
            if args.clip_gradient is not None:
                clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()
            optimizer.zero_grad()
            

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if use_ada_framework:
            r_list.append(r.detach().cpu().numpy())
            reso_r_list.append(reso_r.detach().cpu().numpy())
            if args.skip_twice:
                skip_twice_r = all_policy_r[:,:,:,-2]
                skip_twice_r_list.append(skip_twice_r.detach().cpu().numpy())

        if i % args.print_freq == 0:
            print_output = ('Epoch:[{0:02d}][{1:03d}/{2:03d}] '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            '{data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))  # TODO
            wandb.log({ "Train Loss val" : losses.val,
                        "Train Prec@1 val" : top1.val,
                        "Train Prec@5 val" : top5.val })

            if use_ada_framework:
                roh_r = reverse_onehot(r[-1, :, :].detach().cpu().numpy())
                if args.skip_twice:
                    st_roh_r = reverse_onehot(skip_twice_r[-1, :, :].detach().cpu().numpy())
                    if args.use_kld_loss:
                        print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t k_l {kld_loss.val:.4f} ({kld_loss.avg:.4f})\t r {r} st_r {st_r} pick {pick}'.format(
                            aloss=alosses, eloss=elosses, kld_loss=kld_losses, r=elastic_list_print(roh_r), st_r=elastic_list_print(st_roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                        )
                    else:
                        print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  r {r} st_r {st_r} pick {pick}'.format(
                            aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r), st_r=elastic_list_print(st_roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                        )
                elif args.use_kld_loss: 
                    print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  k_l {kld_loss.val:.4f} ({kld_loss.avg:.4f})\t r {r} pick {pick}'.format(
                        aloss=alosses, eloss=elosses, kld_loss=kld_losses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                    )
                    
                elif args.use_conf_btw_blocks:
                    print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  i_a_l {inner_aloss.val:.4f} ({inner_aloss.avg:.4f})\t  p_g_l {p_g_loss.val:.4f} ({p_g_loss.avg:.4f})\tr {r} pick {pick}'.format(
                        aloss=alosses, eloss=elosses, inner_aloss=inner_alosses, p_g_loss=policy_gt_losses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                    )
                    
                elif args.use_local_policy_module:
                    print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  r {r} pick {pick}'.format(
                        aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                    )
                    
                    red_r = dual_policy_r[-1, :, -1, 0, 1].detach().cpu().numpy()
                    noi_r = dual_policy_r[-1, :, -1, 1, 1].detach().cpu().numpy()
                    print_output += '\n red_p_l {red_p_l.val:.4f} ({red_p_l.avg:.4f})\t' .format(
                        red_p_l = redundant_policy_losses
                    )
                    print_output += '\n noi_p_l {noi_p_l.val:.4f} ({noi_p_l.avg:.4f})\t' .format(
                        noi_p_l = noisy_policy_losses
                    )
                    
                else:
                    print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  r {r} pick {pick}'.format(
                        aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                    )
                
                if args.use_early_stop:
                    es_r = early_stop_r[-1,:,0].detach().cpu().numpy()
                    print_output += '\n es_g_l {es_g_loss.val:.4f} ({es_g_loss.avg:.4f}), es_r {es} \t' .format(
                        es_g_loss=es_gt_losses, es = np.nonzero(es_r)
                    )
                if args.use_early_exit:
                    print_output += '\n e_x_l {e_x_loss.val:.4f} ({e_x_loss.avg:.4f})\t' .format(
                        e_x_loss=early_exit_losses
                    )
                
                    
                print_output += extra_each_loss_str(each_terms)
            if args.show_pred:
                print_output += elastic_list_print(output[-1, :].detach().cpu().numpy())
            print(print_output)

    if use_ada_framework:
        if args.ada_depth_skip :
            if args.skip_twice :
                usage_str, gflops = amd_get_policy_usage_str(r_list, skip_twice_r_list, len(args.block_rnn_list)+1, None)
            else:
                usage_str, gflops = amd_get_policy_usage_str(r_list, None, len(args.block_rnn_list)+1, reso_r_list)

        else:
            usage_str, gflops = get_policy_usage_str(r_list, model.module.reso_dim)
        print(usage_str)

    if tf_writer is not None:
        tf_writer.add_scalar('loss/train', losses.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

    return usage_str if use_ada_framework else None


def validate(val_loader, model, criterion, epoch, logger, exp_full_path, tf_writer=None):
    batch_time, losses, top1, top5 = get_average_meters(4)
    tau = 0
    # TODO(yue)
    all_results = []
    all_targets = []
    all_local = {"TN":0, "FN":0, "FP":0, "TP":0}
    all_all_preds = []

    i_dont_need_bb = True

    if args.visual_log != '':
        try:
            if not(os.path.isdir(args.visual_log)):
               os.makedirs(ospj(args.visual_log))

            visual_log_path = args.visual_log
            if args.ada_depth_skip :
                visual_log_txt_path = ospj(visual_log_path, "amd_visual_log.txt")
            else:
                visual_log_txt_path = ospj(visual_log_path, "visual_log.txt")
            visual_log = open(visual_log_txt_path, "w")


        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!")
                raise
                
    if args.cnt_log != '':
        try:
            if not(os.path.isdir(args.cnt_log)):
               os.makedirs(ospj(args.cnt_log))

            cnt_log_path = args.cnt_log
            if args.ada_depth_skip :
                cnt_log_txt_path = ospj(cnt_log_path, "amd_cnt_log.txt")
            else:
                cnt_log_txt_path = ospj(cnt_log_path, "cnt_log.txt")
            cnt_log = open(cnt_log_txt_path, "w")
            input_result_dict = {}
            total_cnt_dict = {}
            target_dict = {}


        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!")
                raise

    if use_ada_framework:
        tau = get_current_temperature(epoch)
        if args.use_early_stop:
            if args.use_conf_btw_blocks:
                alosses, elosses, inner_alosses, policy_gt_losses, es_gt_losses = get_average_meters(5)
            else:
                alosses, elosses, kld_losses, es_gt_losses = get_average_meters(4)
        else:        
            if args.use_conf_btw_blocks:
                alosses, elosses, inner_alosses, policy_gt_losses, early_exit_losses = get_average_meters(5)
            elif args.use_local_policy_module:
                alosses, elosses, redundant_policy_losses, noisy_policy_losses = get_average_meters(4)
            else: 
                alosses, elosses, kld_losses = get_average_meters(3)
                      
            
        kld_loss = 0
        iter_list = args.backbone_list

        if not i_dont_need_bb:
            all_bb_results = [[] for _ in range(len(iter_list))]
            if args.policy_also_backbone:
                all_bb_results.append([])

        each_terms = get_average_meters(NUM_LOSSES)
        
        r_list = []
        reso_r_list = []
        if args.skip_twice:
            skip_twice_r_list = [] 
            
        if args.save_meta:
            name_list = []
            indices_list = []

    meta_offset = -2 if args.save_meta else 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    accuracy_weight, efficiency_weight = update_weights(epoch, args.accuracy_weight, args.efficency_weight)

    
    accumulation_steps = args.repeat_batch
    total_loss = 0
    total_runtime = 0
    with torch.no_grad():
        for i, input_tuple in enumerate(val_loader):
            #input_tuple = input_tuple[0]
            target = input_tuple[-1].cuda()
#             local_target = input_tuple[2].cuda()
#             pdb.set_trace()
            input = input_tuple[0]

            # compute output

            if args.ada_reso_skip or args.ada_depth_skip:
                if args.real_scsampler:
                    output, r, all_policy_r, real_pred, lite_pred = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                    if args.sal_rank_loss:
                        acc_loss = cal_sal_rank_loss(real_pred, lite_pred, target)
                    else:
                        acc_loss = get_criterion_loss(criterion, lite_pred.mean(dim=1), target)
                else:
                    start_runtime = time.time()
                    if args.save_meta and args.save_all_preds:
                        output, r, all_policy_r, all_preds = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                    elif args.use_conf_btw_blocks or args.use_early_stop:
                        output, r, all_policy_r, feat_outs, early_stop_r, reso_r = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                    elif args.use_local_policy_module:
                        output, r, all_policy_r, base_outs, dual_policy_r, similarity_r = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                    else:
                        output, r, all_policy_r, feat_outs, base_outs, reso_r = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                    total_runtime += (time.time() - start_runtime)
                    acc_loss = get_criterion_loss(criterion, output, target)

                if use_ada_framework:
                    acc_loss, eff_loss, each_losses = compute_every_losses(r, all_policy_r, acc_loss, epoch)
                    acc_loss = acc_loss * accuracy_weight
                    eff_loss = eff_loss * efficiency_weight
                    if args.use_kld_loss:
                        kld_loss = args.accuracy_weight * get_criterion_loss(criterion, amd_cal_kld(output, r, base_outs), target)
                        kld_losses.update(kld_loss.item(), input.size(0))

                    elif args.use_conf_btw_blocks:
#                         policy_gt_loss, inner_aloss= confidence_criterion_loss(criterion, all_policy_r, feat_outs, target)
#                         policy_gt_loss, inner_aloss = guide_criterion_loss_selected(criterion, all_policy_r, feat_outs, target, output, epoch)

                        policy_gt_loss, inner_aloss= confidence_criterion_loss_selected(criterion, all_policy_r, feat_outs, target)
                        policy_gt_loss, inner_aloss= acc_loss, eff_loss
                        policy_gt_loss = efficiency_weight * policy_gt_loss
                        inner_aloss = accuracy_weight * inner_aloss
                        inner_alosses.update(inner_aloss.item(), input.size(0))
                        policy_gt_losses.update(policy_gt_loss.item(), input.size(0))
                        
                    elif args.use_local_policy_module:
                        redundant_policy_loss, noisy_policy_loss = dual_policy_criterion_loss(criterion, base_outs, target, dual_policy_r, similarity_r)
                        redundant_policy_loss = redundant_policy_loss
                        noisy_policy_loss = noisy_policy_loss

                        redundant_policy_losses.update(redundant_policy_loss.item(), input.size(0))
                        noisy_policy_losses.update(noisy_policy_loss.item(), input.size(0))
                        
                    if args.use_early_exit:
                        early_exit_loss = early_exit_future_criterion_loss(criterion, exit_r_t, r, feat_outs, target)
#                         early_exit_loss = efficiency_weight * early_exit_loss
                        early_exit_losses.update(early_exit_loss.item(), input.size(0))
                    
                    if args.use_early_stop:
                        early_stop_gt_loss = args.efficency_weight * early_stop_criterion_loss(criterion, all_policy_r, early_stop_r, feat_outs, target)
                        es_gt_losses.update(early_stop_gt_loss.item(), input.size(0))
                    
                    if args.use_reinforce and not args.freeze_policy:
                        if args.separated:
                            acc_loss_items = []
                            eff_loss_items = []
                            for b_i in range(output.shape[0]):
                                acc_loss_item = get_criterion_loss(criterion, output[b_i:b_i + 1],
                                                                   target[b_i:b_i + 1])
                                acc_loss_item, eff_loss_item, each_losses_item = compute_every_losses(r[b_i:b_i + 1],
                                                                                                      acc_loss_item,
                                                                                                      epoch)
                                acc_loss_items.append(acc_loss_item)
                                eff_loss_items.append(eff_loss_item)

                            if args.no_baseline:
                                b_acc = 0
                                b_eff = 0
                            else:
                                b_acc = sum(acc_loss_items) / len(acc_loss_items)
                                b_eff = sum(eff_loss_items) / len(eff_loss_items)

                            log_p = torch.mean(r_log_prob, dim=1)
                            acc_loss = 0
                            eff_loss = 0
                            for b_i in range(len(acc_loss_items)):
                                acc_loss += -log_p[b_i] * (acc_loss_items[b_i] - b_acc)
                                eff_loss += -log_p[b_i] * (eff_loss_items[b_i] - b_eff)
                            acc_loss = acc_loss / len(acc_loss_items)
                            eff_loss = eff_loss / len(eff_loss_items)
                            each_losses = [0 * each_l for each_l in each_losses]
                        else:
                            sum_log_prob = torch.sum(r_log_prob) / r_log_prob.shape[0] / r_log_prob.shape[1]
                            acc_loss = - sum_log_prob * acc_loss
                            eff_loss = - sum_log_prob * eff_loss
                            each_losses = [-sum_log_prob * each_l for each_l in each_losses]

                    alosses.update(acc_loss.item(), input.size(0))
                    elosses.update(eff_loss.item(), input.size(0))
                    for l_i, each_loss in enumerate(each_losses):
                        each_terms[l_i].update(each_loss, input.size(0))
                if args.use_kld_loss:
                    loss = acc_loss + eff_loss + kld_loss
                elif args.use_conf_btw_blocks:
                    loss = acc_loss + eff_loss + policy_gt_loss + inner_aloss
#                     loss = acc_loss + eff_loss + inner_aloss

#                     loss = acc_loss + policy_gt_loss 

                    if args.use_early_exit:
                        loss += early_exit_loss
                else:
#                     loss = acc_loss
                    loss = acc_loss + eff_loss
                    
                if args.use_early_stop:
                    loss = loss + early_stop_gt_loss
            else:
                output = model(input=[input])
                loss = get_criterion_loss(criterion, output, target)
                
            print("avg_runtime:{0:.8f},  total_runtime:{1:.2f}, total = {2:d}".format(total_runtime/float(len(val_loader)),total_runtime,  len(val_loader)))
             # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
              
            if args.cnt_log != '':
                target_vals = target.cpu().numpy()
                output_vals = output.max(dim=1)[1].cpu().numpy()
                
                for i in range(len(target_vals)):
                    target_val = target_vals[i][0]
                    output_val = output_vals[i]
                    input_path = os.path.join(args.root_path, input_tuple[meta_offset-1][i])
                    
                    if input_path in input_result_dict:
                        if target_val == output_val:
                            input_result_dict[input_path] +=1
                        total_cnt_dict[input_path] +=1
                    else:
                        input_result_dict[input_path] = 1 if target_val == output_val else 0
                        total_cnt_dict[input_path] = 1
                        target_dict[input_path] = output_val
                   
                
            if args.visual_log != '':
                target_val = target.cpu().numpy()[0][0]
                output_val = output.max(dim=1)[1].cpu().numpy()[0]
#                 loc_target_val = local_target.cpu().numpy()[0]
#                 loc_output_val = r[:,:,-1].cpu().numpy()[0]
                
            
                input_path_list = list()
                image_tmpl='image_{:05d}.jpg'
                for seg_ind in input_tuple[meta_offset][0]:
                    input_path_list.append(os.path.join(args.root_path, input_tuple[meta_offset-1][0], image_tmpl.format(int(seg_ind))))

              
                
                if target_val == output_val :
                    print("True")
                    visual_log.write("\nTrue")
                else :
                    print("False")
                    visual_log.write("\nFalse")

                print('input path list')
                print(input_path_list[0])
#                 print(input_path_list[0])
#                 print(lambda x : x.cpu.numpy(), input_path_list[1:])
                print('target')
                print(target_val)
                print('output')
                print(output_val)
                print('r')
#                 print('loc_target')
#                 print(loc_target_val)
#                 print('loc_output')
#                 print(loc_output_val)
                
                for i in range(1):
                    print(reverse_onehot(r[i, :, :].cpu().numpy()))

                #visual_log.write('\ninput path list: ')
                for i in range(len(input_path_list)):
                    visual_log.write('\n')
                    visual_log.write(input_path_list[i])

                visual_log.write('\n')
                visual_log.write(str(target_val))
                visual_log.write('\n')
                visual_log.write(str(output_val))
                visual_log.write('\n')
#                 visual_log.write(str(loc_target_val))
#                 visual_log.write('\n')
#                 visual_log.write(str(loc_output_val))
#                 visual_log.write('\n')

                
                for i in range(1):
                    visual_log.writelines(str(reverse_onehot(r[i, :, :].cpu().numpy())))
                visual_log.write('\n')

            # TODO(yue)
            all_results.append(output)
            all_targets.append(target)
#             total_loc = (local_target+2*r[:,:,-1]).cpu().numpy()# (0,1) + 2*(0,1) =? TN:0 FN:1 FP:2 TP:3
#             all_local['TN'] += np.count_nonzero(total_loc == 0)
#             all_local['FN'] += np.count_nonzero(total_loc == 1)
#             all_local['FP'] += np.count_nonzero(total_loc == 2)
#             all_local['TP'] += np.count_nonzero(total_loc == 3)
            

            
            if not i_dont_need_bb:
                for bb_i in range(len(all_bb_results)):
                    all_bb_results[bb_i].append(base_outs[:, bb_i])

            if args.save_meta and args.save_all_preds:
                all_all_preds.append(all_preds)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
              
            
            if use_ada_framework:
                r_list.append(r.cpu().numpy())
                reso_r_list.append(reso_r.cpu().numpy())
                if args.skip_twice:
                    skip_twice_r = all_policy_r[:,:,:,-2]
                    skip_twice_r_list.append(skip_twice_r.detach().cpu().numpy())

#                 if args.save_meta:
#                     name_list += input_tuple[-3]
#                     indices_list.append(input_tuple[-2])

            if i % args.print_freq == 0:
                print_output = ('Test: [{0:03d}/{1:03d}] '
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

                if use_ada_framework:
                    roh_r = reverse_onehot(r[-1, :, :].detach().cpu().numpy())
                    if args.skip_twice:
                        st_roh_r = reverse_onehot(skip_twice_r[-1, :, :].detach().cpu().numpy())
                        print_output += ' \n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  r {r} st_r {st_r} pick {pick}'.format(
                            aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r), st_r=elastic_list_print(st_roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                        )
                    elif args.use_kld_loss: 
                        print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  k_l {kld_loss.val:.4f} ({kld_loss.avg:.4f})\t r {r} pick {pick}'.format(
                            aloss=alosses, eloss=elosses, kld_loss=kld_losses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                        )
                    
                    elif args.use_conf_btw_blocks:
                        print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  i_a_l {inner_aloss.val:.4f} ({inner_aloss.avg:.4f})\t  p_g_l {p_g_loss.val:.4f} ({p_g_loss.avg:.4f})\tr {r} pick {pick}'.format(aloss=alosses, eloss=elosses, inner_aloss=inner_alosses, p_g_loss=policy_gt_losses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                        )
                    elif args.use_local_policy_module:
                        print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  r {r} pick {pick}'.format(
                            aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                        )

                        red_r = dual_policy_r[-1, :, -1, 0, 1].detach().cpu().numpy()
                        noi_r = dual_policy_r[-1, :, -1, 1, 1].detach().cpu().numpy()

                        print_output += '\n red_p_l {red_p_l.val:.4f} ({red_p_l.avg:.4f})\t' .format(
                            red_p_l = redundant_policy_losses
                        )
                        print_output += '\n noi_p_l {noi_p_l.val:.4f} ({noi_p_l.avg:.4f})\t' .format(
                            noi_p_l = noisy_policy_losses
                        )
                    else:
                        print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  r {r} pick {pick}'.format(
                            aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                        )
                        
                    if args.use_early_stop:
                        es_r = early_stop_r[-1,:,0].detach().cpu().numpy()
                        print_output += '\n es_g_l {es_g_loss.val:.4f} ({es_g_loss.avg:.4f}), es_r {es} \t' .format(
                            es_g_loss=es_gt_losses, es = np.nonzero(es_r)
                        )
                    if args.use_early_exit:
                        print_output += '\n e_x_l {e_x_loss.val:.4f} ({e_x_loss.avg:.4f}) \t' .format(
                            e_x_loss=early_exit_losses
                        )
                    
#                     #TN:0 FN:1 FP:2 TP:3
#                     print_output += extra_each_loss_str(each_terms)
#                     print_output += '\n location TP:{}, FP:{}, FN:{} ,TN: {} \t'.format(
#                         all_local['TP'], all_local['FP'], all_local['FN'], all_local['TN']
#                     )

                print(print_output)
    
    mAP, _ = cal_map(torch.cat(all_results, 0).cpu(),
                     torch.cat(all_targets, 0)[:, 0:1].cpu())  # TODO(yue) single-label mAP
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())  # TODO(yue)  multi-label mAP
    print('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses))

    wandb.log({"Test Loss" : losses.avg,
        "Test mAP" : mAP,
        "Test Prec@1" : top1.avg,
        "Test Prec@5" : top5.avg })
    
    if not i_dont_need_bb:
        bbmmaps = []
        bbprec1s = []
        all_targets_cpu = torch.cat(all_targets, 0).cpu()
        for bb_i in range(len(all_bb_results)):
            bb_results_cpu = torch.mean(torch.cat(all_bb_results[bb_i], 0), dim=1).cpu()
            bb_i_mmAP, _ = cal_map(bb_results_cpu, all_targets_cpu)  # TODO(yue)  multi-label mAP
            bbmmaps.append(bb_i_mmAP)

            bbprec1, = accuracy(bb_results_cpu, all_targets_cpu[:, 0], topk=(1,))
            bbprec1s.append(bbprec1)

        print("bbmmAP: " + " ".join(["{0:.3f}".format(bb_i_mmAP) for bb_i_mmAP in bbmmaps]))
        print("bb_Acc: " + " ".join(["{0:.3f}".format(bbprec1) for bbprec1 in bbprec1s]))
    gflops = 0

    
    if use_ada_framework:
        if args.ada_depth_skip :
            if args.skip_twice :
                usage_str, gflops = amd_get_policy_usage_str(r_list, skip_twice_r_list, len(args.block_rnn_list)+1,None)
            else:
                usage_str, gflops = amd_get_policy_usage_str(r_list, None, len(args.block_rnn_list)+1, reso_r_list)

        else:
            usage_str, gflops = get_policy_usage_str(r_list, model.module.reso_dim)
        print(usage_str)
    
#         if args.save_meta:  # TODO save name, label, r, result

#             npa = np.concatenate(r_list)
#             npb = np.stack(name_list)
#             npc = torch.cat(all_results).cpu().numpy()
#             npd = torch.cat(all_targets).cpu().numpy()
#             if args.save_all_preds:
#                 npe = torch.cat(all_all_preds).cpu().numpy()
#             else:
#                 npe = np.zeros(1)

#             npf = torch.cat(indices_list).cpu().numpy()

#             np.savez("%s/meta-val-%s.npy" % (exp_full_path, logger._timestr),
#                      rs=npa, names=npb, results=npc, targets=npd, all_preds=npe, indices=npf)

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    
    if args.cnt_log != '':
        for k,v in input_result_dict.items():
            cnt_log.write(str(k))
            cnt_log.write(',')
            cnt_log.write(str(target_dict[k]))
            cnt_log.write(',')
            cnt_log.write(str(v))
            cnt_log.write(',')
            cnt_log.write(str(total_cnt_dict[k]))
            cnt_log.write('\n')
            
            
        cnt_log.close()
    
    if args.visual_log != '':
        visual_log.close()

    return mAP, mmAP, top1.avg, usage_str if use_ada_framework else None, gflops


def runtime_validate(val_loader, model, criterion, epoch, logger, exp_full_path, tf_writer=None):
    batch_time, losses, top1, top5 = get_average_meters(4)
    tau = 0
    # TODO(yue)
    all_results = []
    all_targets = []
    all_local = {"TN":0, "FN":0, "FP":0, "TP":0}
    all_all_preds = []

    i_dont_need_bb = True

    if args.visual_log != '':
        try:
            if not(os.path.isdir(args.visual_log)):
               os.makedirs(ospj(args.visual_log))

            visual_log_path = args.visual_log
            if args.ada_depth_skip :
                visual_log_txt_path = ospj(visual_log_path, "amd_visual_log.txt")
            else:
                visual_log_txt_path = ospj(visual_log_path, "visual_log.txt")
            visual_log = open(visual_log_txt_path, "w")


        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!")
                raise
                
    if args.cnt_log != '':
        try:
            if not(os.path.isdir(args.cnt_log)):
               os.makedirs(ospj(args.cnt_log))

            cnt_log_path = args.cnt_log
            if args.ada_depth_skip :
                cnt_log_txt_path = ospj(cnt_log_path, "amd_cnt_log.txt")
            else:
                cnt_log_txt_path = ospj(cnt_log_path, "cnt_log.txt")
            cnt_log = open(cnt_log_txt_path, "w")
            input_result_dict = {}
            total_cnt_dict = {}
            target_dict = {}


        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!")
                raise

    if use_ada_framework:
        tau = get_current_temperature(epoch)
        if args.use_early_stop:
            if args.use_conf_btw_blocks:
                alosses, elosses, inner_alosses, policy_gt_losses, es_gt_losses = get_average_meters(5)
            else:
                alosses, elosses, kld_losses, es_gt_losses = get_average_meters(4)
        else:        
            if args.use_conf_btw_blocks:
                alosses, elosses, inner_alosses, policy_gt_losses, early_exit_losses = get_average_meters(5)
            elif args.use_local_policy_module:
                alosses, elosses, redundant_policy_losses, noisy_policy_losses = get_average_meters(4)
            else: 
                alosses, elosses, kld_losses = get_average_meters(3)
                      
            
        kld_loss = 0
        iter_list = args.backbone_list

        if not i_dont_need_bb:
            all_bb_results = [[] for _ in range(len(iter_list))]
            if args.policy_also_backbone:
                all_bb_results.append([])

        each_terms = get_average_meters(NUM_LOSSES)
        
        r_list = []
        reso_r_list = []
        if args.skip_twice:
            skip_twice_r_list = [] 
            
        if args.save_meta:
            name_list = []
            indices_list = []

    meta_offset = -2 if args.save_meta else 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    accuracy_weight, efficiency_weight = update_weights(epoch, args.accuracy_weight, args.efficency_weight)

    
    accumulation_steps = args.repeat_batch
    total_loss = 0
    total_runtime = 0
    total_used = 0
    with torch.no_grad():
        for i, input_tuple in enumerate(val_loader):
            #input_tuple = input_tuple[0]
            target = input_tuple[-1].cuda()
#             local_target = input_tuple[2].cuda()
#             pdb.set_trace()
            input = input_tuple[0]

            # compute output
            if args.ada_reso_skip or args.ada_depth_skip:
                if args.real_scsampler:
                    output, r, all_policy_r, real_pred, lite_pred = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                    if args.sal_rank_loss:
                        acc_loss = cal_sal_rank_loss(real_pred, lite_pred, target)
                    else:
                        acc_loss = get_criterion_loss(criterion, lite_pred.mean(dim=1), target)
                else:
                    start_runtime = time.time()
                    if args.save_meta and args.save_all_preds:
                        output, r, all_policy_r, all_preds = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                    elif args.use_conf_btw_blocks or args.use_early_stop:
                        output, r, all_policy_r, feat_outs, early_stop_r, reso_r = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                    elif args.use_local_policy_module:
                        output, r, all_policy_r, base_outs, dual_policy_r, similarity_r = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                    else:
                        output, r, all_policy_r, feat_outs, reso_r = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                    total_runtime += (time.time() - start_runtime)
                    acc_loss = get_criterion_loss(criterion, output, target)

                if use_ada_framework:
#                     acc_loss, eff_loss, each_losses = compute_every_losses(r, all_policy_r, acc_loss, epoch)
#                     acc_loss = acc_loss/args.accuracy_weight * accuracy_weight
#                     eff_loss = eff_loss/args.efficency_weight* efficiency_weight
#                     if args.use_kld_loss:
#                         kld_loss = args.accuracy_weight * get_criterion_loss(criterion, amd_cal_kld(output, r, base_outs), target)
#                         kld_losses.update(kld_loss.item(), input.size(0))

                    if args.use_conf_btw_blocks:
#                         policy_gt_loss, inner_aloss= confidence_criterion_loss(criterion, all_policy_r, feat_outs, target)
#                         policy_gt_loss, inner_aloss = guide_criterion_loss_selected(criterion, all_policy_r, feat_outs, target, output, epoch)

#                         policy_gt_loss, inner_aloss= confidence_criterion_loss_selected(criterion, all_policy_r, feat_outs, target)

#                         policy_gt_loss = efficiency_weight * policy_gt_loss
#                         inner_aloss = accuracy_weight * inner_aloss
                        eff_loss = torch.tensor(0.0).cuda()
                        inner_aloss = torch.tensor(0.0).cuda()
                        policy_gt_loss = torch.tensor(0.0).cuda()

                        inner_alosses.update(inner_aloss.item(), input.size(0))
                        policy_gt_losses.update(policy_gt_loss.item(), input.size(0))
                        
                    elif args.use_local_policy_module:
                        redundant_policy_loss, noisy_policy_loss = dual_policy_criterion_loss(criterion, base_outs, target, dual_policy_r, similarity_r)
                        redundant_policy_loss = redundant_policy_loss
                        noisy_policy_loss = noisy_policy_loss

                        redundant_policy_losses.update(redundant_policy_loss.item(), input.size(0))
                        noisy_policy_losses.update(noisy_policy_loss.item(), input.size(0))
                        
                    if args.use_early_exit:
                        early_exit_loss = early_exit_future_criterion_loss(criterion, exit_r_t, r, feat_outs, target)
#                         early_exit_loss = efficiency_weight * early_exit_loss
                        early_exit_losses.update(early_exit_loss.item(), input.size(0))
                    
                    if args.use_early_stop:
                        early_stop_gt_loss = args.efficency_weight * early_stop_criterion_loss(criterion, all_policy_r, early_stop_r, feat_outs, target)
                        es_gt_losses.update(early_stop_gt_loss.item(), input.size(0))
                    
                    if args.use_reinforce and not args.freeze_policy:
                        if args.separated:
                            acc_loss_items = []
                            eff_loss_items = []
                            for b_i in range(output.shape[0]):
                                acc_loss_item = get_criterion_loss(criterion, output[b_i:b_i + 1],
                                                                   target[b_i:b_i + 1])
                                acc_loss_item, eff_loss_item, each_losses_item = compute_every_losses(r[b_i:b_i + 1],
                                                                                                      acc_loss_item,
                                                                                                      epoch)
                                acc_loss_items.append(acc_loss_item)
                                eff_loss_items.append(eff_loss_item)

                            if args.no_baseline:
                                b_acc = 0
                                b_eff = 0
                            else:
                                b_acc = sum(acc_loss_items) / len(acc_loss_items)
                                b_eff = sum(eff_loss_items) / len(eff_loss_items)

                            log_p = torch.mean(r_log_prob, dim=1)
                            acc_loss = 0
                            eff_loss = 0
                            for b_i in range(len(acc_loss_items)):
                                acc_loss += -log_p[b_i] * (acc_loss_items[b_i] - b_acc)
                                eff_loss += -log_p[b_i] * (eff_loss_items[b_i] - b_eff)
                            acc_loss = acc_loss / len(acc_loss_items)
                            eff_loss = eff_loss / len(eff_loss_items)
                            each_losses = [0 * each_l for each_l in each_losses]
                        else:
                            sum_log_prob = torch.sum(r_log_prob) / r_log_prob.shape[0] / r_log_prob.shape[1]
                            acc_loss = - sum_log_prob * acc_loss
                            eff_loss = - sum_log_prob * eff_loss
                            each_losses = [-sum_log_prob * each_l for each_l in each_losses]
                    
                    alosses.update(acc_loss.item(), input.size(0))
                    elosses.update(eff_loss.item(), input.size(0))
#                     for l_i, each_loss in enumerate(each_losses):
#                         each_terms[l_i].update(each_loss, input.size(0))
                if args.use_kld_loss:
                    loss = acc_loss + eff_loss + kld_loss
                elif args.use_conf_btw_blocks:
                    loss = acc_loss + eff_loss + policy_gt_loss + inner_aloss
#                     loss = acc_loss + eff_loss + inner_aloss

#                     loss = acc_loss + policy_gt_loss 

                    if args.use_early_exit:
                        loss += early_exit_loss
                else:
#                     loss = acc_loss
                    loss = acc_loss + eff_loss
                    
                if args.use_early_stop:
                    loss = loss + early_stop_gt_loss
            else:
                start_runtime = time.time()

                output, num_used_frame = model(input=[input])
                loss = get_criterion_loss(criterion, output, target)
                total_runtime += (time.time() - start_runtime)
                total_used += num_used_frame

            print("avg_runtime:{0:.8f},  total_runtime:{1:.2f}, num_of_used:{2:d}, total = {3:d}".format(total_runtime/float(len(val_loader)),total_runtime, total_used, len(val_loader)))
             # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
              
            if args.cnt_log != '':
                target_vals = target.cpu().numpy()
                output_vals = output.max(dim=1)[1].cpu().numpy()
                
                for i in range(len(target_vals)):
                    target_val = target_vals[i][0]
                    output_val = output_vals[i]
                    input_path = os.path.join(args.root_path, input_tuple[meta_offset-1][i])
                    
                    if input_path in input_result_dict:
                        if target_val == output_val:
                            input_result_dict[input_path] +=1
                        total_cnt_dict[input_path] +=1
                    else:
                        input_result_dict[input_path] = 1 if target_val == output_val else 0
                        total_cnt_dict[input_path] = 1
                        target_dict[input_path] = output_val
                   
                
            if args.visual_log != '':
                target_val = target.cpu().numpy()[0][0]
                output_val = output.max(dim=1)[1].cpu().numpy()[0]
#                 loc_target_val = local_target.cpu().numpy()[0]
#                 loc_output_val = r[:,:,-1].cpu().numpy()[0]
                
            
                input_path_list = list()
                image_tmpl='image_{:05d}.jpg'
                for seg_ind in input_tuple[meta_offset][0]:
                    input_path_list.append(os.path.join(args.root_path, input_tuple[meta_offset-1][0], image_tmpl.format(int(seg_ind))))

              
                
                if target_val == output_val :
                    print("True")
                    visual_log.write("\nTrue")
                else :
                    print("False")
                    visual_log.write("\nFalse")

                print('input path list')
                print(input_path_list[0])
#                 print(input_path_list[0])
#                 print(lambda x : x.cpu.numpy(), input_path_list[1:])
                print('target')
                print(target_val)
                print('output')
                print(output_val)
                print('r')
#                 print('loc_target')
#                 print(loc_target_val)
#                 print('loc_output')
#                 print(loc_output_val)
                
                for i in range(1):
                    print(reverse_onehot(r[i, :, :].cpu().numpy()))

                #visual_log.write('\ninput path list: ')
                for i in range(len(input_path_list)):
                    visual_log.write('\n')
                    visual_log.write(input_path_list[i])

                visual_log.write('\n')
                visual_log.write(str(target_val))
                visual_log.write('\n')
                visual_log.write(str(output_val))
                visual_log.write('\n')
#                 visual_log.write(str(loc_target_val))
#                 visual_log.write('\n')
#                 visual_log.write(str(loc_output_val))
#                 visual_log.write('\n')

                
                for i in range(1):
                    visual_log.writelines(str(reverse_onehot(r[i, :, :].cpu().numpy())))
                visual_log.write('\n')

            # TODO(yue)
            all_results.append(output)
            all_targets.append(target)
#             total_loc = (local_target+2*r[:,:,-1]).cpu().numpy()# (0,1) + 2*(0,1) =? TN:0 FN:1 FP:2 TP:3
#             all_local['TN'] += np.count_nonzero(total_loc == 0)
#             all_local['FN'] += np.count_nonzero(total_loc == 1)
#             all_local['FP'] += np.count_nonzero(total_loc == 2)
#             all_local['TP'] += np.count_nonzero(total_loc == 3)
            

            
            if not i_dont_need_bb:
                for bb_i in range(len(all_bb_results)):
                    all_bb_results[bb_i].append(base_outs[:, bb_i])

            if args.save_meta and args.save_all_preds:
                all_all_preds.append(all_preds)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
              
            
            if use_ada_framework:
                r_list.append(r.cpu().numpy())
                reso_r_list.append(reso_r.cpu().numpy())
                if args.skip_twice:
                    skip_twice_r = all_policy_r[:,:,:,-2]
                    skip_twice_r_list.append(skip_twice_r.detach().cpu().numpy())

#                 if args.save_meta:
#                     name_list += input_tuple[-3]
#                     indices_list.append(input_tuple[-2])

            if i % args.print_freq == 0:
                print_output = ('Test: [{0:03d}/{1:03d}] '
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

#                 if use_ada_framework:
#                     roh_r = reverse_onehot(r[-1, :, :].detach().cpu().numpy())
#                     if args.skip_twice:
#                         st_roh_r = reverse_onehot(skip_twice_r[-1, :, :].detach().cpu().numpy())
#                         print_output += ' \n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  r {r} st_r {st_r} pick {pick}'.format(
#                             aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r), st_r=elastic_list_print(st_roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
#                         )
#                     elif args.use_kld_loss: 
#                         print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  k_l {kld_loss.val:.4f} ({kld_loss.avg:.4f})\t r {r} pick {pick}'.format(
#                             aloss=alosses, eloss=elosses, kld_loss=kld_losses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
#                         )
                    
#                     elif args.use_conf_btw_blocks:
#                         print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  i_a_l {inner_aloss.val:.4f} ({inner_aloss.avg:.4f})\t  p_g_l {p_g_loss.val:.4f} ({p_g_loss.avg:.4f})\tr {r} pick {pick}'.format(aloss=alosses, eloss=elosses, inner_aloss=inner_alosses, p_g_loss=policy_gt_losses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
#                         )
#                     elif args.use_local_policy_module:
#                         print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  r {r} pick {pick}'.format(
#                             aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
#                         )

#                         red_r = dual_policy_r[-1, :, -1, 0, 1].detach().cpu().numpy()
#                         noi_r = dual_policy_r[-1, :, -1, 1, 1].detach().cpu().numpy()

#                         print_output += '\n red_p_l {red_p_l.val:.4f} ({red_p_l.avg:.4f})\t' .format(
#                             red_p_l = redundant_policy_losses
#                         )
#                         print_output += '\n noi_p_l {noi_p_l.val:.4f} ({noi_p_l.avg:.4f})\t' .format(
#                             noi_p_l = noisy_policy_losses
#                         )
#                     else:
#                         print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  r {r} pick {pick}'.format(
#                             aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
#                         )
                        
#                     if args.use_early_stop:
#                         es_r = early_stop_r[-1,:,0].detach().cpu().numpy()
#                         print_output += '\n es_g_l {es_g_loss.val:.4f} ({es_g_loss.avg:.4f}), es_r {es} \t' .format(
#                             es_g_loss=es_gt_losses, es = np.nonzero(es_r)
#                         )
#                     if args.use_early_exit:
#                         print_output += '\n e_x_l {e_x_loss.val:.4f} ({e_x_loss.avg:.4f}) \t' .format(
#                             e_x_loss=early_exit_losses
#                         )
                    
#                     #TN:0 FN:1 FP:2 TP:3
#                     print_output += extra_each_loss_str(each_terms)
#                     print_output += '\n location TP:{}, FP:{}, FN:{} ,TN: {} \t'.format(
#                         all_local['TP'], all_local['FP'], all_local['FN'], all_local['TN']
#                     )

                print(print_output)

    mAP, _ = cal_map(torch.cat(all_results, 0).cpu(),
                     torch.cat(all_targets, 0)[:, 0:1].cpu())  # TODO(yue) single-label mAP
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())  # TODO(yue)  multi-label mAP
    print('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses))

    wandb.log({"Test Loss" : losses.avg,
        "Test mAP" : mAP,
        "Test Prec@1" : top1.avg,
        "Test Prec@5" : top5.avg })
    
    if not i_dont_need_bb:
        bbmmaps = []
        bbprec1s = []
        all_targets_cpu = torch.cat(all_targets, 0).cpu()
        for bb_i in range(len(all_bb_results)):
            bb_results_cpu = torch.mean(torch.cat(all_bb_results[bb_i], 0), dim=1).cpu()
            bb_i_mmAP, _ = cal_map(bb_results_cpu, all_targets_cpu)  # TODO(yue)  multi-label mAP
            bbmmaps.append(bb_i_mmAP)

            bbprec1, = accuracy(bb_results_cpu, all_targets_cpu[:, 0], topk=(1,))
            bbprec1s.append(bbprec1)

        print("bbmmAP: " + " ".join(["{0:.3f}".format(bb_i_mmAP) for bb_i_mmAP in bbmmaps]))
        print("bb_Acc: " + " ".join(["{0:.3f}".format(bbprec1) for bbprec1 in bbprec1s]))
    gflops = 0

    
    if use_ada_framework:
        if args.ada_depth_skip :
            if args.skip_twice :
                usage_str, gflops = amd_get_policy_usage_str(r_list, skip_twice_r_list, len(args.block_rnn_list)+1)
            else:
                 usage_str, gflops = amd_get_policy_usage_str(r_list, None, len(args.block_rnn_list)+1, reso_r_list)

        else:
            usage_str, gflops = get_policy_usage_str(r_list, model.module.reso_dim)
        print(usage_str)
    
#         if args.save_meta:  # TODO save name, label, r, result

#             npa = np.concatenate(r_list)
#             npb = np.stack(name_list)
#             npc = torch.cat(all_results).cpu().numpy()
#             npd = torch.cat(all_targets).cpu().numpy()
#             if args.save_all_preds:
#                 npe = torch.cat(all_all_preds).cpu().numpy()
#             else:
#                 npe = np.zeros(1)

#             npf = torch.cat(indices_list).cpu().numpy()

#             np.savez("%s/meta-val-%s.npy" % (exp_full_path, logger._timestr),
#                      rs=npa, names=npb, results=npc, targets=npd, all_preds=npe, indices=npf)

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    
    if args.cnt_log != '':
        for k,v in input_result_dict.items():
            cnt_log.write(str(k))
            cnt_log.write(',')
            cnt_log.write(str(target_dict[k]))
            cnt_log.write(',')
            cnt_log.write(str(v))
            cnt_log.write(',')
            cnt_log.write(str(total_cnt_dict[k]))
            cnt_log.write('\n')
            
            
        cnt_log.close()
    
    if args.visual_log != '':
        visual_log.close()

    return mAP, mmAP, top1.avg, usage_str if use_ada_framework else None, gflops




def save_checkpoint(state, is_best, exp_full_path, epoch):
    #torch.save(state, '{}/models/ckpt{:03}.pth.tar'.format(exp_full_path, epoch))
    if is_best:
        torch.save(state, '%s/models/ckpt.best.pth.tar' % (exp_full_path))

def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def setup_log_directory(logger, log_dir, exp_header):
    if args.ablation:
        return None

    exp_full_name = "g%s_%s" % (logger._timestr, exp_header)
    exp_full_path = ospj(log_dir, exp_full_name)
    os.makedirs(exp_full_path)
    os.makedirs(ospj(exp_full_path, "models"))
    logger.create_log(exp_full_path, test_mode, args.num_segments, args.batch_size, args.top_k)
    return exp_full_path


if __name__ == '__main__':
    best_prec1 = 0
    num_class = -1
    use_ada_framework = False
    NUM_LOSSES = 10
    gflops_table = {}
    args = parser.parse_args()
    test_mode = (args.test_from != "")

    if test_mode:  # TODO test mode:
        print("======== TEST MODE ========")
        args.skip_training = True

    main()

