from __future__ import print_function, division
import os
import time
import subprocess
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import re
import glob
import pdb

parser = argparse.ArgumentParser(description="Dataset processor: Video->Frames")
parser.add_argument("dir_path", type=str, help="original dataset path")
parser.add_argument("dst_dir_path", type=str, help="dest path to save the frames")
parser.add_argument("--prefix", type=str, default="image_%05d.jpg", help="output image type")
parser.add_argument("--accepted_formats", type=str, default=[".mp4", ".mkv", ".webm", ".avi"], nargs="+",
                    help="list of input video formats")
parser.add_argument("--begin", type=int, default=0)
parser.add_argument("--end", type=int, default=666666666)
parser.add_argument("--file_list", type=str, default="")
parser.add_argument("--frame_rate", type=int, default=-1)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--parallel", action="store_true")
parser.add_argument("--dataset", type=str, default="activitynet", help="dataset activitynet/ minik/ fcvid")
parser.add_argument("--char", type=str, default="a", help="dataset activitynet/ minik/ fcvid")

parser.add_argument("--mode", type=str, default="train", help="dataset activitynet/ minik/ fcvid")
args = parser.parse_args()


def par_job(command):
    if args.dry_run:
        print(command)
    else:
        subprocess.call(command, shell=True)


if __name__ == "__main__":
    t0 = time.time()
    dir_path = args.dir_path
    dst_dir_path = args.dst_dir_path
    empty_video_list = []

    if args.dataset == 'fcvid':
        dominant_c = args.char
        if args.file_list == "":
            file_names = glob.glob(os.path.join(dir_path,'{}*/*'.format(dominant_c)))

        else:
            _file_names = [x.strip() for x in open(args.file_list).readlines()]

    else:    
        if args.file_list == "":
            file_names = sorted(os.listdir(dir_path))
        else:
            _file_names = [x.strip() for x in open(args.file_list).readlines()]
            file_names = []
            file_names_r = []
            #print(_file_names)
            if args.dataset == 'minik': 
                for _file_name in _file_names:
                    path = re.split('/',_file_name)[0] 
                    if args.mode =="train":
                        sub_path = re.split(' ', re.split('/',_file_name)[1])[0]
                        dst_path = sub_path
                    elif args.mode == "val":
                        dst_path = re.split(' ', re.split('/',_file_name)[1])[0]
                        sub_path = re.split('_00', dst_path)[0]
                        
                        
                    full_path = re.split(' ', path)[0]
                    full_path_r = re.split(' ', path)[0]
                    if len(re.split(' ', path)) > 1 :                        
                        for i, word in enumerate(re.split(' ', path)):
                            if i is 0:
                                continue
                            full_path += '_' 
                            full_path_r += ' '
                            full_path += word
                            full_path_r += word
                    full_path += '/'
                    full_path += sub_path
                    full_path += '.mp4'
                    full_path_r += '/'
                    full_path_r += dst_path
                    full_path_r += '.mp4'
            #        print(full_path)
             #       print(full_path_r)
                    file_names.append(full_path)
                    file_names_r.append(full_path_r)
                
#                 file_names = [re.split('/',x)[0] if len(re.split(' ', re.split('/',x)[0])) is 1 else ((full_path = word if i is 0 else full_path += ' ' + word for i, word in enumerate(re.split(' ', re.split('/',x)[0]))) + '/' + re.split(' ', re.split('/',x)[1])[0] + '.mp4') for x in _file_names]
#                 file_names = [re.split('/',x)[0] if len(re.split(' ', re.split('/',x)[0])) is 1 else (re.split(' ', re.split('/',x)[0])[0] + ' ' + re.split(' ', re.split('/',x)[0])[1])  + '/' + re.split(' ', re.split('/',x)[1])[0] + '.mp4' for x in _file_names]

                
    del_list = []
    for i, file_name in enumerate(file_names):
        if not any([x in file_name for x in args.accepted_formats]):
            del_list.append(i)
    file_names = [x for i, x in enumerate(file_names) if i not in del_list]
    file_names = file_names[args.begin:args.end + 1]
    print("%d videos to handle (after %d being removed)" % (len(file_names), len(del_list)))
    cmd_list = []
    for i, file_name in tqdm(enumerate(file_names)):

        name, ext = os.path.splitext(file_name)
        dst_directory_path = os.path.join(dst_dir_path, name)

        if args.dataset == 'minik':
            name_r, ext_r = os.path.splitext(file_names_r[i])
            dst_directory_path = os.path.join(dst_dir_path, name_r)
        
        video_file_path = os.path.join(dir_path, file_name)   
        if not os.path.exists(dst_directory_path):
            os.makedirs(dst_directory_path, exist_ok=True)
        else:
            continue

        
        for form in args.accepted_formats:
            video_file_path = name + form
            video_file_path = os.path.join(dir_path, video_file_path)
            if os.path.isfile(video_file_path):
                break
            
        if not os.path.isfile(video_file_path):
           print("empty")
           empty_video_list.append(video_file_path)
           print(video_file_path)
           print(len(empty_video_list))
#         if len(os.listdir(dst_directory_path)) != 0:
#             continue
        else:
            if args.dataset == 'minik':
                _name_r = re.split(' ', name_r)
                modified_path = _name_r[0]
                if len(modified_path) > 1:
                    for c in range(1, len(_name_r)):
                        modified_path += '\ '
                 #       print(_name_r[c]) 
#                        if '\(' or '\)'  in _name_r[c]:
                #            print(_name_r[c])
 #                           modified_path += '\\'
                        modified_path += _name_r[c]
                
                dst_directory_path = os.path.join(dst_dir_path, modified_path) 
                #print(dst_directory_path)
            
        if args.frame_rate > 0:
            frame_rate_str = "-r %d" % args.frame_rate
        else:
            frame_rate_str = ""
        
        cmd = 'ffmpeg -nostats -loglevel 0 -i {} -vf scale=-1:360 {} {}/{}'.format(video_file_path, frame_rate_str,
                                                                       dst_directory_path, args.prefix)
        if not args.parallel:
            if args.dry_run:
                 print(cmd)
            else:
                subprocess.call(cmd, shell=True)
        cmd_list.append(cmd)

    if args.parallel:
        with Pool(processes=args.num_workers) as pool:
            with tqdm(total=len(cmd_list)) as pbar:
                for _ in tqdm(pool.imap_unordered(par_job, cmd_list)):
                    pbar.update()
    t1 = time.time()
    print("Finished in %.4f seconds" % (t1 - t0))
    print("empty video list")
    print(empty_video_list)
    print(len(empty_video_list))
    os.system("stty sane")
