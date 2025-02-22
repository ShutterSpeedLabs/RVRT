import argparse
import os
import numpy as np
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from collections import OrderedDict
from torch.utils.data import DataLoader
from data.dataset_video_test import VideoRecurrentTestDataset, VideoTestVimeo90KDataset, SingleVideoRecurrentTestDataset
from utils import utils_image as util

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    return trt_runtime.deserialize_cuda_engine(engine_data)

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    stream.synchronize()
    return [out['host'] for out in outputs]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='003_RVRT_videosr_bd_Vimeo_14frames', help='tasks: 001 to 006')
    parser.add_argument('--sigma', type=int, default=0, help='noise level for denoising: 10, 20, 30, 40, 50')
    parser.add_argument('--folder_lq', type=str, default='testsets/REDS4/sharp_bicubic',
                        help='input low-quality test video folder')
    parser.add_argument('--folder_gt', type=str, default=None,
                        help='input ground-truth test video folder')
    parser.add_argument('--tile', type=int, nargs='+', default=[100,128,128],
                        help='Tile size, [0,0,0] for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, nargs='+', default=[2,20,20],
                        help='Overlapping of different tiles')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers in data loading')
    parser.add_argument('--save_result', action='store_true', help='save resulting image')
    args = parser.parse_args()

    # Load TensorRT engine
    trt_runtime = trt.Runtime(TRT_LOGGER)
    engine_path = os.path.join(os.path.dirname(__file__), "../model_zoo_trt/003_RVRT_videosr_bd_Vimeo_14frames.trt")
    engine = load_engine(trt_runtime, engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Prepare dataset
    if 'vimeo' in args.folder_lq.lower():
        test_set = VideoTestVimeo90KDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                             'meta_info_file': "data/meta_info/meta_info_Vimeo90K_test_GT.txt",
                                             'mirror_sequence': True, 'num_frame': 7, 'cache_data': False})
    elif args.folder_gt is not None:
        test_set = VideoRecurrentTestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                              'sigma':args.sigma, 'num_frame':-1, 'cache_data': False})
    else:
        test_set = SingleVideoRecurrentTestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                                    'sigma':args.sigma, 'num_frame':-1, 'cache_data': False})

    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=1, shuffle=False)

    save_dir = f'results/{args.task}'
    if args.save_result:
        os.makedirs(save_dir, exist_ok=True)
        print(f'Results will be saved at {save_dir}')
    else:
        print("results will not be saved")

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    assert len(test_loader) != 0, f'No dataset found at {args.folder_lq}'

    for idx, batch in enumerate(test_loader):
        lq = batch['L'].numpy()
        folder = batch['folder']
        gt = batch['H'] if 'H' in batch else None

        # Inference
        inputs[0]['host'] = lq.ravel()
        output = do_inference(context, bindings, inputs, outputs, stream)
        output = np.array(output).reshape((1, 3, 2, 64, 64))

        if 'vimeo' in args.folder_lq.lower():
            output = (output[:, 3:4, :, :, :] + output[:, 10:11, :, :, :]) / 2
            batch['lq_path'] = batch['gt_path']

        test_results_folder = OrderedDict()
        test_results_folder['psnr'] = []
        test_results_folder['ssim'] = []

        if gt is not None:
            img_gt = gt[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if img_gt.ndim == 3:
                img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = np.squeeze(img_gt)

            test_results_folder['psnr'].append(util.calculate_psnr(output, img_gt, border=0))
            test_results_folder['ssim'].append(util.calculate_ssim(output, img_gt, border=0))
            if img_gt.ndim == 3:  # RGB image
                output = util.bgr2ycbcr(output.astype(np.float32) / 255.) * 255.
                img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                test_results_folder['psnr_y'].append(util.calculate_psnr(output, img_gt, border=0))
                test_results_folder['ssim_y'].append(util.calculate_ssim(output, img_gt, border=0))
            else:
                test_results_folder['psnr_y'] = test_results_folder['psnr']
                test_results_folder['ssim_y'] = test_results_folder['ssim']

        if gt is not None:
            psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
            ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
            psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
            ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['psnr_y'].append(psnr_y)
            test_results['ssim_y'].append(ssim_y)
            print('Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                  format(folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y))
        else:
            print('Testing {:20s}  ({:2d}/{})'.format(folder[0], idx, len(test_loader)))

    if gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        print('\n{} \n-- Average PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
              format(save_dir, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))

if __name__ == '__main__':
    main()