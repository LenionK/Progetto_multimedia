import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

from SwinIR.models.network_swinir import SwinIR as net

def test(img_lq, model, tile ,window_size):

    tile_overlap = 32

    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = tile_overlap
        sf = 8

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


def hight_res(label,folder_frames, save_dir ): 

    model_path = "model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth"
    scale = 8
    training_patch_size = 48


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {model_path}')
        open(model_path, 'wb').write(r.content)

    #----- define_model ---- -

    model = net(upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key_g = 'params'

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    model.eval()
    model = model.to(device)

    #----- set_up ------ 

    #save_dir = f'results/swinir_x{scale}'
    #folder = './frames/'
    border = scale
    window_size = 8

    os.makedirs(save_dir, exist_ok=True)
    iteration(label, folder_frames , device, window_size, scale, model, save_dir , "HR")

    return f'./{save_dir}/HR'


def iteration(label, folder_frames, device, window_size, scale, model, save_dir , prefix):

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder_frames, '*')))):
        # read image
        imgname = os.path.basename(path)
        print(f"Sto elaborando frame [{imgname}]")
        label.setText(f"Sto elaborando frame [{imgname}]")

        img_lq = cv2.imread(f'{folder_frames}/{imgname}', cv2.IMREAD_COLOR).astype(np.float32) / 255.


        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, model, None ,window_size)
            output = output[..., :h_old * scale, :w_old * scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

        folder_path = f'{save_dir}/{prefix}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        cv2.imwrite(f'{folder_path}/{prefix}_{imgname}', output)

        



def reconstruct_video_from_frames(frame_dir, save_dir="results/video", video_name="output_video.mp4", frame_rate=30):

    frames = sorted(glob.glob(os.path.join(frame_dir, '*.png')))

    if not frames:
        print(f"No frames found in {frame_dir}. Make sure the frames exist.")
        return

    # Read the first frame to get the size (height, width)
    frame = cv2.imread(frames[0])
    height, width, _ = frame.shape

    # Create a VideoWriter object to write the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files, use 'XVID' for .avi files
    out_video = cv2.VideoWriter(os.path.join(save_dir, video_name), fourcc, frame_rate, (width, height))

    # Write each frame to the video
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out_video.write(frame)  # Write the frame to the video

    # Release the video writer and close the video file
    out_video.release()

    print(f"Video saved as {video_name} in {save_dir}")

def detect_image_type(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if len(image.shape) == 2 or image.shape[2] == 1:
        return "gray_dn"
    else:
        return "color_dn"

def denoise(label, folder_frames, save_dir): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prendi la prima immagine dalla cartella
    image_files = [f for f in os.listdir(folder_frames) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise FileNotFoundError("Nessuna immagine trovata in {}".format(folder_frames))

    first_image_path = os.path.join(folder_frames, image_files[0])
    task = detect_image_type(first_image_path)  # Determina il tipo di immagine

    # Seleziona il modello in base al tipo di immagine
    if task == "gray_dn":
        model_path = "model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth"
    else:
        model_path = "model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth"

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = f'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{os.path.basename(model_path)}'
        print(f'Downloading model {model_path}')
        r = requests.get(url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
    else:
        print(f'Loading model from {model_path}')

    # Definizione del modello
    model = net(
        upscale=1, 
        in_chans=1 if task == "gray_dn" else 3, 
        img_size=128, 
        window_size=8,
        img_range=1., 
        depths=[6, 6, 6, 6, 6, 6], 
        embed_dim=180, 
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, 
        upsampler='', 
        resi_connection='1conv'
    )

    param_key_g = 'params'
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model.get(param_key_g, pretrained_model), strict=True)

    model.eval()
    model = model.to(device)

    os.makedirs(save_dir, exist_ok=True)
    iteration(label, folder_frames, device, 8, 1, model, save_dir , "DE")

    return f'./{save_dir}/DE'
