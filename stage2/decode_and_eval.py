import torch
import numpy as np
import argparse
import enc_dec_model
from PIL import Image
from torchvision import transforms
from glob import glob
from natsort import ns, natsorted
from sklearn import metrics
from utils_ybb import get_logger, img_to_DMlatents, distorsion_unit
import torch.nn.functional as F
from diffusers import AutoencoderKL
from loss.pytorch_ssim import ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import pdb

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_tensor_to_np(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    # Scale the values to [0, 255] and convert to uint8
    numpy_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return numpy_array

def main(args=None):
    os.makedirs(args.output_dir, exist_ok=True)

    # load model
    pretrained_dir = args.enc_dec_model_dir
    if args.image_size == 256:
        decoder = enc_dec_model.Extractor_forLatent_size256(secret_size=args.secret_length)
    else:
        decoder = enc_dec_model.Extractor_forLatent(secret_size=args.secret_length)
    decoder.load_state_dict(torch.load(os.path.join(pretrained_dir, "decoder_best_total_loss.pth")))
    decoder.eval()     
    decoder = decoder.to(device)

    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae")
    vae = vae.to(device)

    # load fixed secret
    secret_input = np.array([int(bit) for bit in args.fixed_secret], dtype=np.float32)
    secret_tensor = torch.from_numpy(secret_input).float().unsqueeze(0).to(device)

    for subfolder in os.listdir(args.wm_images_dir):
        subfolder_path = os.path.join(args.wm_images_dir, subfolder)
        print(f"正在处理子文件夹: {subfolder_path}")

        log_path = os.path.join(args.output_dir, f'{subfolder}.log')
        logger = get_logger(filename=log_path, name='Stage2-1_48bit_ablation')

        # prepare data
        wm_img_paths = glob(os.path.join(subfolder_path, '*.png'))
        wm_img_paths = natsorted(wm_img_paths, alg=ns.PATH)

        cover_img_paths = glob(os.path.join(args.cover_images_dir, '*.png'))
        cover_img_paths = natsorted(cover_img_paths, alg=ns.PATH)

        images_count = min(len(wm_img_paths), args.images_count)

        # metrics
        psnr_cover_wm = []
        ssim_cover_wm = []

        wm_metrics = []
        cover_metrics = []

        bit_acc = []
        brightness_wm_acc = []
        saturation_wm_acc = []
        contrast_wm_acc = []
        blur_wm_acc = []
        noise_wm_acc = []
        jpeg_compress_wm_acc = []
        resize_wm_acc = []
        sharpness_wm_acc = []

        wm_present = [] # 统计测试：p_value < 0.01
        brightness_wm_present = []
        saturation_wm_present = []
        contrast_wm_present = []
        blur_wm_present = []
        noise_wm_present = []
        jpeg_compress_wm_present = []
        resize_wm_present = []
        sharpness_wm_present = []

        with torch.no_grad():
            for i in range(images_count):
                per_cover_img = cover_img_paths[i]
                per_wm_img = wm_img_paths[i]
                
                img_name = os.path.basename(per_wm_img)
                logger.info(f'----- {img_name} -----')

                cover_img = Image.open(per_cover_img).convert('RGB')
                cover_img = cover_img.resize((args.image_size, args.image_size))
                cover_img = transforms.ToTensor()(cover_img).unsqueeze(0).to(device)
                
                wm_img = Image.open(per_wm_img).convert('RGB')
                wm_img = wm_img.resize((args.image_size, args.image_size))
                wm_img = transforms.ToTensor()(wm_img).unsqueeze(0).to(device)

                # image quality: PSNR SSIM LPIPS
                psnr_value = compare_psnr(convert_tensor_to_np(torch.clamp(cover_img,min=0,max=1)), convert_tensor_to_np(torch.clamp(wm_img,min=0,max=1)))
                psnr_cover_wm.append(psnr_value)
                ssim_value = ssim(wm_img, cover_img).item()
                ssim_cover_wm.append(ssim_value)
                logger.info(f'PSNR = {psnr_value:.2f}\tSSIM = {ssim_value:.3f}')

                # decode accuracy
                wm_latent = img_to_DMlatents(wm_img, vae)
                reveal_output = decoder(wm_latent)
                results_W = torch.round(torch.sigmoid(reveal_output))

                bit_accuracy = torch.sum(results_W - secret_tensor == 0).item() / secret_tensor.numel()
                logger.info(f'decode 48bits accuracy = {bit_accuracy:.4f}')
                bit_acc.append(bit_accuracy)
                wm_metrics.append(bit_accuracy)
                if bit_accuracy > 34/48:    # 34/48, 40/56, 65/96, 82/120
                    wm_present.append(img_name)

                # # adversary accuracy
                # distortion_list = ['blur', 'noise', 'jpeg_compress', 'resize', 'sharpness', "brightness", "contrast", "saturation"]
            
                # for distortion in distortion_list:
                #     distorted_image = distorsion_unit(wm_img, distortion)

                #     distorted_image = F.interpolate(
                #                         distorted_image,
                #                         size=(args.image_size, args.image_size),
                #                         mode='bilinear')
                #     distorted_latent = img_to_DMlatents(distorted_image, vae)
                #     reveal_output = decoder(distorted_latent)
                #     results = torch.round(torch.sigmoid(reveal_output))

                #     if distortion == 'resize':
                #         resize_acc = torch.sum(results - secret_tensor==0).item() / secret_tensor.numel()
                #         logger.info(f'resize_wm_acc: {resize_acc:.4f}')
                #         resize_wm_acc.append(resize_acc)
                #         if resize_acc > 34/48:    # 0.7083
                #             resize_wm_present.append(img_name)
                #     elif distortion == 'brightness':
                #         brightness_acc = torch.sum(results - secret_tensor==0).item() / secret_tensor.numel()
                #         logger.info(f'brightness_wm_acc: {brightness_acc:.4f}')
                #         brightness_wm_acc.append(brightness_acc)
                #         if brightness_acc > 34/48:    # 0.7083
                #             brightness_wm_present.append(img_name)
                #     elif distortion == 'contrast':
                #         contrast_acc = torch.sum(results - secret_tensor==0).item() / secret_tensor.numel()
                #         logger.info(f'contrast_wm_acc: {contrast_acc:.4f}')
                #         contrast_wm_acc.append(contrast_acc)
                #         if contrast_acc > 34/48:    # 0.7083
                #             contrast_wm_present.append(img_name)
                #     elif distortion == 'saturation':
                #         saturation_acc = torch.sum(results - secret_tensor==0).item() / secret_tensor.numel()
                #         logger.info(f'saturation_wm_acc: {saturation_acc:.4f}')
                #         saturation_wm_acc.append(saturation_acc)
                #         if saturation_acc > 34/48:    # 0.7083
                #             saturation_wm_present.append(img_name)
                #     elif distortion == 'blur':
                #         blur_acc = torch.sum(results - secret_tensor==0).item() / secret_tensor.numel()
                #         logger.info(f'blur_wm_acc: {blur_acc:.4f}')
                #         blur_wm_acc.append(blur_acc)
                #         if blur_acc > 34/48:    # 0.7083
                #             blur_wm_present.append(img_name)
                #     elif distortion == 'noise':
                #         noise_acc = torch.sum(results - secret_tensor==0).item() / secret_tensor.numel()
                #         logger.info(f'noise_wm_acc: {noise_acc:.4f}')
                #         noise_wm_acc.append(noise_acc)
                #         if noise_acc > 34/48:    # 0.7083
                #             noise_wm_present.append(img_name)
                #     elif distortion == 'jpeg_compress':
                #         jpeg_compress_acc = torch.sum(results - secret_tensor==0).item() / secret_tensor.numel()
                #         logger.info(f'jpeg_compress_wm_acc: {jpeg_compress_acc:.4f}')
                #         jpeg_compress_wm_acc.append(jpeg_compress_acc)
                #         if jpeg_compress_acc > 34/48:    # 0.7083
                #             jpeg_compress_wm_present.append(img_name)
                #     elif distortion == 'sharpness':
                #         sharpness_acc = torch.sum(results - secret_tensor==0).item() / secret_tensor.numel()
                #         logger.info(f'sharpness_wm_acc: {sharpness_acc:.4f}')
                #         sharpness_wm_acc.append(sharpness_acc)
                #         if sharpness_acc > 34/48:    # 0.7083
                #             sharpness_wm_present.append(img_name)

                # auroc
                cover_latent = img_to_DMlatents(cover_img, vae)
                reveal_output = decoder(cover_latent)
                results_C = torch.round(torch.sigmoid(reveal_output))

                cover_bit_accuracy = torch.sum(results_C - secret_tensor == 0).item() / secret_tensor.numel()
                cover_metrics.append(cover_bit_accuracy)
            
            preds_clean = cover_metrics +  wm_metrics
            t_labels_clean = [0] * len(cover_metrics) + [1] * len(wm_metrics)

            fpr_c, tpr_c, thresholds = metrics.roc_curve(t_labels_clean, preds_clean, pos_label=1)
            auc_c = metrics.auc(fpr_c, tpr_c)
            acc_c = np.max(1 - (fpr_c + (1 - tpr_c))/2)
            low_c = tpr_c[np.where(fpr_c<.01)[0][-1]]
            logger.info(f'\nClean watermarked images: AUC: {auc_c:.3f}, ACC: {acc_c:.3f}, TPR@1%FPR: {low_c}')

            
            logger.info('\n========= Average ============')
            logger.info(f'images number = {images_count}')
            logger.info('psnr')
            logger.info(f'{sum(psnr_cover_wm)/len(psnr_cover_wm):.4f}')
            logger.info('ssim')
            logger.info(f'{sum(ssim_cover_wm)/len(ssim_cover_wm):.4f}')
            logger.info('identity')
            logger.info(f'acc = {sum(bit_acc)/len(bit_acc):.4f}\twdr = {len(wm_present)/images_count:.4f}')
            # logger.info('resize_WM')
            # logger.info(f'acc = {sum(resize_wm_acc)/len(resize_wm_acc):.4f}\twdr = {len(resize_wm_present)/images_count:.4f}')
            # logger.info('contrast_WM')
            # logger.info(f'acc = {sum(contrast_wm_acc)/len(contrast_wm_acc):.4f}\twdr = {len(contrast_wm_present)/images_count:.4f}')
            # logger.info('brightness_WM')
            # logger.info(f'acc = {sum(brightness_wm_acc)/len(brightness_wm_acc):.4f}\twdr = {len(brightness_wm_present)/images_count:.4f}')
            # logger.info('saturation_WM')
            # logger.info(f'acc = {sum(saturation_wm_acc)/len(saturation_wm_acc):.4f}\twdr = {len(saturation_wm_present)/images_count:.4f}')
            # logger.info('blur_WM')
            # logger.info(f'acc = {sum(blur_wm_acc)/len(blur_wm_acc):.4f}\twdr = {len(blur_wm_present)/images_count:.4f}')
            # logger.info('noise_WM')
            # logger.info(f'acc = {sum(noise_wm_acc)/len(noise_wm_acc):.4f}\twdr = {len(noise_wm_present)/images_count:.4f}')
            # logger.info('jpeg_compress_WM')
            # logger.info(f'acc = {sum(jpeg_compress_wm_acc)/len(jpeg_compress_wm_acc):.4f}\twdr = {len(jpeg_compress_wm_present)/images_count:.4f}')
            # logger.info('sharpness_WM')
            # logger.info(f'acc = {sum(sharpness_wm_acc)/len(sharpness_wm_acc):.4f}\twdr = {len(sharpness_wm_present)/images_count:.4f}')
            logger.info('\n==============================')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--enc_dec_model_dir", default='stage1/output_dir_size256/checkpoints', type=str)
    parser.add_argument("--sd_model", default="HuggingFaceModels/stable-diffusion-2-1-base", type=str)
    parser.add_argument("--wm_images_dir", 
                        default="results/secret_size_48", 
                        type=str)
    parser.add_argument("--cover_images_dir", 
                        default="dataset/imagenet_compatible_org", 
                        type=str)
    parser.add_argument("--fixed_secret", default="011100011101100001001000010110100111001000010101", type=str)
    parser.add_argument("--secret_length", default=48, type=int)
    parser.add_argument("--images_count", default=100, type=int)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--output_dir", 
                        default="./output_v3_size256/log", 
                        type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    main(args=args)
