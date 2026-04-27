import os
import model
import torch
import argparse
import lpips
import logging
import numpy as np
from pathlib import Path
from dataset import ImageData
from transformers import get_linear_schedule_with_warmup
from diffusers import AutoencoderKL

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def get_logger(filename, verbosity=1, name=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # 确保不会重复添加处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建文件日志记录器
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 创建控制台日志记录器
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# 计算模型梯度范数
@torch.no_grad()
def log_avg_gradient_norm(obj):
    if isinstance(obj, torch.Tensor):
        grad_norm_squared = (torch.norm(obj.grad).item()) ** 2
        param_count = obj.numel()
        return  torch.sqrt(torch.tensor(grad_norm_squared)/param_count)
    else:
        total_grad_norm_squared = 0.0
        count = 0
        for param in obj.parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                total_grad_norm_squared += grad_norm ** 2
                count += param.numel()
        avg_grad_norm = torch.sqrt(torch.tensor(total_grad_norm_squared)/count)         
        return avg_grad_norm                                                    

# 计算模型参数范数
@torch.no_grad()
def log_avg_param_norm(obj):
    if isinstance(obj, torch.Tensor):
        param_norm_squared = (torch.norm(obj).item()) ** 2
        return  torch.sqrt(torch.tensor(param_norm_squared)/obj.numel())
    else:
        total_param_norm_squared = 0.0
        for param in obj.parameters():
            param_norm = torch.norm(param).item()
            total_param_norm_squared += param_norm**2
        avg_param_norm = torch.sqrt(torch.tensor(total_param_norm_squared)/sum(p.numel() for p in obj.parameters()))
        return avg_param_norm
    
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='dataset/imagenet_compatible_org')
parser.add_argument('--validation_path', type=str, default='stage1/imagenet_val')
parser.add_argument('--output_dir', type=str, default='results/Stage1_secret_size_56')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=10000)
parser.add_argument('--warm_up_steps', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--image_loss_scale', type=int, default=30)
parser.add_argument('--image_loss_ramp', type=int, default=2000)
parser.add_argument('--secret_loss_scale', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--start_step", type=int, default=0)
parser.add_argument('--validation_batch_size', type=int, default=2)
parser.add_argument('--max_val_samples', type=int, default=200)
parser.add_argument('--recordImg_freq', type=int, default=100)
parser.add_argument('--validation_freq', type=int, default=100)
parser.add_argument('--secret_size', type=int, default=56)      # --[debug] 48, 56, 96, 120 
parser.add_argument('--sd_model', type=str, default="HuggingFaceModels/stable-diffusion-2-1-base")
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--lpips_scale', type=float, default=0.25)
parser.add_argument('--lpips_ramp', type=int, default=4000)
parser.add_argument("--max_grad_norm", default=1e-2, type=float, help="Max gradient norm.")
parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="Weight decay to use.")
args = parser.parse_args()

checkpoints_path = f"{args.output_dir}/checkpoints"
saved_models_path = f"{args.output_dir}/saved_models"
log_path = f"{args.output_dir}/log"
os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(saved_models_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

def main():

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    logger = get_logger(filename=f'{log_path}/training_enc_dec.log', name='Stage1_training_size256')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    logger.info(f"Using {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")

    lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(device)
    lpips_alex.requires_grad_(False)

    train_dataset = ImageData(args.train_path, secret_size=args.secret_size, img_size=(args.image_size, args.image_size))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size * num_gpus,
        shuffle=True, 
        pin_memory=True,
        num_workers=4
    )

    validation_dataset = ImageData(args.validation_path, secret_size=args.secret_size, img_size=(args.image_size, args.image_size), num_samples=args.max_val_samples)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=args.validation_batch_size * num_gpus,
        shuffle=False, 
        pin_memory=True
    )

    if args.image_size == 256:
        sec_encoder = model.SecretEncoder_size256(args.secret_size)
        decoder = model.Extractor_forLatent_size256(args.secret_size)
    else:
        sec_encoder = model.SecretEncoder(secret_size=args.secret_size)
        decoder = model.Extractor_forLatent(secret_size=args.secret_size)
    
    
    # if args.pretrained_dir:
    #     decoder.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "decoder.pth")))
    #     sec_encoder.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "encoder.pth")))
    #     logger.info(f"Loaded pretrained models from {args.pretrained_dir}")
    
    if num_gpus > 1:
        sec_encoder = torch.nn.DataParallel(sec_encoder)
        decoder = torch.nn.DataParallel(decoder)
        logger.info(f"Using DataParallel on {num_gpus} GPUs")
    
    sec_encoder = sec_encoder.to(device)
    decoder = decoder.to(device)
    
    # 优化器和学习率调度器
    from itertools import chain
    params_to_optimize = [p for p in chain(sec_encoder.parameters(), decoder.parameters())]
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr= args.lr,    # * num_gpus,  # 根据GPU数量调整学习率
        weight_decay=args.adam_weight_decay,
    )
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up_steps, num_training_steps=args.num_steps)
    
    # VAE模型
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.eval()
    
    global_step = args.start_step
    min_loss = 10000
    
    iterator = iter(train_dataloader)
    while global_step < args.num_steps:
        sec_encoder.train()
        decoder.train()
        
        try:
            image_input, secret_input = next(iterator)
        except StopIteration:
            iterator = iter(train_dataloader)
            image_input, secret_input = next(iterator)

        image_input = image_input.to(device)
        secret_input = secret_input.to(device)
        
        # 计算损失权重（ramp策略：前N步权重线性增加，之后保持最大值）
        image_loss_scale = min(args.image_loss_scale * global_step / args.image_loss_ramp, args.image_loss_scale)
        lpips_scale = min(args.lpips_scale * global_step / args.lpips_ramp, args.lpips_scale)
        loss_scales = args.secret_loss_scale, image_loss_scale, lpips_scale
        
        # 计算损失
        # loss, secret_loss = model.build_model(secret_input, sec_encoder, decoder, image_input, loss_scales, args, global_step, vae, lpips_alex, device)
        loss, secret_loss = model.build_model_ybb(secret_input, sec_encoder, decoder, image_input, loss_scales, args, global_step, vae, lpips_alex, logger, device)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        
        # 验证
        if global_step % args.validation_freq == 0:
            decoder.eval()
            sec_encoder.eval()
            
            psnr_input_ls = []
            psnr_recons_ls = []
            
            acc_WM_ls = []
            blur_wm_acc = []
            noise_wm_acc = []
            jpeg_compress_wm_acc = []
            resize_wm_acc = []
            sharpness_wm_acc = []
            brightness_wm_acc = []
            contrast_wm_acc = []
            saturation_wm_acc = []
            
            distortion_list = ['identity', 'blur', 'noise', 'jpeg_compress', 'resize', 'sharpness', "brightness", "contrast", "saturation"]
            
            with torch.no_grad():
                for batch in validation_dataloader:
                    image_input, secret_input = batch
                    image_input = image_input.to(device)
                    secret_input = secret_input.to(device)
                    
                    for distortion in distortion_list:
                        avg_psnr_input, avg_psnr_recons, predict_acc_WM = model.validate_model(secret_input, sec_encoder, decoder, image_input, vae, distortion)
                        
                        if distortion == 'identity':
                            acc_WM_ls.append(predict_acc_WM)
                        elif distortion == 'resize':
                            resize_wm_acc.append(predict_acc_WM)
                        elif distortion == 'brightness':
                            brightness_wm_acc.append(predict_acc_WM)
                        elif distortion == 'contrast':
                            contrast_wm_acc.append(predict_acc_WM)
                        elif distortion == 'saturation':
                            saturation_wm_acc.append(predict_acc_WM)
                        elif distortion == 'blur':
                            blur_wm_acc.append(predict_acc_WM)
                        elif distortion == 'noise':
                            noise_wm_acc.append(predict_acc_WM)
                        elif distortion == 'jpeg_compress':
                            jpeg_compress_wm_acc.append(predict_acc_WM)
                        elif distortion == 'sharpness':
                            sharpness_wm_acc.append(predict_acc_WM)
                    
                    psnr_input_ls.append(avg_psnr_input)
                    psnr_recons_ls.append(avg_psnr_recons)
            
            # 计算平均值
            avg_acc_WM = torch.tensor(acc_WM_ls).mean().item()  # 无失真场景下的平均提取准确率
            avg_psnr_input = torch.tensor(psnr_input_ls).mean().item()  # 原始图像与水印图像的差异
            avg_psnr_recons = torch.tensor(psnr_recons_ls).mean().item()    # 原始图像与VAE自身重建图像的差异
            avg_acc_resize = torch.tensor(resize_wm_acc).mean().item() if resize_wm_acc else 0
            avg_acc_bright = torch.tensor(brightness_wm_acc).mean().item() if brightness_wm_acc else 0
            avg_acc_contrast = torch.tensor(contrast_wm_acc).mean().item() if contrast_wm_acc else 0
            avg_acc_saturation = torch.tensor(saturation_wm_acc).mean().item() if saturation_wm_acc else 0
            avg_acc_blur = torch.tensor(blur_wm_acc).mean().item() if blur_wm_acc else 0
            avg_acc_noise = torch.tensor(noise_wm_acc).mean().item() if noise_wm_acc else 0
            avg_acc_jpeg_compress = torch.tensor(jpeg_compress_wm_acc).mean().item() if jpeg_compress_wm_acc else 0
            avg_acc_sharpness = torch.tensor(sharpness_wm_acc).mean().item() if sharpness_wm_acc else 0
            
            # 记录验证结果
            logger.info(f"Validation at step {global_step}:")
            logger.info(f"  PSNR Input: {avg_psnr_input:.3f}, PSNR Recons: {avg_psnr_recons:.3f}")
            logger.info(f"  Accuracy (no distortion): {avg_acc_WM:.3f}")
            logger.info(f"  Accuracy (resize): {avg_acc_resize:.3f}")
            logger.info(f"  Accuracy (brightness): {avg_acc_bright:.3f}")
            logger.info(f"  Accuracy (contrast): {avg_acc_contrast:.3f}")
            logger.info(f"  Accuracy (saturation): {avg_acc_saturation:.3f}")
            logger.info(f"  Accuracy (blur): {avg_acc_blur:.3f}")
            logger.info(f"  Accuracy (noise): {avg_acc_noise:.3f}")
            logger.info(f"  Accuracy (jpeg): {avg_acc_jpeg_compress:.3f}")
            logger.info(f"  Accuracy (sharpness): {avg_acc_sharpness:.3f}")
        
        # 记录训练信息
        decoder_grad_norm = log_avg_gradient_norm(decoder)
        decoder_param_norm = log_avg_param_norm(decoder)
        encoder_grad_norm = log_avg_gradient_norm(sec_encoder)
        encoder_param_norm = log_avg_param_norm(sec_encoder)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Step {global_step}: Loss = {loss.item():.3f}, Secret loss = {secret_loss:.3f}")
        logger.info(f"  Decoder Grad Norm: {decoder_grad_norm:.6f}, Param Norm: {decoder_param_norm:.6f}")
        logger.info(f"  Encoder Grad Norm: {encoder_grad_norm:.6f}, Param Norm: {encoder_param_norm:.6f}")
        logger.info(f"  Learning Rate: {current_lr:.6e}")
        
        # 保存模型
        if global_step % args.save_freq == 0 and global_step >= 4000:
            save_dir = os.path.join(saved_models_path, f"step{global_step}_loss{loss.item():.3f}")
            os.makedirs(save_dir, exist_ok=True)
            
            if num_gpus > 1:
                torch.save(sec_encoder.module.state_dict(), f"{save_dir}/encoder.pth")
                torch.save(decoder.module.state_dict(), f"{save_dir}/decoder.pth")
            else:
                torch.save(sec_encoder.state_dict(), f"{save_dir}/encoder.pth")
                torch.save(decoder.state_dict(), f"{save_dir}/decoder.pth")
            
            logger.info(f"Saved models to {save_dir}")
        
        # 保存最佳模型
        if global_step > args.lpips_ramp and loss.item() < min_loss:
            min_loss = loss.item()
            
            if num_gpus > 1:
                torch.save(sec_encoder.module.state_dict(), os.path.join(checkpoints_path, "encoder_best_total_loss.pth"))
                torch.save(decoder.module.state_dict(), os.path.join(checkpoints_path, "decoder_best_total_loss.pth"))
            else:
                torch.save(sec_encoder.state_dict(), os.path.join(checkpoints_path, "encoder_best_total_loss.pth"))
                torch.save(decoder.state_dict(), os.path.join(checkpoints_path, "decoder_best_total_loss.pth"))
            
            logger.info(f"Saved best models with loss {min_loss:.3f}")
        
        global_step += 1
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
