import os
import torch
import numpy as np
import torch.optim as optim
import argparse
import enc_dec_model
from tqdm import tqdm
from diffusers import DDIMScheduler
from utils import CustomImageFolder, get_logger, img_to_DMlatents
from loss.loss import EnhancedLossProvider
from loss.pytorch_ssim import ssim
from others.wmdiffusion import WMDetectStableDiffusionPipeline
from others.utils import get_img_tensor, save_img, compute_psnr
import random
from contextlib import nullcontext
from torchvision.transforms import GaussianBlur, RandomResizedCrop, RandomHorizontalFlip
from torchvision.transforms.functional import adjust_contrast, adjust_saturation

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


def apply_random_attack(
    img_tensor: torch.Tensor, 
    prob: float = 0.5, 
    noise_std_range: tuple = (0.05, 0.15),
    blur_kernel_range: tuple = (3, 5),
    brightness_range: tuple = (0.8, 1.2),
    contrast_range: tuple = (0.8, 1.2),
    saturation_range: tuple = (0.8, 1.2),
    crop_scale_range: tuple = (0.8, 1.0),
    random_flip_prob: float = 0.3
) -> torch.Tensor:
    """
    增强版随机攻击函数：覆盖更多常见图像失真场景，适配鲁棒性训练需求
    :param img_tensor: 输入图像张量 (shape: [B, C, H, W] 或 [C, H, W])
    :param prob: 每种攻击独立触发的概率
    :param noise_std_range: 高斯噪声标准差范围（验证时用0.1，训练覆盖更宽）
    :param blur_kernel_range: 高斯模糊核大小范围（奇数）
    :param brightness_range: 亮度调整因子范围
    :param contrast_range: 对比度调整因子范围
    :param saturation_range: 饱和度调整因子范围
    :param crop_scale_range: 随机裁剪的面积比例范围
    :param random_flip_prob: 水平翻转的概率
    :return: 受攻击后的图像张量
    """
    # 确保输入是4维张量（适配单张图像[C,H,W]和批量图像[B,C,H,W]）
    if img_tensor.dim() == 3:
        attacked = img_tensor.unsqueeze(0).clone()  # [C,H,W] → [1,C,H,W]
    else:
        attacked = img_tensor.clone()
    B, C, H, W = attacked.shape

    # 1. 高斯噪声攻击（重点优化：匹配验证场景，增加强度多样性）
    if random.random() < prob:
        std = random.uniform(*noise_std_range)
        noise = torch.randn_like(attacked) * std
        attacked = attacked + noise
        attacked = torch.clamp(attacked, -1.0, 1.0)  # 保持图像张量范围（扩散模型常用[-1,1]）

    # 2. 高斯模糊攻击（优化：动态调整核大小和 sigma）
    if random.random() < prob:
        kernel_size = random.choice([k for k in range(*blur_kernel_range) if k % 2 == 1])
        sigma = random.uniform(0.1, 0.8)
        blur = GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
        attacked = blur(attacked)

    # 3. 亮度调整攻击（优化：扩大范围，避免过度失真）
    if random.random() < prob:
        factor = random.uniform(*brightness_range)
        attacked = attacked * factor
        attacked = torch.clamp(attacked, -1.0, 1.0)

    # 4. 对比度调整攻击（常见图像增强/失真场景）
    if random.random() < prob:
        factor = random.uniform(*contrast_range)
        attacked = adjust_contrast(attacked, factor)

    # 5. 饱和度调整攻击（针对彩色图像的鲁棒性）
    if random.random() < prob and C == 3:
        factor = random.uniform(*saturation_range)
        attacked = adjust_saturation(attacked, factor)

    # 6. 随机局部裁剪+Resize攻击（模拟图像裁剪/缩放场景）
    if random.random() < prob:
        scale = random.uniform(*crop_scale_range)
        crop = RandomResizedCrop(
            size=(H, W),
            scale=(scale, scale),
            ratio=(1.0, 1.0)
        )
        attacked = crop(attacked)

    # 7. 水平翻转攻击（几何失真，验证水印对像素位置变化的鲁棒性）
    if random.random() < random_flip_prob:
        flip = RandomHorizontalFlip(p=1.0)
        attacked = flip(attacked)

    if img_tensor.dim() == 3:
        attacked = attacked.squeeze(0)

    return attacked

def binary_search_theta(gt_img_tensor, wm_img_tensor, threshold, lower=0., upper=1., precision=1e-6, max_iter=1000):
    for i in range(max_iter):
        mid_theta = (lower + upper) / 2
        img_tensor = (gt_img_tensor - wm_img_tensor) * mid_theta + wm_img_tensor
        ssim_value = ssim(img_tensor, gt_img_tensor).item()

        if ssim_value <= threshold:
            lower = mid_theta
        else:
            upper = mid_theta
        if upper - lower < precision:
            break
    return lower

def compute_watermark_accuracy(decoder, img_tensor, secret_tensor, pipe):
    """计算水印提取准确率"""
    # 评估时使用确定性latent
    with nullcontext():  # 确保不启用推理模式
        img_latent = pipe.get_image_latents_ybb(img_tensor, sample=False, inference_mode=False)
    pred_secret = decoder(img_latent)
    pred_secret_binary = (pred_secret > 0.5).float()
    accuracy = (pred_secret_binary == secret_tensor).float().mean().item()
    return accuracy

def main(args=None):
    # 路径设置
    wm_tmp_path = '/public/yunbeibei/Watermarks/StrongMark_results/Stage1_secret_size_120/wm_tmp'
    os.makedirs(wm_tmp_path, exist_ok=True)
    wm_path = '/public/yunbeibei/Watermarks/StrongMark_results/Stage1_secret_size_120/wm'
    os.makedirs(wm_path, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'log')
    os.makedirs(log_path, exist_ok=True)
    # ckpt_path = os.path.join(args.output_dir, 'ckpt')
    # os.makedirs(ckpt_path, exist_ok=True)

    logger = get_logger(filename=f'{log_path}/training_init_latent_secret_120.log', name='Stage2-1_training_size256_secret_120')
    logger.iter_count = 0

    # 加载模型
    pretrained_dir = args.enc_dec_model_dir
    if args.image_size == 256:
        encoder = enc_dec_model.SecretEncoder_size256(secret_size=args.secret_length)
        decoder = enc_dec_model.Extractor_forLatent_size256(secret_size=args.secret_length)
    else:
        encoder = enc_dec_model.SecretEncoder(secret_size=args.secret_length)
        decoder = enc_dec_model.Extractor_forLatent(secret_size=args.secret_length)
    
    decoder.load_state_dict(torch.load(os.path.join(pretrained_dir, "decoder_best_total_loss.pth")))
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(os.path.join(pretrained_dir, "encoder_best_total_loss.pth")))
    encoder.eval()
    encoder = encoder.to(device)

    scheduler = DDIMScheduler.from_pretrained(args.sd_model, subfolder="scheduler")
    pipe = WMDetectStableDiffusionPipeline.from_pretrained(args.sd_model, scheduler=scheduler).to(device)
    pipe.set_progress_bar_config(disable=True)

    # 准备数据和水印
    dataset = CustomImageFolder(data_dir=args.data_dir, data_cnt=100)  # --[debug]

    np.random.seed(42)
    secret_input = np.random.binomial(1, 0.5, args.secret_length)
    secret_str = ''.join(str(bit) for bit in secret_input)
    logger.info(f'Fixed secret: {secret_str}')
    secret_tensor = torch.from_numpy(secret_input).float().unsqueeze(0).to(device)
    residual_latent = encoder(secret_tensor)
    
    # secret_info = {
    #     'secret_tensor': secret_tensor,
    #     'secret_array': secret_input,
    #     'secret_length': args.secret_length,
    #     'generation_info': 'binomial distribution p=0.5'
    # }
    # secret_save_path = f"{ckpt_path}/fixed_secret.pt"
    # torch.save(secret_info, secret_save_path)
    # logger.info(f'Saved fixed secret to: {secret_save_path}')

    # 训练
    total_accuracy_before = 0.0
    total_accuracy_after = 0.0
    total_robust_accuracy = 0.0
    image_count = 0

    for img_path in tqdm(dataset.filenames):
        image_count += 1
        img_name = os.path.basename(img_path)
        logger.info(f'\n----- {img_name} -----')
        gt_img_tensor = get_img_tensor(img_path, img_size=args.image_size, device=device)

        # 步骤1: 嵌入水印残差
        with nullcontext():
            gt_img_latent = pipe.get_image_latents_ybb(gt_img_tensor, sample=False, inference_mode=False)
        wm_img_latent = gt_img_latent + residual_latent.detach()

        # 步骤2: 获取初始噪声 - DDIM反转
        empty_text_embeddings = pipe.get_text_embedding('')
        init_latents_approx = pipe.forward_diffusion(
            latents = wm_img_latent, 
            text_embeddings = empty_text_embeddings,
            guidance_scale = 1.0,
            num_inference_steps = 50    # --[ablation study] T = 0, 1, 10, 30, 50
        )

        # 步骤3: 准备训练
        init_latent_wm = init_latents_approx.detach().clone()
        init_latent_wm.requires_grad = True

        optimizer = optim.AdamW([init_latent_wm], lr=0.005, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

        totalLoss = EnhancedLossProvider(args.loss_weights, device)

        # 步骤4: 训练init_latent_wm，平衡图像质量和水印性能
        best_loss = float('inf')
        best_latent = None
        
        for j in range(args.train_iters):
            # 生成带水印的图像
            pred_img_tensor = pipe(
                                prompt='', 
                                guidance_scale=1.0, 
                                num_inference_steps=50,     # --[ablation study] T = 0, 1, 10, 30, 50
                                output_type='tensor',
                                use_trainable_latents=True, 
                                init_latents=init_latent_wm
                                ).images
            pred_img_tensor.requires_grad_(True)  # 确保生成图像可求导
            
            # 获取潜在变量用于水印提取
            with nullcontext():  # 禁用推理模式
                pred_img_latent = pipe.get_image_latents_ybb(pred_img_tensor, sample=True, inference_mode=False)
            pred_img_latent = pred_img_latent.clone().requires_grad_(True)
            
            # 对抗训练：随机应用轻微攻击来增强鲁棒性
            if j % 5 == 0 and j > 0 and args.robust_training:  # 每5步进行一次鲁棒性训练
                attacked_img = apply_random_attack(pred_img_tensor, 
                                                   prob=0.4,  # 每种攻击40%概率触发，单次可能叠加1-2种攻击
                                                    )
                with nullcontext():
                    attacked_latent = pipe.get_image_latents_ybb(attacked_img, sample=True, inference_mode=False)
                attacked_latent = attacked_latent.clone().requires_grad_(True)

                decoder.train()
                loss = totalLoss(pred_img_tensor, gt_img_tensor, attacked_latent, decoder, secret_tensor, logger)
                decoder.eval()
            else:
                # 正常训练：同时考虑图像质量和水印提取
                decoder.train()
                loss = totalLoss(pred_img_tensor, gt_img_tensor, pred_img_latent, decoder, secret_tensor, logger)
                decoder.eval()
            
            if loss < best_loss:
                best_loss = loss
                best_latent = init_latent_wm.detach().clone()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([init_latent_wm], max_norm=1.0)
            optimizer.step()
            scheduler.step(loss)
            
            if (j+1) % 20 == 0:
                with torch.no_grad():
                    acc = compute_watermark_accuracy(decoder, pred_img_tensor, secret_tensor, pipe)
                    logger.info(f'Iteration {j+1}, Loss: {loss.item():.4f}, Watermark Accuracy: {acc:.4f}')
        
        # 使用最佳潜在变量生成最终图像
        with torch.no_grad():
            pred_img_tensor = pipe(
                                prompt='', 
                                guidance_scale=1.0, 
                                num_inference_steps=50,      # --[ablation study] T = 0, 1, 10, 30, 50
                                output_type='tensor',
                                use_trainable_latents=True, 
                                init_latents=best_latent
                                ).images
            
            path = os.path.join(wm_tmp_path, img_name)
            save_img(path, pred_img_tensor, pipe)
        
        torch.cuda.empty_cache()

        # 评估结果
        wm_img_path = os.path.join(wm_tmp_path, img_name)
        wm_img_tensor = get_img_tensor(wm_img_path, args.image_size, device)
        
        # 计算图像质量指标
        ssim_value = ssim(wm_img_tensor, gt_img_tensor).item()
        psnr_value = compute_psnr(wm_img_tensor, gt_img_tensor)
        
        # 计算水印提取准确率
        wm_accuracy = compute_watermark_accuracy(decoder, wm_img_tensor, secret_tensor, pipe)
        total_accuracy_before += wm_accuracy
        
        logger.info(f'Before enhancement - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}, '
                   f'Watermark Accuracy: {wm_accuracy:.4f}')

        # 步骤5: 自适应增强后处理
        optimal_theta = binary_search_theta(gt_img_tensor, wm_img_tensor, args.ssim_threshold, precision=0.01)
        img_tensor = (gt_img_tensor - wm_img_tensor) * optimal_theta + wm_img_tensor

        # 增强后的评估
        ssim_value_after = ssim(img_tensor, gt_img_tensor).item()
        psnr_value_after = compute_psnr(img_tensor, gt_img_tensor)
        wm_accuracy_after = compute_watermark_accuracy(decoder, img_tensor, secret_tensor, pipe)
        total_accuracy_after += wm_accuracy_after
        
        # 测试鲁棒性
        attacked_img = apply_random_attack(img_tensor)
        robust_accuracy = compute_watermark_accuracy(decoder, attacked_img, secret_tensor, pipe)
        total_robust_accuracy += robust_accuracy
        
        logger.info(f'After enhancement  - PSNR: {psnr_value_after:.2f}, SSIM: {ssim_value_after:.3f}, '
                   f'Watermark Accuracy: {wm_accuracy_after:.4f}, Robust Accuracy: {robust_accuracy:.4f}')

        path = os.path.join(wm_path, img_name)
        save_img(path, img_tensor, pipe)
    
    # 计算平均指标
    avg_accuracy_before = total_accuracy_before / image_count
    avg_accuracy_after = total_accuracy_after / image_count
    avg_robust_accuracy = total_robust_accuracy / image_count
    
    logger.info(f'\n===== Final Results =====')
    logger.info(f'Average Watermark Accuracy (Before Enhancement): {avg_accuracy_before:.4f}')
    logger.info(f'Average Watermark Accuracy (After Enhancement): {avg_accuracy_after:.4f}')
    logger.info(f'Average Robust Accuracy: {avg_robust_accuracy:.4f}')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--enc_dec_model_dir", default='results/Stage1_secret_size_120/checkpoints', type=str)
    parser.add_argument("--sd_model", default="HuggingFaceModels/stable-diffusion-2-1-base", type=str)
    parser.add_argument("--data_dir", default="dataset/imagenet_compatible_org", type=str)
    parser.add_argument("--secret_length", default=120, type=int)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--output_dir", default="results/Stage1_secret_size_120", type=str)
    parser.add_argument('--train_iters', default=120, type=int)
    parser.add_argument('--ssim_threshold', default=0.95, type=float)
    parser.add_argument('--loss_weights', default=[5.0, 0.5, 3.0, 3.0], type=list, 
                        help='调整损失权重: L2 loss, watson-vgg loss, SSIM loss, watermark BCE loss')   # [5.0, 0.1, 1.0, 3.0]
    parser.add_argument('--robust_training', action='store_true', default=True,
                        help='启用鲁棒性训练，增强水印抗攻击能力')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args=args)
