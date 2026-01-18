"""
Ground-SAM 视觉特征提取模块
使用预训练的 Grounding DINO + SAM 模型进行特征提取,并冻结预训练模型
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from modelscope import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FrozenSAMBackbone(nn.Module):
    """
    冻结的SAM骨干网络,用于特征提取
    """
    def __init__(self, model_type: str = "vit_h", checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        super(FrozenSAMBackbone, self).__init__()
        
        self.model_type = model_type
        
        # 加载SAM模型
        if checkpoint_path is None:
            checkpoint_path = self._download_sam_model(model_type)
        
        # 统一设备选择
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(self.device)

        # 冻结所有预训练参数
        self._freeze_pretrained_weights()

        # 验证并修正设备
        self._validate_device()
        
        # 创建自动掩码生成器(用于无text_prompt的自动分割)
        # 注意:SamAutomaticMaskGenerator内部会使用self.sam,需要确保sam在正确设备上
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,      # 控制采样密度
            points_per_batch=64,     # 批处理大小
            pred_iou_thresh=0.86,    # 预测IoU阈值(降低以检测更多对象)
            stability_score_thresh=0.92,  # 稳定性分数阈值
            stability_score_offset=1.0,
            box_nms_thresh=0.7,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,  # 最小掩码区域面积
        )

        # 确保mask_generator使用的模型也在正确设备上
        if hasattr(self.mask_generator, 'model'):
            self.mask_generator.model = self.mask_generator.model.to(self.device)
        
        logger.info(f"SAM模型加载完成并冻结 ({model_type}), 设备: {device}")
    
    def _download_sam_model(self, model_type: str) -> str:
        """下载SAM模型权重"""
        import urllib.request
        
        model_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        
        checkpoint_url = model_urls.get(model_type, model_urls["vit_h"])
        model_dir = os.path.expanduser("~/.cache/segment_anything")
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, os.path.basename(checkpoint_url))
        
        if not os.path.exists(checkpoint_path):
            logger.info(f"正在下载SAM模型: {checkpoint_url}")
            urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
            logger.info(f"SAM模型已保存到: {checkpoint_path}")
        
        return checkpoint_path
    
    def _freeze_pretrained_weights(self):
        """冻结所有预训练参数"""
        for param in self.sam.parameters():
            param.requires_grad = False
        logger.info(f"SAM预训练权重已冻结 (设备: {next(self.sam.parameters()).device})")

    def _validate_device(self):
        """验证模型参数都在正确的设备上,不打印警告"""
        expected_device = torch.device(self.device)
        mismatches = 0
        for name, param in self.sam.named_parameters():
            if param.device != expected_device:
                # 尝试移动到正确设备
                param.data = param.data.to(expected_device)
                mismatches += 1

        if mismatches > 0:
            logger.info(f"已将 {mismatches} 个参数移动到 {self.device} 设备")
    
    def extract_image_features(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        提取图像特征
        Args:
            image: numpy数组图像 (H, W, 3)
        Returns:
            包含特征的字典
        """
        # 确保模型在正确的设备上
        if hasattr(self, 'device'):
            self.sam.to(self.device)

        # 创建predictor并设置图像
        predictor = SamPredictor(self.sam)
        # 确保predictor的模型也在正确的设备上
        if hasattr(self, 'device'):
            predictor.model = predictor.model.to(self.device)

        # 设置图像，确保输入数据也在正确的设备上
        predictor.set_image(image)

        # 获取图像嵌入
        features = {
            'original_size': predictor.original_size,
            'input_size': predictor.input_size,
        }

        # 获取特征
        features['image_embedding'] = predictor.features

        # 确保特征在正确的设备上
        if hasattr(self, 'device'):
            features['image_embedding'] = features['image_embedding'].to(self.device)

        # 获取位置编码(如果存在)
        try:
            # 不同的SAM版本可能有不同的属性访问方式
            if hasattr(predictor.model, 'pe_layer'):
                features['image_pe'] = predictor.model.pe_layer.positional_embedding
                if hasattr(self, 'device'):
                    features['image_pe'] = features['image_pe'].to(self.device)
            elif hasattr(predictor.model, 'sam') and hasattr(predictor.model.sam, 'pe_layer'):
                features['image_pe'] = predictor.model.sam.pe_layer.positional_embedding
                if hasattr(self, 'device'):
                    features['image_pe'] = features['image_pe'].to(self.device)
        except Exception as e:
            # 如果位置编码获取失败,不影响主要功能
            logger.debug(f"无法获取位置编码: {e}")

        return features
    
    def extract_mask_features(self, image: np.ndarray, boxes: List[np.ndarray]) -> List[torch.Tensor]:
        """
        提取给定边界框的掩码特征
        Args:
            image: numpy数组图像 (H, W, 3)
            boxes: 边界框列表,每个box格式为 [x0, y0, x1, y1]
        Returns:
            掩码特征列表
        """
        # 确保模型在正确的设备上
        if hasattr(self, 'device'):
            self.sam.to(self.device)

        # 创建predictor
        predictor = SamPredictor(self.sam)
        # 确保predictor的模型也在正确的设备上
        if hasattr(self, 'device'):
            predictor.model = predictor.model.to(self.device)
        predictor.set_image(image)

        mask_features = []
        for box in boxes:
            masks, iou_scores, low_res_masks = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(box)[None, :],
                multimask_output=False,
            )
            # 确保输出张量在正确的设备上
            if hasattr(self, 'device'):
                masks = masks.to(self.device)
                iou_scores = iou_scores.to(self.device)
                low_res_masks = low_res_masks.to(self.device)
            mask_features.append({
                'mask': masks[0],
                'iou_score': iou_scores[0],
                'low_res_mask': low_res_masks[0]
            })

        return mask_features
    
    def automatic_segmentation(self, image: np.ndarray) -> List[Dict]:
        """
        使用SAM自动掩码生成器进行自动分割(不需要text_prompt)
        Args:
            image: numpy数组图像 (H, W, 3)
        Returns:
            自动分割结果列表,每个结果包含: segmentation, area, bbox, predicted_iou等
        """
        # 确保模型在正确的设备上
        if hasattr(self, 'device'):
            self.sam.to(self.device)
            # 确保mask_generator的模型也在正确的设备上
            if hasattr(self.mask_generator, 'model'):
                self.mask_generator.model = self.mask_generator.model.to(self.device)
            # 确保predictor也在正确设备上
            if hasattr(self.mask_generator, 'predictor'):
                self.mask_generator.predictor.model = self.mask_generator.predictor.model.to(self.device)

        masks = self.mask_generator.generate(image)
        return masks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播,返回图像嵌入
        Args:
            x: 输入图像张量 (B, 3, H, W)
        Returns:
            图像嵌入特征
        """
        # 将张量转换为numpy数组
        if x.dim() == 4:
            # 批量处理
            features_list = []
            for i in range(x.shape[0]):
                image_np = x[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                features = self.extract_image_features(image_np)
                features_list.append(features['image_embedding'])
            return torch.stack(features_list)
        else:
            # 单张图像
            image_np = x.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            features = self.extract_image_features(image_np)
            return features['image_embedding']


class FrozenGroundingDINO(nn.Module):
    """
    冻结的Grounding DINO模型,用于文本引导的目标检测
    """
    def __init__(self, model_name: str = "IDEA-Research/grounding-dino-tiny", device: Optional[str] = None):
        super(FrozenGroundingDINO, self).__init__()
        
        self.model_name = model_name
        
        # 加载Grounding DINO模型
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        
        # 统一设备选择
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)

        # 冻结所有预训练参数
        self._freeze_pretrained_weights()

        # 验证并修正设备
        self._validate_device()

        logger.info(f"Grounding DINO模型加载完成并冻结, 设备: {device}, 实际设备: {next(self.model.parameters()).device}")
    
    def _freeze_pretrained_weights(self):
        """冻结所有预训练参数"""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Grounding DINO预训练权重已冻结")

    def _validate_device(self):
        """验证模型参数都在正确的设备上,不打印警告"""
        expected_device = torch.device(self.device)
        mismatches = 0
        for name, param in self.model.named_parameters():
            if param.device != expected_device:
                # 尝试移动到正确设备
                param.data = param.data.to(expected_device)
                mismatches += 1

        if mismatches > 0:
            logger.info(f"已将 {mismatches} 个参数移动到 {self.device} 设备")
    
    def detect_objects(self, image: Image.Image, text_prompt: str, 
                      threshold: float = 0.3, text_threshold: float = 0.25) -> Dict:
        """
        检测图像中的对象
        Args:
            image: PIL图像
            text_prompt: 文本提示
            threshold: 检测阈值
            text_threshold: 文本阈值
        Returns:
            检测结果字典,包含boxes, labels, scores
        """
        device = next(self.model.parameters()).device
        
        # 处理输入
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 后处理
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
        
        return results[0]
    
    def extract_detection_features(self, image: Image.Image, text_prompt: str) -> Dict:
        """
        提取检测特征
        Args:
            image: PIL图像
            text_prompt: 文本提示
        Returns:
            包含检测框、标签、分数和特征的字典
        """
        device = next(self.model.parameters()).device
        
        # 处理输入
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 提取特征
        detection_features = {
            'boxes': outputs['pred_boxes'],
            'logits': outputs['pred_logits'],
            'hidden_states': self.get_intermediate_features(outputs)
        }
        
        return detection_features
    
    def get_intermediate_features(self, outputs: Dict) -> Dict:
        """获取中间层特征"""
        features = {}
        
        if 'encoder_last_hidden_state' in outputs:
            features['encoder_features'] = outputs['encoder_last_hidden_state']
        
        if 'last_hidden_state' in outputs:
            features['decoder_features'] = outputs['last_hidden_state']
        
        return features
    
    def forward(self, image: Image.Image, text_prompt: str) -> Dict:
        """
        前向传播
        Args:
            image: PIL图像
            text_prompt: 文本提示
        Returns:
            检测结果
        """
        return self.detect_objects(image, text_prompt)


class GroundSAMFeatureExtractor(nn.Module):
    """
    Ground-SAM特征提取器
    结合Grounding DINO和SAM进行特征提取,所有预训练模型均被冻结
    """
    def __init__(self, sam_model_type: str = "vit_h", 
                 dino_model_name: str = "IDEA-Research/grounding-dino-tiny",
                 sam_checkpoint_path: Optional[str] = None):
        super(GroundSAMFeatureExtractor, self).__init__()
        
        # 初始化冻结的SAM
        # 统一设备选择
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sam_backbone = FrozenSAMBackbone(
            model_type=sam_model_type,
            checkpoint_path=sam_checkpoint_path,
            device=self.device
        )
        
        # 初始化冻结的Grounding DINO
        self.grounding_dino = FrozenGroundingDINO(
            model_name=dino_model_name,
            device=self.device
        )
        
        logger.info("Ground-SAM特征提取器初始化完成")
    
    def extract_features(self, image: np.ndarray, text_prompt: Optional[str] = None,
                        boxes: Optional[List[np.ndarray]] = None,
                        threshold: float = 0.2,
                        text_threshold: float = 0.2,
                        use_auto_segmentation: bool = False) -> Dict:
        """
        提取图像特征
        Args:
            image: numpy数组图像 (H, W, 3)
            text_prompt: 文本提示(可选)
            boxes: 边界框列表(可选)
            threshold: 检测阈值(默认0.2,更低的阈值可以检测更多对象)
            text_threshold: 文本阈值(默认0.2)
            use_auto_segmentation: 是否使用自动分割(不需要text_prompt)
        Returns:
            特征字典
        """
        # 提取全局图像特征
        image_features = self.sam_backbone.extract_image_features(image)

        features = {
            'image_features': image_features,
        }

        # 如果启用自动分割,使用SAM自动掩码生成器
        if use_auto_segmentation:
            auto_masks = self.sam_backbone.automatic_segmentation(image)
            features['auto_masks'] = auto_masks

            # 将自动掩码转换为mask_features格式
            if auto_masks:
                mask_features = []
                for mask_data in auto_masks:
                    mask_features.append({
                        'mask': mask_data['segmentation'],
                        'bbox': mask_data['bbox'],
                        'area': mask_data['area'],
                        'predicted_iou': mask_data.get('predicted_iou', 0)
                    })
                features['mask_features'] = mask_features

        # 如果提供了文本提示,使用Grounding DINO进行目标检测
        elif text_prompt is not None:
            pil_image = Image.fromarray(image)
            detection_results = self.grounding_dino.detect_objects(
                pil_image,
                text_prompt,
                threshold=threshold,
                text_threshold=text_threshold
            )
            features['detection_results'] = detection_results

            # 如果检测到目标,提取区域特征
            if len(detection_results['boxes']) > 0:
                boxes_np = detection_results['boxes'].cpu().numpy()
                mask_features = self.sam_backbone.extract_mask_features(image, boxes_np)
                features['mask_features'] = mask_features

        # 如果直接提供了边界框,提取掩码特征
        elif boxes is not None:
            mask_features = self.sam_backbone.extract_mask_features(image, boxes)
            features['mask_features'] = mask_features

        return features
    
    def extract_visual_embedding(self, image: np.ndarray) -> torch.Tensor:
        """
        提取视觉嵌入,用于后续的强化学习等任务
        Args:
            image: numpy数组图像 (H, W, 3)
        Returns:
            视觉嵌入张量
        """
        features = self.extract_image_features(image)
        return features['image_embedding']
    
    def get_feature_dim(self) -> Tuple[int, int, int]:
        """
        获取特征维度
        Returns:
            (C, H, W) 特征维度
        """
        # 使用一个虚拟图像获取特征维度
        dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
        features = self.sam_backbone.extract_image_features(dummy_image)
        embedding = features['image_embedding']
        
        # 处理不同维度的embedding
        if embedding.ndim == 3:
            # 格式: (H, W, C) -> (C, H, W)
            h, w, c = embedding.shape
            return c, h, w
        elif embedding.ndim == 4:
            # 格式: (B, H, W, C) 或 (B, C, H, W)
            if embedding.shape[0] == 1:  # 如果batch size为1
                embedding = embedding.squeeze(0)
                if embedding.ndim == 3:
                    h, w, c = embedding.shape
                    return c, h, w
            # 如果是(B, C, H, W)格式
            if embedding.shape[1] < embedding.shape[2] and embedding.shape[1] < embedding.shape[3]:
                c, h, w = embedding.shape[1], embedding.shape[2], embedding.shape[3]
                return c, h, w
            else:
                h, w, c = embedding.shape[1], embedding.shape[2], embedding.shape[3]
                return c, h, w
        else:
            # 其他情况，尝试返回默认值
            logger.warning(f"未知的embedding维度: {embedding.shape}, 使用默认值")
            return 256, 64, 64
    
    def forward(self, x: torch.Tensor, text_prompt: Optional[str] = None) -> Dict:
        """
        前向传播
        Args:
            x: 输入图像张量 (B, 3, H, W) 或 (3, H, W)
            text_prompt: 文本提示(可选)
        Returns:
            特征字典
        """
        if x.dim() == 4:
            # 批量处理
            features_list = []
            for i in range(x.shape[0]):
                image_np = x[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                features = self.extract_features(image_np, text_prompt)
                features_list.append(features)
            return {'batch_features': features_list}
        else:
            # 单张图像
            image_np = x.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            return self.extract_features(image_np, text_prompt)


# 缓存模型实例
_cached_extractor: Optional[GroundSAMFeatureExtractor] = None


def get_groundsam_extractor(sam_model_type: str = "vit_h", 
                           dino_model_name: str = "IDEA-Research/grounding-dino-tiny",
                           force_reload: bool = False) -> GroundSAMFeatureExtractor:
    """
    获取Ground-SAM特征提取器(单例模式)
    Args:
        sam_model_type: SAM模型类型
        dino_model_name: Grounding DINO模型名称
        force_reload: 是否强制重新加载
    Returns:
        GroundSAM特征提取器实例
    """
    global _cached_extractor
    
    if _cached_extractor is None or force_reload:
        logger.info("正在初始化Ground-SAM特征提取器...")
        _cached_extractor = GroundSAMFeatureExtractor(
            sam_model_type=sam_model_type,
            dino_model_name=dino_model_name
        )
        logger.info("Ground-SAM特征提取器初始化完成")
    else:
        logger.info("使用已缓存的Ground-SAM特征提取器")
    
    return _cached_extractor


def extract_features_from_image(image: np.ndarray, 
                                text_prompt: Optional[str] = None) -> Dict:
    """
    便捷函数:从图像提取特征
    Args:
        image: numpy数组图像 (H, W, 3)
        text_prompt: 文本提示(可选)
    Returns:
        特征字典
    """
    extractor = get_groundsam_extractor()
    return extractor.extract_features(image, text_prompt)


if __name__ == "__main__":
    # 测试代码
    import cv2
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 测试特征提取器
    print("测试Ground-SAM特征提取器...")
    extractor = get_groundsam_extractor()
    
    # 获取特征维度
    feature_dim = extractor.get_feature_dim()
    print(f"特征维度: {feature_dim}")
    
    # 测试图像特征提取
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    features = extractor.extract_features(test_image, "cat . dog .")
    print(f"提取到特征: {list(features.keys())}")
    
    print("测试完成!")
