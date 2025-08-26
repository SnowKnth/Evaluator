#当前的 check_page_image_match 函数使用 imagehash.average_hash 来比较图片相似性。这个方法有一些局限性，特别是在处理不同分辨率的图片时。以下是一些更优的算法建议：
# 主要改进点：

# 结构相似性指数 (SSIM)：比哈希方法更能捕捉图像的结构信息
# 多算法组合：结合多种相似性度量方法提高准确性
# 智能缩放：自动调整图片到相同尺寸进行比较
# 直方图比较：对颜色分布敏感，适合检测UI变化
# 改进的感知哈希：使用多种哈希算法组合
# 使用建议：

# 对于UI自动化测试，推荐使用 "ssim" 或 "combined" 方法
# "combined" 方法更稳健但计算开销更大
# 可以根据具体需求调整相似性阈值
# 需要安装额外依赖：

# pip install opencv-python scikit-image
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def check_page_image_match_improved(
    gr_screenshot_path: str, 
    exec_screenshot_path: str, 
    similarity_threshold: float = 0.95,
    method: str = "ssim"
) -> bool:
    """
    改进的页面图片匹配算法，支持多种相似性度量方法
    
    Args:
        gr_screenshot_path: 参考截图路径
        exec_screenshot_path: 执行截图路径
        similarity_threshold: 相似性阈值 (0-1)
        method: 匹配方法 ("ssim", "template", "histogram", "perceptual", "combined")
    
    Returns:
        布尔值表示图片是否匹配
    """
    
    if method == "ssim":
        return _check_ssim_match(gr_screenshot_path, exec_screenshot_path, similarity_threshold)
    elif method == "template":
        return _check_template_match(gr_screenshot_path, exec_screenshot_path, similarity_threshold)
    elif method == "histogram":
        return _check_histogram_match(gr_screenshot_path, exec_screenshot_path, similarity_threshold)
    elif method == "perceptual":
        return _check_perceptual_match(gr_screenshot_path, exec_screenshot_path, similarity_threshold)
    elif method == "combined":
        return _check_combined_match(gr_screenshot_path, exec_screenshot_path, similarity_threshold)
    else:
        raise ValueError(f"Unsupported method: {method}")

def _resize_to_same_size(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """将两个图片调整到相同尺寸"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 选择较小的尺寸作为目标尺寸
    target_h = min(h1, h2)
    target_w = min(w1, w2)
    
    img1_resized = cv2.resize(img1, (target_w, target_h))
    img2_resized = cv2.resize(img2, (target_w, target_h))
    
    return img1_resized, img2_resized

def _check_ssim_match(gr_path: str, exec_path: str, threshold: float) -> bool:
    """使用结构相似性指数 (SSIM) 进行匹配"""
    img1 = cv2.imread(gr_path)
    img2 = cv2.imread(exec_path)
    
    if img1 is None or img2 is None:
        logging.error(f"Failed to load images: {gr_path}, {exec_path}")
        return False
    
    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 调整到相同尺寸
    gray1_resized, gray2_resized = _resize_to_same_size(gray1, gray2)
    
    # 计算SSIM
    similarity = ssim(gray1_resized, gray2_resized)
    
    match = similarity >= threshold
    logging.info(f"[SSIM] similarity: {similarity:.4f}, match: {match}")
    return match

def _check_template_match(gr_path: str, exec_path: str, threshold: float) -> bool:
    """使用模板匹配进行相似性检测"""
    img1 = cv2.imread(gr_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(exec_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return False
    
    # 调整尺寸
    img1_resized, img2_resized = _resize_to_same_size(img1, img2)
    
    # 模板匹配
    result = cv2.matchTemplate(img1_resized, img2_resized, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    match = max_val >= threshold
    logging.info(f"[Template] similarity: {max_val:.4f}, match: {match}")
    return match

def _check_histogram_match(gr_path: str, exec_path: str, threshold: float) -> bool:
    """使用直方图比较进行匹配"""
    img1 = cv2.imread(gr_path)
    img2 = cv2.imread(exec_path)
    
    if img1 is None or img2 is None:
        return False
    
    # 调整尺寸
    img1_resized, img2_resized = _resize_to_same_size(img1, img2)
    
    # 计算HSV直方图
    hsv1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2HSV)
    
    hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    
    # 计算直方图相关性
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    match = similarity >= threshold
    logging.info(f"[Histogram] similarity: {similarity:.4f}, match: {match}")
    return match

def _check_perceptual_match(gr_path: str, exec_path: str, similarity_bound: int = 5) -> bool:
    """改进的感知哈希匹配，使用多种哈希算法组合"""
    with Image.open(gr_path) as img1, Image.open(exec_path) as img2:
        # 使用多种哈希算法
        avg_hash1 = imagehash.average_hash(img1)
        avg_hash2 = imagehash.average_hash(img2)
        
        phash1 = imagehash.phash(img1)
        phash2 = imagehash.phash(img2)
        
        dhash1 = imagehash.dhash(img1)
        dhash2 = imagehash.dhash(img2)
        
        # 计算多个哈希距离
        avg_dist = avg_hash1 - avg_hash2
        p_dist = phash1 - phash2
        d_dist = dhash1 - dhash2
        
        # 综合评分（所有距离都要小于阈值）
        match = all(dist <= similarity_bound for dist in [avg_dist, p_dist, d_dist])
        
        logging.info(
            f"[Perceptual] avg_dist: {avg_dist}, p_dist: {p_dist}, d_dist: {d_dist}, match: {match}"
        )
        return match

def _check_combined_match(gr_path: str, exec_path: str, threshold: float) -> bool:
    """组合多种方法的综合匹配"""
    # 使用多种方法进行评估
    ssim_result = _check_ssim_match(gr_path, exec_path, 0.8)
    hist_result = _check_histogram_match(gr_path, exec_path, 0.8)
    perceptual_result = _check_perceptual_match(gr_path, exec_path, 8)
    
    # 至少两种方法通过才认为匹配
    match_count = sum([ssim_result, hist_result, perceptual_result])
    match = match_count >= 2
    
    logging.info(
        f"[Combined] SSIM: {ssim_result}, Hist: {hist_result}, Perceptual: {perceptual_result}, match: {match}"
    )
    return match

# 为了向后兼容，保留原函数并改进
def check_page_image_match(
    gr_screenshot_path: str, 
    exec_screenshot_path: str, 
    image_similarity_bound: Optional[int] = 1
) -> bool:
    """
    改进的页面图片匹配，默认使用SSIM方法
    """
    # 可以选择使用新的方法
    return check_page_image_match_improved(
        gr_screenshot_path, 
        exec_screenshot_path, 
        similarity_threshold=0.9,
        method="ssim"  # 或者 "combined" 获得更好的鲁棒性
    )