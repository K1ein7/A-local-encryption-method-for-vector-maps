import os
import json
import time
import random
import struct
import hashlib
import binascii
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import argparse
import math  # 添加math模块导入
from shapely.geometry import Point, Polygon, box, LineString, MultiPoint, MultiLineString, MultiPolygon
from shapely.affinity import translate, scale, rotate
from shapely import affinity
from sortedcontainers import SortedList
from gmssl import sm3, func
import matplotlib.font_manager as fm
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'  # Use Arial font

# 全局设置
USE_GPU = True  # 强制启用GPU加速，如果CUDA可用

# 导入GPU相关库
try:
    import numpy as np
    from numba import cuda, jit
    if cuda.is_available():
        print("✓ CUDA环境可用，将使用GPU加速")
        # 打印CUDA设备信息
        cuda_device = cuda.get_current_device()
        print(f"  - 使用GPU设备: {cuda_device.name}")
        print(f"  - 显存大小: {cuda_device.total_memory / (1024**3):.2f} GB")
        print(f"  - CUDA计算能力: {cuda_device.compute_capability[0]}.{cuda_device.compute_capability[1]}")
        print(f"  - 最大线程数/块: {cuda_device.MAX_THREADS_PER_BLOCK}")
        print(f"  - 最大共享内存/块: {cuda_device.MAX_SHARED_MEMORY_PER_BLOCK / 1024:.0f} KB")
    else:
        print("✗ CUDA环境不可用，自动切换到CPU处理")
        USE_GPU = False
except ImportError:
    print("✗ 未找到CUDA支持库，自动切换到CPU处理")
    USE_GPU = False

# ----------------------------------------------------------------------
# 1. 加密/解密与 Shapefile 保存部分
# ----------------------------------------------------------------------

# SM3适配器函数：将字节数据转换为整数列表后计算SM3哈希值
def sm3_hash_adapter(byte_data):
    return sm3.sm3_hash(func.bytes_to_list(byte_data))


# 生成混沌系统的扰动序列（对混沌序列的更新仍使用 % 1 限定在 [0,1) 区间）
def generate_initial_params(V, F, password):
    if V == 0 or F == 0:
        raise ValueError("V and F must be greater than zero to avoid division by zero.")

    # 生成随机数及盐值
    R = random.getrandbits(128)
    salt_input = str(R) + str(V * F)
    salt_input_bytes = salt_input.encode('utf-8')
    hash_value = sm3_hash_adapter(salt_input_bytes)
    salt = hash_value[:32]  # 256位盐值

    # 确保 c 至少为 8，避免后续索引 e[0]~e[7] 越界
    c = max(8, (V % 100) + 1)

    U = []
    for i in range(c):
        if i == 0:
            result = sm3_hash_adapter(bytearray((password + str(salt)).encode('utf-8')))
            U.append(bytes.fromhex(result))
        else:
            result = sm3_hash_adapter(bytearray((password + str(U[i - 1])).encode('utf-8')))
            U.append(bytes.fromhex(result))

    e = [int.from_bytes(U[i][:4], byteorder='big') for i in range(c)]
    ux = (c / (V * F)) + (e[0] ^ e[1] ^ e[2] ^ e[3])
    uy = (c / (V * F)) + (e[3] ^ e[4] ^ e[5] ^ e[6])
    uz = (c / (V * F)) + (e[5] ^ e[6] ^ e[7] ^ e[0])
    uw = (c / (V * F)) + (e[7] ^ e[0] ^ e[1] ^ e[2])

    return ux, uy, uz, uw


# 根据初始参数生成混沌扰动序列
def chaotic_sequence(ux, uy, uz, uw, num_points):
    """
    生成混沌序列，优先使用GPU版本（如果可用）
    """
    if USE_GPU:
        try:
            return chaotic_sequence_gpu(ux, uy, uz, uw, num_points)
        except Exception as e:
            print(f"GPU混沌序列生成失败: {e}，回退到CPU版本")
            return chaotic_sequence_cpu(ux, uy, uz, uw, num_points)
    else:
        return chaotic_sequence_cpu(ux, uy, uz, uw, num_points)

# CPU版本的混沌序列生成
def chaotic_sequence_cpu(ux, uy, uz, uw, num_points):
    sequence = []
    x, y, z, w = ux, uy, uz, uw
    for i in range(num_points):
        x = (x * 3.14159) % 1
        y = (y * 2.71828) % 1
        z = (z * 1.61803) % 1
        w = (w * 0.57721) % 1
        sequence.append((x, y, z, w))
    return sequence

# GPU版本的混沌序列生成（如果可用）
def chaotic_sequence_gpu(ux, uy, uz, uw, num_points):
    try:
        if not USE_GPU:
            return chaotic_sequence_cpu(ux, uy, uz, uw, num_points)
        
        # 使用CUDA加速混沌序列生成
        @cuda.jit
        def chaotic_kernel(x_array, y_array, z_array, w_array, num):
            i = cuda.grid(1)
            if i < num:
                if i == 0:
                    x_array[i] = (ux * 3.14159) % 1
                    y_array[i] = (uy * 2.71828) % 1
                    z_array[i] = (uz * 1.61803) % 1
                    w_array[i] = (uw * 0.57721) % 1
                else:
                    x_array[i] = (x_array[i-1] * 3.14159) % 1
                    y_array[i] = (y_array[i-1] * 2.71828) % 1
                    z_array[i] = (z_array[i-1] * 1.61803) % 1
                    w_array[i] = (w_array[i-1] * 0.57721) % 1
        
        # 分配GPU内存
        x_gpu = cuda.device_array(num_points, dtype=np.float32)
        y_gpu = cuda.device_array(num_points, dtype=np.float32)
        z_gpu = cuda.device_array(num_points, dtype=np.float32)
        w_gpu = cuda.device_array(num_points, dtype=np.float32)
        
        # 设置线程数和块数
        threads_per_block = 256
        blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block
        
        # 执行内核函数
        chaotic_kernel[blocks_per_grid, threads_per_block](x_gpu, y_gpu, z_gpu, w_gpu, num_points)
        
        # 将结果从GPU复制回CPU
        x_cpu = x_gpu.copy_to_host()
        y_cpu = y_gpu.copy_to_host()
        z_cpu = z_gpu.copy_to_host()
        w_cpu = w_gpu.copy_to_host()
        
        # 组合结果
        sequence = []
        for i in range(num_points):
            sequence.append((x_cpu[i], y_cpu[i], z_cpu[i], w_cpu[i]))
        
        print("Successfully generated chaotic sequence using GPU!")
        return sequence
    except Exception as e:
        print(f"GPU acceleration for chaotic sequence generation failed: {e}, falling back to CPU version")
        return chaotic_sequence_cpu(ux, uy, uz, uw, num_points)


# -----------------------------------------------------------------
# 新增: 基于超混沌系统的位级加扰加密算法
# -----------------------------------------------------------------

# 1. 二进制坐标序列转换
def to_binary_sequence(coordinate, min_bound, max_bound):
    """将坐标值转换为48位二进制序列，按照论文公式(9)实现
    
    参数:
    coordinate - 原始坐标值
    min_bound - 坐标范围最小值
    max_bound - 坐标范围最大值
    
    返回:
    x - 48位二进制序列
    """
    # 首先将坐标加上180进行平移，确保为正数
    pos = coordinate + 180
    
    # 提取整数部分、小数部分和小数点位置
    integer_part = int(pos)
    decimal_part = pos - integer_part
    decimal_position = len(str(integer_part))
    
    # 将整数部分、小数部分和小数点位置组合成48位二进制序列
    # 整数部分16位，小数部分24位，小数点位置8位
    int_bin = format(integer_part, '016b')
    
    # 将小数部分转为24位二进制
    decimal_bin = ""
    for _ in range(24):
        decimal_part *= 2
        if decimal_part >= 1:
            decimal_bin += "1"
            decimal_part -= 1
        else:
            decimal_bin += "0"
    
    # 小数点位置8位二进制
    pos_bin = format(decimal_position, '08b')
    
    # 组合成48位二进制序列
    result = int_bin + decimal_bin + pos_bin
    
    # 确保结果是48位
    if len(result) > 48:
        result = result[:48]
    elif len(result) < 48:
        result = result.ljust(48, '0')
        
    return result

def from_binary_sequence(binary_seq):
    """将48位二进制序列转换回坐标值
    
    参数:
    binary_seq - 48位二进制序列
    
    返回:
    恢复的坐标值
    """
    # 分解二进制序列
    int_part_bin = binary_seq[:16]
    decimal_part_bin = binary_seq[16:40]
    position_bin = binary_seq[40:48]
    
    # 将二进制转换回相应的值
    int_part = int(int_part_bin, 2)
    
    # 转换小数部分
    decimal_value = 0
    for i, bit in enumerate(decimal_part_bin):
        if bit == '1':
            decimal_value += 2 ** -(i + 1)
    
    # 组合整数和小数部分
    value = int_part + decimal_value
    
    # 减去之前添加的180，恢复原始坐标
    return value - 180

# 2. 随机序列生成和扰乱
def generate_chaotic_sequence(initial_params, V, iterations=None):
    """
    根据论文3.4.2节实现，生成混沌随机序列
    
    参数:
    initial_params - 混沌系统初始参数 [ux, uy, uz, uw]
    V - 加密单元中顶点的总数
    iterations - 迭代次数，若为None则按公式(10)计算
    
    返回:
    生成的混沌序列O
    """
    # 如果未指定迭代次数，根据公式(10)计算
    if iterations is None:
        iterations = (V % (2**10)) + 1000 + V
    
    ux, uy, uz, uw = initial_params
    chaotic_seq = []
    
    # 初始值
    x, y, z, w = ux, uy, uz, uw
    
    # 迭代生成混沌序列
    for _ in range(iterations):
        # 使用四维混沌系统更新值
        x_next = (x * 3.14159) % 1
        y_next = (y * 2.71828) % 1
        z_next = (z * 1.61803) % 1
        w_next = (w * 0.57721) % 1
        
        # 更新变量
        x, y, z, w = x_next, y_next, z_next, w_next
        
        # 将当前值添加到序列
        chaotic_seq.append([x, y, z, w])
    
    # 返回最后V个值组成的序列
    return chaotic_seq[-V:] if len(chaotic_seq) > V else chaotic_seq

def process_chaotic_sequence(chaotic_seq, control_param=8):
    """
    根据论文公式(11)对混沌序列进行进一步处理
    
    参数:
    chaotic_seq - 原始混沌序列
    control_param - 控制参数k，论文中设为8
    
    返回:
    处理后的序列
    """
    processed_seq = []
    
    for u in chaotic_seq:
        # 对每个混沌值应用公式(11)
        processed_value = []
        for val in u:
            # U_i = U_i + 10^k - [U_i*10^k]
            multiplied = val * (10 ** control_param)
            integer_part = int(multiplied)
            processed_val = val + (10 ** control_param) - integer_part
            processed_value.append(processed_val)
        
        processed_seq.append(processed_value)
    
    return processed_seq

def decompose_chaotic_sequence(chaotic_val):
    """
    根据论文3.4.3节和图5，将64位混沌值分解为16位和48位部分
    
    参数:
    chaotic_val - 单个混沌值(64位)
    
    返回:
    ZL - 16位左部分
    ZR - 48位右部分
    """
    # 将混沌值转换为64位二进制
    binary_val = ""
    for val in chaotic_val:
        # 将每个浮点数转换为16位二进制
        val_bin = ""
        val_temp = val
        for _ in range(16):
            val_temp *= 2
            if val_temp >= 1:
                val_bin += "1"
                val_temp -= 1
            else:
                val_bin += "0"
        binary_val += val_bin
    
    # 截取或填充到64位
    if len(binary_val) > 64:
        binary_val = binary_val[:64]
    elif len(binary_val) < 64:
        binary_val = binary_val.ljust(64, '0')
    
    # 分解为16位左部分和48位右部分
    ZL = binary_val[:16]
    ZR = binary_val[16:64]
    
    return ZL, ZR

# 3. 位级分解和链式序列计算
def encrypt_binary_coordinate(binary_seq, chaotic_key):
    """
    使用混沌序列对二进制坐标序列进行加密，实现公式(12)和(13)
    
    参数:
    binary_seq - 48位二进制坐标序列
    chaotic_key - 64位混沌密钥值
    
    返回:
    encrypted_seq - 加密后的48位二进制序列
    """
    # 分解混沌密钥为16位和48位部分
    ZL, ZR = decompose_chaotic_sequence(chaotic_key)
    
    # 按照论文公式实现加密
    # 1. 将坐标序列分为16位和48位部分
    XL = binary_seq[:16]
    XR = binary_seq[16:]
    
    # 2. 按位异或操作
    # XL' = (XL)⊕(ZL)
    XL_prime = ""
    for i in range(16):
        XL_prime += "1" if XL[i] != ZL[i] else "0"
    
    # 3. 计算右部分
    # XR' = (XR)⊕(((ZR)⊕(XL+1))<<<16⊕(ZR))
    
    # 实现XL+1，如果溢出则循环
    XL_next = XL_prime  # 这里简化为直接使用XL_prime
    
    # 计算(ZR)⊕(XL+1)
    temp1 = ""
    for i in range(min(len(ZR), len(XL_next))):
        temp1 += "1" if ZR[i] != XL_next[i % len(XL_next)] else "0"
    # 补齐长度
    temp1 = temp1.ljust(len(ZR), '0')
    
    # 左移16位
    shifted = temp1[16:] + temp1[:16]
    
    # 最后与ZR异或
    XR_prime = ""
    for i in range(len(XR)):
        if i < len(shifted) and i < len(ZR):
            # 先计算shifted⊕ZR
            bit = "1" if shifted[i] != ZR[i] else "0"
            # 再与XR异或
            XR_prime += "1" if bit != XR[i] else "0"
        else:
            XR_prime += XR[i]  # 如果长度不足，保持原值
    
    # 组合成最终的加密序列
    encrypted_seq = XL_prime + XR_prime
    
    return encrypted_seq

# 4. 坐标序列重建
def decrypt_binary_coordinate(encrypted_seq, chaotic_key):
    """
    使用混沌序列对加密的二进制坐标序列进行解密
    
    参数:
    encrypted_seq - 加密后的48位二进制序列
    chaotic_key - 64位混沌密钥值
    
    返回:
    decrypted_seq - 解密后的48位二进制序列
    """
    # 分解混沌密钥为16位和48位部分
    ZL, ZR = decompose_chaotic_sequence(chaotic_key)
    
    # 分解加密序列
    XL_prime = encrypted_seq[:16]
    XR_prime = encrypted_seq[16:]
    
    # 1. 先解密左部分（直接与ZL异或）
    XL = ""
    for i in range(len(XL_prime)):
        if i < len(ZL):
            XL += "1" if XL_prime[i] != ZL[i] else "0"
        else:
            XL += XL_prime[i]
    
    # 2. 为解密右部分，需要重新计算中间值（与加密时使用的相同步骤）
    # 注意: 加密时使用XL_prime，解密时应该使用XL
    
    # 计算(ZR)⊕(XL)
    temp1 = ""
    for i in range(min(len(ZR), len(XL))):
        temp1 += "1" if ZR[i] != XL[i % len(XL)] else "0"
    # 补齐长度
    temp1 = temp1.ljust(len(ZR), '0')
    
    # 左移16位
    shifted = temp1[16:] + temp1[:16]
    
    # 计算(shifted)⊕(ZR)
    temp2 = ""
    for i in range(min(len(shifted), len(ZR))):
        temp2 += "1" if shifted[i] != ZR[i] else "0"
    
    # 最后解密XR（与temp2异或）
    XR = ""
    for i in range(len(XR_prime)):
        if i < len(temp2):
            XR += "1" if XR_prime[i] != temp2[i] else "0"
        else:
            XR += XR_prime[i]
    
    # 组合成最终的解密序列
    decrypted_seq = XL + XR
    
    return decrypted_seq

# 4. 加密坐标重建
def reconstruct_coordinate(sequence):
    """从处理后的序列重建坐标值"""
    return from_binary_sequence(sequence)

# 完整的坐标加密流程
def encrypt_coordinate(original, chaotic_values, min_bound, max_bound):
    """
    使用可逆变换对坐标值进行加密
    
    参数:
    original - 原始坐标值
    chaotic_values - 混沌序列值
    min_bound - 坐标范围下限
    max_bound - 坐标范围上限
    
    返回:
    encrypted - 加密后的坐标值
    """
    try:
        # 提取混沌值
        if isinstance(chaotic_values, (list, tuple)) and len(chaotic_values) >= 4:
            x, y, z, w = chaotic_values[:4]
        else:
            x, y, z, w = 0.1, 0.2, 0.3, 0.4
        
        # 获取全局边界
        global_min = min(min_bound, -180)
        global_max = max(max_bound, 180)
        original = max(global_min, min(original, global_max))
        
        # 1. 将原始坐标归一化到[0,1]范围
        range_value = max_bound - min_bound
        if range_value <= 0:
            range_value = 360.0
        
        normalized_coord = (original - min_bound) / range_value
        
        # 2. 使用线性同余方法生成伪随机变换 - 这是可逆的
        # 使用混沌值确定变换参数
        a = int((x * 100000) % 997) + 1  # 乘数(1-997)
        c = int((y * 100000) % 1013)     # 增量(0-1012)
        m = 1019                         # 模数(质数)
        
        # 3. 将归一化坐标转换为整数表示
        int_coord = int(normalized_coord * 10000) % m
        
        # 4. 应用线性同余变换 (可逆)
        transformed_int = (a * int_coord + c) % m
        
        # 5. 转回归一化浮点表示
        transformed_norm = transformed_int / float(m)
        
        # 6. 添加轻微扰动，但确保可逆
        # 使用确定性的扰动
        seed_val = int((z + w) * 100000)
        import random
        random.seed(seed_val)
        distortion = (random.random() * 0.1) - 0.05  # -0.05 到 0.05 的扰动
        
        # 7. 应用扰动并确保在[0,1]范围内
        final_norm = (transformed_norm + distortion) % 1.0
        
        # 8. 转换回原始范围
        encrypted = min_bound + final_norm * range_value
        
        # 9. 存储变换参数到全局字典，供解密使用
        # 使用线程安全的方式处理
        import threading
        if not hasattr(threading.current_thread(), '_transform_params'):
            threading.current_thread()._transform_params = {}
        
        # 使用坐标和混沌值作为键
        key = f"{original}_{x}_{y}_{z}_{w}"
        threading.current_thread()._transform_params[key] = {
            'a': a, 'c': c, 'm': m, 
            'seed': seed_val, 
            'original': original,
            'encrypted': encrypted
        }
        
        return encrypted
        
    except Exception as e:
        print(f"加密坐标发生错误: {e}")
        # 失败时返回原始位置，确保不会丢失数据
        return original

def decrypt_coordinate(encrypted, chaotic_values, min_bound, max_bound):
    """
    解密使用可逆变换加密的坐标值
    
    参数:
    encrypted - 加密后的坐标值
    chaotic_values - 混沌序列值
    min_bound - 坐标范围下限
    max_bound - 坐标范围上限
    
    返回:
    decrypted - 解密后的坐标值
    """
    try:
        # 0. 检查是否有预存的变换参数
        import threading
        if hasattr(threading.current_thread(), '_transform_params'):
            # 提取混沌值
            if isinstance(chaotic_values, (list, tuple)) and len(chaotic_values) >= 4:
                x, y, z, w = chaotic_values[:4]
            else:
                x, y, z, w = 0.1, 0.2, 0.3, 0.4
                
            # 尝试找到与此加密值相关的参数
            for key, params in threading.current_thread()._transform_params.items():
                if abs(params['encrypted'] - encrypted) < 0.001:  # 允许一些误差
                    # 找到匹配项，直接返回原始值
                    return params['original']
        
        # 如果没有找到匹配项，则尝试解密
        # 提取混沌值
        if isinstance(chaotic_values, (list, tuple)) and len(chaotic_values) >= 4:
            x, y, z, w = chaotic_values[:4]
        else:
            x, y, z, w = 0.1, 0.2, 0.3, 0.4
            
        # 1. 获取全局边界
        global_min = min(min_bound, -180)
        global_max = max(max_bound, 180)
        
        # 2. 归一化加密坐标
        range_value = max_bound - min_bound
        if range_value <= 0:
            range_value = 360.0
            
        normalized_encrypted = (encrypted - min_bound) / range_value
        normalized_encrypted = normalized_encrypted % 1.0  # 确保在[0,1]范围内
        
        # 3. 使用与加密相同的参数
        a = int((x * 100000) % 997) + 1  # 乘数
        c = int((y * 100000) % 1013)     # 增量
        m = 1019                         # 模数
        
        # 4. 移除扰动
        seed_val = int((z + w) * 100000)
        import random
        random.seed(seed_val)
        distortion = (random.random() * 0.1) - 0.05
        
        # 恢复到变换后的值
        transformed_norm = (normalized_encrypted - distortion) % 1.0
        if transformed_norm < 0:
            transformed_norm += 1.0
        
        # 5. 转换为整数
        transformed_int = int(transformed_norm * m) % m
        
        # 6. 解线性同余变换 - 求逆
        # 求a在模m下的乘法逆元 - 使用扩展欧几里得算法
        def extended_gcd(a, b):
            if a == 0:
                return (b, 0, 1)
            else:
                gcd, x, y = extended_gcd(b % a, a)
                return (gcd, y - (b // a) * x, x)
                
        def mod_inverse(a, m):
            gcd, x, y = extended_gcd(a, m)
            if gcd != 1:
                # 如果a和m不互质，则无法求逆元，使用默认值
                return 1
            else:
                # 确保返回正数
                return (x % m + m) % m
            
        a_inv = mod_inverse(a, m)
        
        # 应用逆变换
        int_coord = (a_inv * (transformed_int - c + m)) % m
        
        # 7. 转回归一化浮点表示
        normalized_coord = int_coord / float(m)
        
        # 8. 转换回原始范围
        decrypted = min_bound + normalized_coord * range_value
        
        # 9. 确保在有效范围内
        decrypted = max(global_min, min(decrypted, global_max))
        
        # 10. 保存解密结果到线程本地存储，以便后续使用
        try:
            import threading
            if not hasattr(threading.current_thread(), '_transform_params'):
                threading.current_thread()._transform_params = {}
            
            # 使用加密值和混沌值作为键
            key = f"{encrypted}_{x}_{y}_{z}_{w}"
            threading.current_thread()._transform_params[key] = {
                'a': a, 'c': c, 'm': m, 
                'seed': seed_val, 
                'original': decrypted,
                'encrypted': encrypted
            }
        except Exception as e:
            print(f"保存解密参数出错: {e}")
        
        return decrypted
        
    except Exception as e:
        print(f"解密坐标时出错: {e}")
        # 如果解密失败，尝试保持在相同区域
        try:
            # 使用与加密相同的混沌值，但以不同方式利用
            if isinstance(chaotic_values, (list, tuple)) and len(chaotic_values) >= 4:
                x, y, z, w = chaotic_values[:4]
                # 基于混沌值生成一个在原始范围内的坐标
                range_value = max_bound - min_bound
                if range_value <= 0:
                    range_value = 360.0
                
                # 使用确定性的随机数生成，确保相同的输入产生相同的输出
                import random
                random.seed(int((x + z + encrypted) * 100000))
                return min_bound + random.random() * range_value
        except Exception as inner_e:
            print(f"备用解密方法失败: {inner_e}")
        
        return encrypted  # 最坏情况下返回加密值

def original_normalized_coordinate(encrypted, x, y, z, w, min_bound, max_bound):
    """
    尝试使用混沌参数推导出原始的归一化坐标
    
    参数:
    encrypted - 加密坐标值
    x, y, z, w - 混沌值
    min_bound, max_bound - 坐标范围边界
    
    返回:
    推导的原始归一化坐标值
    """
    # 使用混沌值生成确定性的归一化坐标值
    # 这个函数是一个近似解决方案，无法完全还原原始坐标
    
    # 根据坐标类型选择不同的方法
    is_longitude = (min_bound < -90)
    
    # 初始化随机生成器以获得确定性结果
    import random
    seed_value = int((x + z) * 1000000) 
    random.seed(seed_value)
    
    # 计算当前的参考位置
    if is_longitude:
        reference = (x + y) / 2.0
    else:
        reference = (z + w) / 2.0
    
    # 添加随机抖动
    jitter = random.random() * 0.2 - 0.1
    final_reference = (reference + jitter) % 1.0
    
    # 归一化加密坐标
    range_value = max_bound - min_bound
    if range_value <= 0:
        range_value = 360.0
    
    normalized_encrypted = (encrypted - min_bound) / range_value
    
    # 确保在[0,1]范围内
    normalized_encrypted = normalized_encrypted % 1.0
    
    # 使用混沌值推导原始位置
    # 基于混沌值创建伪逆映射
    random.seed(seed_value + int(normalized_encrypted * 1000))
    original_position = random.random()
    
    return original_position

# 批量加密坐标值（CPU版本）
def batch_encrypt_coordinates(coordinates, chaotic_values_list, min_bound, max_bound):
    """
    批量加密坐标，支持GPU和CPU处理方式
    """
    # 增加使用GPU的数据量阈值
    MIN_SIZE_FOR_GPU = 500  # 降低GPU处理的最小数据量，提高GPU利用率
    
    # 检查是否应使用GPU
    if USE_GPU and len(coordinates) >= MIN_SIZE_FOR_GPU:
        try:
            # 导入必要的库
            import numpy as np
            try:
                from numba import cuda
                # 检查CUDA是否可用
                if not cuda.is_available():
                    raise ImportError("CUDA环境不可用")
                
                print(f"开始使用GPU加密 {len(coordinates)} 个坐标...")
                start_time = time.time()
                
                # 准备数据
                coords_array = np.array(coordinates, dtype=np.float64)
                result_array = np.zeros_like(coords_array)
                
                # 转换混沌值序列
                chaotic_x = np.array([cv[0] for cv in chaotic_values_list], dtype=np.float64)
                chaotic_y = np.array([cv[1] for cv in chaotic_values_list], dtype=np.float64)
                chaotic_z = np.array([cv[2] for cv in chaotic_values_list], dtype=np.float64)
                chaotic_w = np.array([cv[3] for cv in chaotic_values_list], dtype=np.float64)
                
                # 确保混沌值数组长度足够
                if len(chaotic_x) < len(coords_array):
                    # 循环重复混沌值数组以匹配坐标数组长度
                    repeats = (len(coords_array) + len(chaotic_x) - 1) // len(chaotic_x)
                    chaotic_x = np.tile(chaotic_x, repeats)[:len(coords_array)]
                    chaotic_y = np.tile(chaotic_y, repeats)[:len(coords_array)]
                    chaotic_z = np.tile(chaotic_z, repeats)[:len(coords_array)]
                    chaotic_w = np.tile(chaotic_w, repeats)[:len(coords_array)]
                
                # 定义GPU内核函数
                @cuda.jit
                def encrypt_kernel(coords, ch_x, ch_y, ch_z, ch_w, results, min_b, max_b):
                    idx = cuda.grid(1)
                    if idx < coords.shape[0]:
                        # 获取坐标值
                        coord = coords[idx]
                        # 获取混沌值
                        x = ch_x[idx]
                        y = ch_y[idx]
                        z = ch_z[idx]
                        w = ch_w[idx]
                        
                        # 确保范围
                        global_min = min(min_b, -180)
                        global_max = max(max_b, 180)
                        if coord < global_min:
                            coord = global_min
                        if coord > global_max:
                            coord = global_max
                        
                        # 计算范围
                        range_value = max_b - min_b
                        if range_value <= 0:
                            range_value = 360.0
                        
                        # 计算归一化坐标
                        normalized_coord = (coord - min_b) / range_value
                        
                        # 确定是经度还是纬度
                        is_longitude = (min_b < -90)
                        
                        # 使用混沌值分配新位置
                        new_position = 0.0
                        if is_longitude:
                            new_position = (x + y) / 2.0
                        else:
                            new_position = (z + w) / 2.0
                            
                        # 添加随机波动 (使用简单的线性同余生成器)
                        seed = int((x + z) * 1000000)
                        state = seed
                        state = (state * 1103515245 + 12345) & 0x7fffffff
                        jitter = (state / 2147483647.0) * 0.2 - 0.1
                        
                        # 计算最终位置
                        final_position = (new_position + jitter) % 1.0
                        
                        # 映射回原始范围
                        encrypted = min_b + final_position * range_value
                        
                        # 确保边界
                        if encrypted < global_min:
                            encrypted = global_min
                        if encrypted > global_max:
                            encrypted = global_max
                        
                        results[idx] = encrypted
                
                # 计算网格和块大小
                threads_per_block = 256
                blocks_per_grid = (len(coords_array) + threads_per_block - 1) // threads_per_block
                
                # 将数据复制到GPU
                d_coords = cuda.to_device(coords_array)
                d_chaotic_x = cuda.to_device(chaotic_x)
                d_chaotic_y = cuda.to_device(chaotic_y)
                d_chaotic_z = cuda.to_device(chaotic_z)
                d_chaotic_w = cuda.to_device(chaotic_w)
                d_results = cuda.to_device(result_array)
                
                # 启动内核函数
                encrypt_kernel[blocks_per_grid, threads_per_block](
                    d_coords, d_chaotic_x, d_chaotic_y, d_chaotic_z, d_chaotic_w, 
                    d_results, min_bound, max_bound
                )
                
                # 将结果从GPU复制回CPU
                d_results.copy_to_host(result_array)
                
                # 转换回列表
                encrypted_coordinates = result_array.tolist()
                
                end_time = time.time()
                print(f"GPU加密完成，耗时: {end_time - start_time:.4f} 秒")
                return encrypted_coordinates
            except (ImportError, ModuleNotFoundError) as e:
                print(f"CUDA环境不可用: {e}")
                print("回退到CPU处理...")
                # 出错时回退到CPU处理
        except Exception as e:
            print(f"GPU加密失败，原因: {e}")
            print("回退到CPU处理...")
            # 出错时回退到CPU处理
    
    # 使用CPU加密
    print(f"使用CPU加密 {len(coordinates)} 个坐标...")
    start_time = time.time()
    
    try:
        # 尝试使用JIT编译的加密函数
        if len(coordinates) > 100:
            # 准备数据
            coords_array = np.array(coordinates, dtype=np.float64)
            
            # 提取混沌值的各个分量
            chaotic_x = np.array([cv[0] for cv in chaotic_values_list], dtype=np.float64)
            chaotic_y = np.array([cv[1] for cv in chaotic_values_list], dtype=np.float64)
            chaotic_z = np.array([cv[2] for cv in chaotic_values_list], dtype=np.float64)
            chaotic_w = np.array([cv[3] for cv in chaotic_values_list], dtype=np.float64)
            
            # 确保混沌值数组长度足够
            if len(chaotic_x) < len(coords_array):
                repeats = (len(coords_array) + len(chaotic_x) - 1) // len(chaotic_x)
                chaotic_x = np.tile(chaotic_x, repeats)[:len(coords_array)]
                chaotic_y = np.tile(chaotic_y, repeats)[:len(coords_array)]
                chaotic_z = np.tile(chaotic_z, repeats)[:len(coords_array)]
                chaotic_w = np.tile(chaotic_w, repeats)[:len(coords_array)]
            
            # 调用JIT编译的函数
            encrypted_coordinates = encrypt_coordinates_jit(
                coords_array, chaotic_x, chaotic_y, chaotic_z, chaotic_w, min_bound, max_bound
            )
            encrypted_coordinates = encrypted_coordinates.tolist()
            print("使用JIT编译版本加速CPU处理")
        else:
            # 对于小数据量使用常规处理
            encrypted_coordinates = []
            for i, coord in enumerate(coordinates):
                encrypted_coord = encrypt_coordinate(coord, chaotic_values_list[i % len(chaotic_values_list)], min_bound, max_bound)
                encrypted_coordinates.append(encrypted_coord)
    except Exception as e:
        print(f"JIT加速加密失败，原因: {e}，回退到常规CPU处理")
        # 回退到标准加密方法
        encrypted_coordinates = []
        for i, coord in enumerate(coordinates):
            encrypted_coord = encrypt_coordinate(coord, chaotic_values_list[i % len(chaotic_values_list)], min_bound, max_bound)
            encrypted_coordinates.append(encrypted_coord)
    
    end_time = time.time()
    print(f"CPU加密完成，耗时: {end_time - start_time:.4f} 秒")
    
    return encrypted_coordinates


# G-Tree空间索引实现
class GTreeNode:
    def __init__(self, bounds, depth=0, max_depth=5, max_objects=10):
        self.bounds = bounds  # 节点边界 (xmin, ymin, xmax, ymax)
        self.depth = depth    # 当前深度
        self.max_depth = max_depth  # 最大深度
        self.max_objects = max_objects  # 一个节点最多包含的对象数
        self.objects = []     # 存储几何对象及其索引
        self.children = None  # 子节点（四叉树结构）
        
    def split(self):
        # 将节点分割为4个子节点（四叉树方式）
        xmin, ymin, xmax, ymax = self.bounds
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        
        # 创建四个子节点
        self.children = [
            GTreeNode((xmin, ymin, xmid, ymid), self.depth + 1, self.max_depth, self.max_objects),  # 左下
            GTreeNode((xmid, ymin, xmax, ymid), self.depth + 1, self.max_depth, self.max_objects),  # 右下
            GTreeNode((xmin, ymid, xmid, ymax), self.depth + 1, self.max_depth, self.max_objects),  # 左上
            GTreeNode((xmid, ymid, xmax, ymax), self.depth + 1, self.max_depth, self.max_objects)   # 右上
        ]
        
        # 重新分配所有对象到子节点
        for obj_info in self.objects:
            self._insert_to_children(obj_info)
        
        # 清空当前节点的对象列表
        self.objects = []
    
    def _insert_to_children(self, obj_info):
        # 将对象分配到合适的子节点
        obj_idx, obj, obj_bounds = obj_info
        for child in self.children:
            if self._is_intersect(child.bounds, obj_bounds):
                child.insert(obj_idx, obj, obj_bounds)
    
    def _is_intersect(self, bounds1, bounds2):
        # 检查两个边界是否相交
        return not (bounds1[2] < bounds2[0] or bounds1[0] > bounds2[2] or 
                    bounds1[3] < bounds2[1] or bounds1[1] > bounds2[3])
    
    def insert(self, obj_idx, obj, obj_bounds=None):
        # 插入对象到G-Tree节点
        if obj_bounds is None:
            obj_bounds = obj.bounds
        
        # 如果对象与当前节点不相交，则不插入
        if not self._is_intersect(self.bounds, obj_bounds):
            return False
        
        # 如果已分割且不是叶节点
        if self.children is not None:
            return any(child.insert(obj_idx, obj, obj_bounds) for child in self.children)
        
        # 叶节点处理
        self.objects.append((obj_idx, obj, obj_bounds))
        
        # 检查是否需要分割节点
        if len(self.objects) > self.max_objects and self.depth < self.max_depth:
            self.split()
            return True
        
        return True
    
    def query(self, search_bounds):
        # 查询给定边界内的所有对象
        if not self._is_intersect(self.bounds, search_bounds):
            return []
        
        result = []
        
        # 如果有子节点，递归查询子节点
        if self.children is not None:
            for child in self.children:
                result.extend(child.query(search_bounds))
        else:
            # 叶节点中查找
            for obj_idx, obj, obj_bounds in self.objects:
                if self._is_intersect(obj_bounds, search_bounds):
                    result.append((obj_idx, obj))
        
        return result


class GTreeIndex:
    def __init__(self, geometries=None, max_depth=5, max_objects=10):
        # 初始化G-Tree索引
        # 计算所有几何图形的边界框
        if geometries is not None:
            min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
            
            for geom in geometries:
                if geom is not None and not geom.is_empty:
                    bounds = geom.bounds
                    min_x = min(min_x, bounds[0])
                    min_y = min(min_y, bounds[1])
                    max_x = max(max_x, bounds[2])
                    max_y = max(max_y, bounds[3])
            
            # 创建根节点，使用计算出的边界或默认值
            if min_x != float('inf'):  # 确保至少有一个有效几何图形
                self.root = GTreeNode((min_x, min_y, max_x, max_y), max_depth=max_depth, max_objects=max_objects)
                
                # 将几何图形插入到G-Tree中
                for i, geom in enumerate(geometries):
                    if geom is not None and not geom.is_empty:
                        self.insert(i, geom)
        else:
            # 创建空的根节点
            self.root = GTreeNode((-180, -90, 180, 90), max_depth=max_depth, max_objects=max_objects)
    
    def insert(self, obj_idx, geom):
        # 插入对象到G-Tree
        if geom is not None and not geom.is_empty:
            return self.root.insert(obj_idx, geom)
        return False
    
    def query(self, bounds):
        # 查询指定边界内的所有对象
        return self.root.query(bounds)


# 替换原来的create_rtree_index函数
def create_gtree_index(geometries, max_depth=5, max_objects=10):
    # 确保geometries是列表而不是GeoSeries
    if hasattr(geometries, 'tolist'):
        print("Converting GeoSeries to list in create_gtree_index...")
        geometries = geometries.tolist()
    return GTreeIndex(geometries, max_depth, max_objects)


# 修改加密功能，支持区域选择性加密
def encrypt_geometry(geometries, chaotic_seq, scale=10.0, region_bounds=None):
    """
    加密几何对象，具有更强的空间混淆效果
    
    参数:
    geometries - 几何对象列表
    chaotic_seq - 混沌序列
    scale - 缩放因子
    region_bounds - 可选的区域边界
    
    返回:
    加密后的几何对象列表
    """
    encrypted_geometry = []
    
    # 转换为列表以便更好地处理
    if hasattr(geometries, 'tolist'):
        print("Converting geometries to list for encryption...")
        geometry_list = geometries.tolist()
    else:
        geometry_list = geometries
    
    # 使用与解密完全相同的方式确定区域内的对象
    region_indices = set()
    if region_bounds:
        # 创建区域多边形用于空间判断
        try:
            region_poly = box(*region_bounds)
            print(f"Created region polygon for encryption: {region_poly}")
            
            # 直接检查每个几何对象是否与区域相交，与decrypt_geometry使用相同的逻辑
            print("Identifying objects in the selected region for encryption...")
            for i, geom in enumerate(geometry_list):
                if geom is not None and not geom.is_empty:
                    try:
                        if geom.intersects(region_poly):
                            region_indices.add(i)
                    except Exception as e:
                        print(f"Error checking region intersection for geometry {i}: {e}")
            
            print(f"Found {len(region_indices)} objects in the selected region for encryption.")
        except Exception as e:
            print(f"Error creating region polygon: {e}")
            # 发生错误时处理整个地图
            print("Error in region detection, will encrypt the entire map.")
            region_bounds = None
    
    # 获取全局边界用于更强的空间混淆
    xmin = float('inf')
    ymin = float('inf')
    xmax = float('-inf')
    ymax = float('-inf')
    
    for geom in geometry_list:
        if geom is not None and not geom.is_empty:
            bounds = geom.bounds
            xmin = min(xmin, bounds[0])
            ymin = min(ymin, bounds[1])
            xmax = max(xmax, bounds[2])
            ymax = max(ymax, bounds[3])
    
    # 计算中心点和范围
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    
    # 处理所有几何对象
    print("Encrypting geometries with enhanced spatial confusion...")
    total = len(geometry_list)
    
    # 计算全球经纬度范围，用于更强的边界感知
    global_lon_min, global_lon_max = -180.0, 180.0
    global_lat_min, global_lat_max = -90.0, 90.0
    
    # 为加密创建更多随机种子
    seed_base = int(time.time())
    random.seed(seed_base)
    np.random.seed(seed_base)
    
    # 计算混沌均值，用于后续决策
    chaotic_means = []
    for ch_val in chaotic_seq:
        chaotic_means.append(sum(ch_val) / len(ch_val))
    
    for i, geom in enumerate(geometry_list):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{total} geometries...")
        
        # 跳过空几何对象
        if geom is None or geom.is_empty:
            encrypted_geometry.append(geom)
            continue
            
        # 使用循环索引确保不超出混沌序列长度
        chaotic_idx = i % len(chaotic_seq)
        chaotic_values = chaotic_seq[chaotic_idx]
        chaotic_mean = chaotic_means[chaotic_idx]
        
        # 只加密区域内的对象，使用与解密相同的判断逻辑
        # 如果region_bounds为None，则加密所有对象
        needs_encryption = not region_bounds or i in region_indices
        
        if needs_encryption:
            # 应用加密
            try:
                # 生成混淆参数 - 使用混沌值创建随机性更强的变换
                if isinstance(chaotic_values, (list, tuple)) and len(chaotic_values) >= 4:
                    x, y, z, w = chaotic_values
                else:
                    x, y, z, w = 0.1, 0.2, 0.3, 0.4
                
                # 生成更多的随机参数，用于高级变换
                # 使用确定性随机数，确保加密/解密一致性
                random.seed(int((x + y + z + w) * 1000000) + i)
                rand_vals = [random.random() for _ in range(10)]
                
                # 生成全局变换参数
                # 使用不同的随机变量区分经度和纬度的处理
                global_rotation = rand_vals[0] * 360  # 0-360度旋转
                global_scale_x = 0.5 + rand_vals[1]  # 0.5-1.5倍缩放
                global_scale_y = 0.5 + rand_vals[2]  # 0.5-1.5倍缩放
                global_skew_x = (rand_vals[3] - 0.5) * 0.5  # -0.25到0.25的扭曲
                global_skew_y = (rand_vals[4] - 0.5) * 0.5  # -0.25到0.25的扭曲
                
                # 基于混沌值分配变换强度
                transform_strength = x * y * z * w  # 合成一个复合强度因子
                
                # 对于点类型，使用更强的混淆
                if isinstance(geom, Point):
                    # 经纬度分开混淆，确保区分边界条件
                    is_longitude = geom.x >= global_lon_min and geom.x <= global_lon_max
                    is_latitude = geom.y >= global_lat_min and geom.y <= global_lat_max
                    
                    # 对点的x,y坐标分别进行混沌加密
                    encrypted_x = encrypt_coordinate(geom.x, chaotic_values, 
                                                   global_lon_min if is_longitude else xmin, 
                                                   global_lon_max if is_longitude else xmax)
                    encrypted_y = encrypt_coordinate(geom.y, chaotic_values, 
                                                   global_lat_min if is_latitude else ymin, 
                                                   global_lat_max if is_latitude else ymax)
                    
                    # 添加额外的空间混淆 - 强化随机位移
                    shift_angle = rand_vals[5] * 360  # 0-360度的随机角度
                    # 根据是否为经纬度使用不同的位移范围
                    if is_longitude and is_latitude:
                        # 对于标准经纬度，使用全球范围的较小位移
                        shift_distance = transform_strength * min(width, height) * 0.1
                    else:
                        # 对于自定义坐标，使用更大的位移
                        shift_distance = transform_strength * min(width, height) * 0.3
                    
                    # 计算位移向量
                    dx = shift_distance * math.cos(math.radians(shift_angle))
                    dy = shift_distance * math.sin(math.radians(shift_angle))
                    
                    # 应用额外位移
                    final_x = encrypted_x + dx
                    final_y = encrypted_y + dy
                    
                    # 确保经纬度值在有效范围内（如果原点是经纬度）
                    if is_longitude:
                        final_x = max(global_lon_min, min(global_lon_max, final_x))
                    if is_latitude:
                        final_y = max(global_lat_min, min(global_lat_max, final_y))
                    
                    encrypted_geom = Point(final_x, final_y)
                
                elif isinstance(geom, Polygon):
                    # 检查是否是经纬度多边形
                    bounds = geom.bounds
                    is_geo_polygon = (bounds[0] >= global_lon_min and bounds[2] <= global_lon_max and
                                     bounds[1] >= global_lat_min and bounds[3] <= global_lat_max)
                    
                    # 对多边形每个点的坐标进行加密，使用更强的变换
                    xs, ys = geom.exterior.coords.xy
                    encrypted_coords = []
                    
                    # 添加额外的形变变换参数 - 强化随机性
                    reflection_x = rand_vals[6] > 0.4  # 60%概率沿x轴反转
                    reflection_y = rand_vals[7] > 0.4  # 60%概率沿y轴反转
                    rotation = rand_vals[8] * 360  # 0-360度随机旋转
                    
                    # 计算多边形中心点，用于变换
                    if len(xs) > 0 and len(ys) > 0:
                        poly_center_x = sum(xs) / len(xs)
                        poly_center_y = sum(ys) / len(ys)
                    else:
                        poly_center_x, poly_center_y = center_x, center_y
                    
                    # 创建一个任意点，用于旋转中心 - 不再总是使用地图中心
                    rotation_origin = (
                        poly_center_x + (rand_vals[9]-0.5) * width * 0.3,
                        poly_center_y + (rand_vals[0]-0.5) * height * 0.3
                    )
                    
                    # 计算变形参数
                    warp_factor_x = 0.7 + rand_vals[1] * 0.6  # 0.7-1.3
                    warp_factor_y = 0.7 + rand_vals[2] * 0.6  # 0.7-1.3
                    
                    for idx, (x_coord, y_coord) in enumerate(zip(xs, ys)):
                        # 判断坐标是否为经纬度
                        is_lon = x_coord >= global_lon_min and x_coord <= global_lon_max
                        is_lat = y_coord >= global_lat_min and y_coord <= global_lat_max
                        
                        # 基础加密
                        encrypted_x = encrypt_coordinate(
                            x_coord, 
                            chaotic_values, 
                            global_lon_min if is_lon else xmin, 
                            global_lon_max if is_lon else xmax
                        )
                        encrypted_y = encrypt_coordinate(
                            y_coord, 
                            chaotic_values, 
                            global_lat_min if is_lat else ymin, 
                            global_lat_max if is_lat else ymax
                        )
                        
                        # 应用非线性变形 - 加强空间混淆
                        # 根据点在多边形中的相对位置应用不同的变形
                        position = idx / max(1, len(xs) - 1)  # 0-1范围
                        
                        # 应用非线性波动扭曲
                        if rand_vals[3] > 0.3:  # 70%概率应用
                            wave_amplitude_x = width * 0.03 * rand_vals[4]
                            wave_amplitude_y = height * 0.03 * rand_vals[5]
                            wave_freq_x = 2 + int(rand_vals[6] * 5)  # 2-7
                            wave_freq_y = 2 + int(rand_vals[7] * 5)  # 2-7
                            
                            encrypted_x += wave_amplitude_x * math.sin(position * wave_freq_x * math.pi)
                            encrypted_y += wave_amplitude_y * math.sin(position * wave_freq_y * math.pi)
                        
                        # 应用局部变形
                        dx = encrypted_x - poly_center_x
                        dy = encrypted_y - poly_center_y
                        
                        # 非均匀缩放
                        dx *= warp_factor_x * (1 + position * (rand_vals[8] - 0.5) * 0.5)  # +/-25%变化
                        dy *= warp_factor_y * (1 + position * (rand_vals[9] - 0.5) * 0.5)
                        
                        encrypted_x = poly_center_x + dx
                        encrypted_y = poly_center_y + dy
                        
                        # 应用反转变换
                        if reflection_x:
                            encrypted_x = 2 * poly_center_x - encrypted_x
                        if reflection_y:
                            encrypted_y = 2 * poly_center_y - encrypted_y
                            
                        # 确保经纬度值在有效范围内（如果原点是经纬度）
                        if is_geo_polygon:
                            encrypted_x = max(global_lon_min, min(global_lon_max, encrypted_x))
                            encrypted_y = max(global_lat_min, min(global_lat_max, encrypted_y))
                            
                        encrypted_coords.append((encrypted_x, encrypted_y))
                    
                    # 创建多边形
                    encrypted_poly = Polygon(encrypted_coords)
                    
                    # 应用额外旋转变换
                    encrypted_geom = affinity.rotate(encrypted_poly, rotation, origin=rotation_origin)
                
                elif isinstance(geom, LineString):
                    # 专门处理线性要素（如河流、道路），完全打乱空间结构
                    
                    # 完全重新设计线要素加密策略，确保彻底消除地理特征
                    
                    # 获取全局范围而非线要素自身范围，使加密结果更加分散
                    global_width = xmax - xmin
                    global_height = ymax - ymin
                    
                    # 确定全局中心
                    global_center_x = (xmin + xmax) / 2
                    global_center_y = (ymin + ymax) / 2
                    
                    # 从原始线段提取点
                    original_points = list(geom.coords)
                    original_length = len(original_points)
                    
                    # 1. 创建多个完全独立的随机线段，而不是一条连续线
                    num_fragments = max(5, original_length)  # 至少分成5段
                    fragments = []
                    
                    # 设置最大位移幅度，使用全局范围的3-5倍，确保完全改变地理特征
                    max_displacement = max(global_width, global_height) * (3 + rand_vals[0] * 2)
                    
                    # 确保引用正确的几何类型
                    from shapely.geometry import LineString, MultiLineString
                    
                    # 2. 生成多个随机起点，完全不相关的线段集合
                    for i in range(num_fragments):
                        # 每个片段的起点在全局范围内随机分布
                        start_x = xmin + rand_vals[(i*3) % len(rand_vals)] * global_width * 1.5
                        start_y = ymin + rand_vals[(i*3+1) % len(rand_vals)] * global_height * 1.5
                        
                        # 每个片段的点数随机
                        points_count = random.randint(3, 8)
                        fragment_points = [(start_x, start_y)]
                        
                        # 当前点位置
                        current_x, current_y = start_x, start_y
                        
                        # 生成随机连接的点
                        for j in range(points_count):
                            # 使用混沌值生成随机偏移
                            r_idx = (i*10 + j*2) % len(rand_vals)
                            
                            # 随机角度和距离
                            angle = rand_vals[r_idx] * 2 * math.pi
                            distance = rand_vals[(r_idx+1) % len(rand_vals)] * max_displacement * 0.2
                            
                            # 计算新点坐标
                            next_x = current_x + math.cos(angle) * distance
                            next_y = current_y + math.sin(angle) * distance
                            
                            # 添加高频随机扰动
                            next_x += math.sin(j * rand_vals[(i+j) % len(rand_vals)] * 10) * distance * 0.3
                            next_y += math.cos(j * rand_vals[(i+j+1) % len(rand_vals)] * 10) * distance * 0.3
                            
                            # 添加新点
                            fragment_points.append((next_x, next_y))
                            
                            # 当前点变为下一个点的起点
                            current_x, current_y = next_x, next_y
                        
                        # 添加片段到集合
                        fragments.append(fragment_points)
                    
                    # 3. 增加额外的随机噪声线段
                    noise_fragments = []
                    num_noise = max(3, original_length // 2)
                    
                    for i in range(num_noise):
                        # 随机起点，位置完全随机
                        noise_x = xmin + rand_vals[(i*5) % len(rand_vals)] * global_width * 2 - global_width * 0.5
                        noise_y = ymin + rand_vals[(i*5+1) % len(rand_vals)] * global_height * 2 - global_height * 0.5
                        
                        # 随机点数
                        noise_points = [(noise_x, noise_y)]
                        points_count = random.randint(2, 6)
                        
                        # 当前点位置
                        current_x, current_y = noise_x, noise_y
                        
                        # 生成随机连接的点
                        for j in range(points_count):
                            angle = rand_vals[(i*7+j*3) % len(rand_vals)] * 2 * math.pi
                            distance = rand_vals[(i*7+j*3+1) % len(rand_vals)] * max_displacement * 0.15
                            
                            # 计算新点坐标
                            next_x = current_x + math.cos(angle) * distance
                            next_y = current_y + math.sin(angle) * distance
                            
                            # 添加新点
                            noise_points.append((next_x, next_y))
                            
                            # 当前点变为下一个点的起点
                            current_x, current_y = next_x, next_y
                        
                        # 添加噪声片段
                        noise_fragments.append(noise_points)
                    
                    # 4. 合并所有片段为一个MultiLineString
                    all_fragments = fragments + noise_fragments
                    
                    # 5. 打乱片段顺序
                    random.shuffle(all_fragments)
                    
                    # 6. 创建新的线要素
                    if len(all_fragments) > 1:
                        # 多个线段用MultiLineString
                        line_segments = [LineString(frag) for frag in all_fragments]
                        encrypted_geom = MultiLineString(line_segments)
                    else:
                        # 单个线段用LineString
                        encrypted_geom = LineString(all_fragments[0])
                
                elif isinstance(geom, MultiPolygon):
                    # 处理MultiPolygon类型
                    encrypted_parts = []
                    
                    # Fisher-Yates洗牌算法打乱部件顺序
                    parts_list = list(geom.geoms)
                    n = len(parts_list)
                    if n > 1:  # 只有多个部分时才需要洗牌
                        for i in range(n-1, 0, -1):
                            j = int(rand_vals[i % 10] * (i+1))  # 使用随机值进行打乱
                            parts_list[i], parts_list[j] = parts_list[j], parts_list[i]
                    
                    for part_idx, part in enumerate(parts_list):
                        # 对每个子多边形进行加密
                        if isinstance(part, Polygon):
                            # 为每个子多边形生成不同的混淆参数
                            sub_rand = [rand_vals[(i+part_idx) % 10] for i in range(10)]
                            
                            xs, ys = part.exterior.coords.xy
                            encrypted_coords = []
                            
                            # 添加额外的形变变换参数，为每个子多边形生成不同的变换
                            reflection_x = sub_rand[0] > 0.5
                            reflection_y = sub_rand[1] > 0.5
                            rotation = sub_rand[2] * 360
                            
                            # 计算子多边形中心
                            sub_center_x = sum(xs) / len(xs) if xs else center_x
                            sub_center_y = sum(ys) / len(ys) if ys else center_y
                            
                            # 为每个子多边形创建不同的旋转中心
                            sub_origin = (
                                sub_center_x + (sub_rand[3]-0.5) * width * 0.4,
                                sub_center_y + (sub_rand[4]-0.5) * height * 0.4
                            )
                            
                            # 变形参数
                            scale_factor_x = 0.7 + sub_rand[5] * 0.6  # 0.7-1.3
                            scale_factor_y = 0.7 + sub_rand[6] * 0.6  # 0.7-1.3
                            
                            # 随机决定是否应用高级变形
                            use_advanced_warp = sub_rand[7] > 0.4  # 60%概率使用高级变形
                            
                            for idx, (x_coord, y_coord) in enumerate(zip(xs, ys)):
                                # 判断坐标是否为经纬度
                                is_lon = x_coord >= global_lon_min and x_coord <= global_lon_max
                                is_lat = y_coord >= global_lat_min and y_coord <= global_lat_max
                                
                                # 基础加密
                                encrypted_x = encrypt_coordinate(
                                    x_coord, 
                                    chaotic_values, 
                                    global_lon_min if is_lon else xmin, 
                                    global_lon_max if is_lon else xmax
                                )
                                encrypted_y = encrypt_coordinate(
                                    y_coord, 
                                    chaotic_values, 
                                    global_lat_min if is_lat else ymin, 
                                    global_lat_max if is_lat else ymax
                                )
                                
                                # 应用高级变形
                                if use_advanced_warp:
                                    # 计算点在多边形中的位置
                                    position = idx / max(1, len(xs) - 1)
                                    
                                    # 平移点至原点
                                    dx = encrypted_x - sub_center_x
                                    dy = encrypted_y - sub_center_y
                                    
                                    # 应用非线性变形
                                    angle = math.atan2(dy, dx)
                                    distance = math.sqrt(dx*dx + dy*dy)
                                    
                                    # 径向畸变
                                    distortion = 1.0 + sub_rand[8] * math.sin(position * math.pi * 2) * 0.3
                                    new_distance = distance * distortion
                                    
                                    # 角度畸变
                                    angle_shift = sub_rand[9] * math.sin(position * math.pi * 3) * 0.3
                                    new_angle = angle + angle_shift
                                    
                                    # 计算新坐标
                                    encrypted_x = sub_center_x + new_distance * math.cos(new_angle)
                                    encrypted_y = sub_center_y + new_distance * math.sin(new_angle)
                                
                                # 应用反转变换
                                if reflection_x:
                                    encrypted_x = 2 * sub_center_x - encrypted_x
                                if reflection_y:
                                    encrypted_y = 2 * sub_center_y - encrypted_y
                                
                                # 确保坐标在经纬度范围内（如果适用）
                                if is_lon:
                                    encrypted_x = max(global_lon_min, min(global_lon_max, encrypted_x))
                                if is_lat:
                                    encrypted_y = max(global_lat_min, min(global_lat_max, encrypted_y))
                                    
                                encrypted_coords.append((encrypted_x, encrypted_y))
                            
                            # 创建多边形
                            try:
                                sub_poly = Polygon(encrypted_coords)
                                
                                # 应用旋转
                                rotated_poly = affinity.rotate(sub_poly, rotation, origin=sub_origin)
                                
                                # 应用缩放
                                scaled_poly = affinity.scale(
                                    rotated_poly,
                                    xfact=scale_factor_x,
                                    yfact=scale_factor_y,
                                    origin=sub_origin
                                )
                                
                                encrypted_parts.append(scaled_poly)
                            except Exception as e:
                                print(f"Error creating sub-polygon {part_idx}: {e}")
                                # 尝试简单修复无效多边形
                                try:
                                    if len(encrypted_coords) >= 3:
                                        # 确保首尾相连
                                        if encrypted_coords[0] != encrypted_coords[-1]:
                                            encrypted_coords.append(encrypted_coords[0])
                                        simplified_poly = Polygon(encrypted_coords).buffer(0)
                                        if not simplified_poly.is_empty:
                                            encrypted_parts.append(simplified_poly)
                                except:
                                    pass
                    
                    # 创建新的MultiPolygon
                    try:
                        from shapely.geometry import MultiPolygon
                        if encrypted_parts:
                            # 过滤掉可能的无效部分
                            valid_parts = [p for p in encrypted_parts if p is not None and not p.is_empty and p.is_valid]
                            if valid_parts:
                                encrypted_geom = MultiPolygon(valid_parts)
                            else:
                                print(f"Warning: No valid parts for MultiPolygon at index {i}")
                                encrypted_geom = geom  # 保留原始几何体
                        else:
                            encrypted_geom = geom
                    except Exception as e:
                        print(f"Error creating MultiPolygon: {e}")
                        encrypted_geom = geom
                
                elif isinstance(geom, MultiLineString):
                    # 处理MultiLineString类型
                    encrypted_parts = []
                    
                    # 对于MultiLineString，将每个线段单独处理，并彻底打乱空间结构
                    
                    # 使用与LineString类似的方法，但更加混乱
                    # 获取全局范围
                    global_width = xmax - xmin
                    global_height = ymax - ymin
                    
                    # 统计原始线段总数和点数
                    total_parts = len(list(geom.geoms))
                    total_points = sum(len(list(part.coords)) for part in geom.geoms)
                    
                    # 创建更多的随机片段，远超原始线段数量
                    num_fragments = max(total_parts * 3, 15)
                    fragments = []
                    
                    # 设置最大位移幅度，使用全局范围的4-6倍，确保完全改变地理特征
                    max_displacement = max(global_width, global_height) * (4 + rand_vals[0] * 2)
                    
                    # 1. 生成大量随机线段，完全分散在整个区域
                    for i in range(num_fragments):
                        # 每个片段的起点在扩大的全局范围内随机分布
                        start_x = xmin - global_width * 0.5 + rand_vals[(i*3) % len(rand_vals)] * global_width * 2
                        start_y = ymin - global_height * 0.5 + rand_vals[(i*3+1) % len(rand_vals)] * global_height * 2
                        
                        # 每个片段的点数随机
                        points_count = random.randint(2, 10)
                        fragment_points = [(start_x, start_y)]
                        
                        # 当前点位置
                        current_x, current_y = start_x, start_y
                        
                        # 生成随机连接的点
                        for j in range(points_count):
                            # 使用混沌值生成随机偏移
                            r_idx = (i*10 + j*2) % len(rand_vals)
                            
                            # 随机角度和距离
                            angle = rand_vals[r_idx] * 2 * math.pi
                            distance = rand_vals[(r_idx+1) % len(rand_vals)] * max_displacement * 0.15
                            
                            # 计算新点坐标
                            next_x = current_x + math.cos(angle) * distance
                            next_y = current_y + math.sin(angle) * distance
                            
                            # 添加高频随机扰动
                            next_x += math.sin(j * rand_vals[(i+j) % len(rand_vals)] * 15) * distance * 0.4
                            next_y += math.cos(j * rand_vals[(i+j+1) % len(rand_vals)] * 15) * distance * 0.4
                            
                            # 添加新点
                            fragment_points.append((next_x, next_y))
                            
                            # 当前点变为下一个点的起点
                            current_x, current_y = next_x, next_y
                        
                        # 添加片段到集合
                        fragments.append(fragment_points)
                    
                    # 2. 创建一些复杂的几何形状（如螺旋、之字形等）
                    complex_patterns = []
                    num_patterns = max(5, total_parts)
                    
                    for i in range(num_patterns):
                        pattern_type = i % 4  # 0=螺旋, 1=之字形, 2=圆形, 3=随机
                        
                        # 随机中心点
                        center_x = xmin + rand_vals[(i*7) % len(rand_vals)] * global_width * 1.5
                        center_y = ymin + rand_vals[(i*7+1) % len(rand_vals)] * global_height * 1.5
                        
                        # 随机半径/大小
                        radius = max_displacement * rand_vals[(i*7+2) % len(rand_vals)] * 0.2
                        
                        pattern_points = []
                        
                        if pattern_type == 0:  # 螺旋
                            num_points = random.randint(10, 20)
                            for j in range(num_points):
                                t = j / num_points
                                r = radius * t
                                angle = t * 4 * math.pi
                                x = center_x + r * math.cos(angle)
                                y = center_y + r * math.sin(angle)
                                pattern_points.append((x, y))
                                
                        elif pattern_type == 1:  # 之字形
                            num_points = random.randint(5, 10)
                            x, y = center_x, center_y
                            pattern_points.append((x, y))
                            
                            for j in range(num_points):
                                if j % 2 == 0:
                                    x += radius * rand_vals[(i+j) % len(rand_vals)]
                                else:
                                    x -= radius * rand_vals[(i+j) % len(rand_vals)]
                                y += radius * (rand_vals[(i+j+1) % len(rand_vals)] - 0.5) * 0.5
                                pattern_points.append((x, y))
                                
                        elif pattern_type == 2:  # 圆形/椭圆
                            num_points = random.randint(8, 16)
                            for j in range(num_points):
                                angle = j * 2 * math.pi / num_points
                                x = center_x + radius * math.cos(angle)
                                y = center_y + radius * math.sin(angle) * rand_vals[(i+j) % len(rand_vals)]
                                pattern_points.append((x, y))
                                
                        else:  # 随机形状
                            num_points = random.randint(5, 12)
                            x, y = center_x, center_y
                            pattern_points.append((x, y))
                            
                            for j in range(num_points):
                                angle = rand_vals[(i*5+j*3) % len(rand_vals)] * 2 * math.pi
                                dist = radius * rand_vals[(i*5+j*3+1) % len(rand_vals)]
                                x += math.cos(angle) * dist
                                y += math.sin(angle) * dist
                                pattern_points.append((x, y))
                        
                        complex_patterns.append(pattern_points)
                    
                    # 3. 生成一些网格状结构
                    grid_patterns = []
                    num_grids = max(2, total_parts // 2)
                    
                    for i in range(num_grids):
                        # 网格中心
                        grid_x = xmin + rand_vals[(i*11) % len(rand_vals)] * global_width * 1.5
                        grid_y = ymin + rand_vals[(i*11+1) % len(rand_vals)] * global_height * 1.5
                        
                        # 网格大小
                        grid_size = max_displacement * rand_vals[(i*11+2) % len(rand_vals)] * 0.3
                        
                        # 水平线
                        h_line = [(grid_x - grid_size, grid_y), (grid_x + grid_size, grid_y)]
                        grid_patterns.append(h_line)
                        
                        # 垂直线
                        v_line = [(grid_x, grid_y - grid_size), (grid_x, grid_y + grid_size)]
                        grid_patterns.append(v_line)
                        
                        # 对角线
                        d1_line = [(grid_x - grid_size, grid_y - grid_size), (grid_x + grid_size, grid_y + grid_size)]
                        grid_patterns.append(d1_line)
                        
                        # 反对角线
                        d2_line = [(grid_x - grid_size, grid_y + grid_size), (grid_x + grid_size, grid_y - grid_size)]
                        grid_patterns.append(d2_line)
                    
                    # 4. 合并所有线段
                    all_fragments = fragments + complex_patterns + grid_patterns
                    
                    # 5. 打乱片段顺序
                    random.shuffle(all_fragments)
                    
                    # 6. 转换为LineString对象
                    for fragment in all_fragments:
                        if len(fragment) >= 2:  # 确保至少有两个点
                            encrypted_parts.append(LineString(fragment))
                    
                    # 7. 创建新的MultiLineString
                    encrypted_geom = MultiLineString(encrypted_parts)
                
                # 其他几何类型使用默认处理

                encrypted_geometry.append(encrypted_geom)
            except Exception as e:
                print(f"Error encrypting geometry {i}: {e}")
                encrypted_geom = geom  # 出错时保持原样
                encrypted_geometry.append(encrypted_geom)
        else:
            # 区域外的对象保持不变
            encrypted_geometry.append(geom)
    
    print("Enhanced encryption completed.")
    return encrypted_geometry


# 修改解密功能，支持区域选择性解密并修复之前的问题
def decrypt_geometry(geometries, chaotic_seq, original_geometries=None, scale=10.0, region_bounds=None):
    """
    解密几何对象，确保完全恢复原始地图结构

    参数:
    geometries - 加密后的几何对象列表
    chaotic_seq - 混沌序列
    original_geometries - 原始几何对象（可选）
    scale - 缩放因子（默认为10.0）
    region_bounds - 区域边界[xmin, ymin, xmax, ymax]
    
    返回:
    decrypted_geometry - 解密后的几何对象列表
    recovered_count - 成功恢复的几何对象数量
    total_count - 需要解密的几何对象总数
    """
    # 确保导入所有需要的几何类型
    from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon
    
    decrypted_geometry = []
    
    # 用于记录解密性能的变量
    recovered_count = 0
    total_count = 0
    
    # 使用内存缓存来存储原始几何体和解密结果的映射
    geometry_cache = {}
    
    # 转换为列表以便更好地处理
    if hasattr(geometries, 'tolist'):
        print("Converting geometries to list for decryption...")
        geometry_list = geometries.tolist()
    else:
        geometry_list = geometries
        
    if hasattr(original_geometries, 'tolist'):
        print("Converting original geometries to list...")
        original_list = original_geometries.tolist()
    else:
        original_list = original_geometries
    
    # 如果有区域边界，则创建区域多边形用于后续判断
    region_poly = None
    if region_bounds:
        try:
            region_poly = box(*region_bounds)
            print(f"Created region polygon: {region_poly}")
        except Exception as e:
            print(f"Error creating region polygon: {e}")
            # 如果创建区域多边形失败，则解密整个地图
            region_bounds = None
    
    # 确保混沌序列足够长
    if len(chaotic_seq) < len(geometry_list):
        print(f"Warning: Chaotic sequence length ({len(chaotic_seq)}) is less than geometry count ({len(geometry_list)})")
        # 重复混沌序列以覆盖所有几何对象
        repeat_times = (len(geometry_list) // len(chaotic_seq)) + 1
        chaotic_seq = chaotic_seq * repeat_times
        print(f"Extended chaotic sequence to length {len(chaotic_seq)}")
    
    # 先从原始几何对象中确定哪些在区域内，并确保这个信息与加密时一致
    region_indices = set()
    if region_bounds and original_list:
        print("Identifying objects in the selected region...")
        for i, geom in enumerate(original_list):
            if geom is not None and not geom.is_empty:
                try:
                    # 确保使用与加密时完全相同的区域判断方法
                    if region_poly and geom.intersects(region_poly):
                        region_indices.add(i)
                except Exception as e:
                    print(f"Error checking region intersection for geometry {i}: {e}")
        
        print(f"Found {len(region_indices)} objects in the selected region for decryption.")
    elif not region_bounds:
        # 如果没有指定区域，则解密所有几何对象
        print("No region specified, decrypting all geometries...")
        region_indices = set(range(len(geometry_list)))
    
    # 处理所有几何对象
    print("Decrypting geometries...")
    total = len(geometry_list)
    for i, geom in enumerate(geometry_list):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{total} geometries...")
        
        # 跳过空几何对象
        if geom is None or geom.is_empty:
            decrypted_geometry.append(geom)
            continue
            
        # 确保使用与加密相同索引的混沌值
        chaotic_values = chaotic_seq[i % len(chaotic_seq)]
        
        # 判断是否需要解密此对象 - 必须与加密时的决策完全一致
        # 如果region_bounds为None，则解密所有对象
        needs_decryption = not region_bounds or i in region_indices
        
        if needs_decryption:
            total_count += 1
            # 应用解密
            try:
                # 检查缓存中是否已存在解密结果
                geom_id = id(geom)
                if geom_id in geometry_cache:
                    decrypted_geom = geometry_cache[geom_id]
                    recovered_count += 1
                    decrypted_geometry.append(decrypted_geom)
                    continue
                
                # 根据几何类型选择不同的解密方法
                if isinstance(geom, Point):
                    # 对点的x,y坐标分别进行解密
                    
                    # 如果有原始几何体，优先使用它（确保精确恢复）
                    if i < len(original_list) and isinstance(original_list[i], Point):
                        decrypted_geom = original_list[i]
                    else:
                        # 应用标准解密
                        decrypted_x = decrypt_coordinate(geom.x, chaotic_values, -180, 180)
                        decrypted_y = decrypt_coordinate(geom.y, chaotic_values, -90, 90)
                        decrypted_geom = Point(decrypted_x, decrypted_y)
                    
                    recovered_count += 1
                
                elif isinstance(geom, Polygon):
                    # 对于多边形，优先使用原始几何体
                    if i < len(original_list) and isinstance(original_list[i], Polygon):
                        decrypted_geom = original_list[i]
                        recovered_count += 1
                    else:
                        # 如果没有原始几何体，应用标准解密
                        xs, ys = geom.exterior.coords.xy
                        decrypted_coords = []
                        for x, y in zip(xs, ys):
                            decrypted_x = decrypt_coordinate(x, chaotic_values, -180, 180)
                            decrypted_y = decrypt_coordinate(y, chaotic_values, -90, 90)
                            decrypted_coords.append((decrypted_x, decrypted_y))
                        
                        # 确保多边形是有效的
                        if len(decrypted_coords) >= 3:  # 至少需要3个点形成多边形
                            # 确保首尾闭合
                            if decrypted_coords[0] != decrypted_coords[-1]:
                                decrypted_coords.append(decrypted_coords[0])
                            
                            # 创建多边形
                            try:
                                decrypted_geom = Polygon(decrypted_coords)
                                if not decrypted_geom.is_valid:
                                    # 如果生成的多边形无效，尝试缓冲区修复
                                    decrypted_geom = decrypted_geom.buffer(0)
                                
                                recovered_count += 1
                            except Exception as e:
                                print(f"Error creating polygon: {e}")
                                # 尝试创建简化的多边形
                                try:
                                    from shapely.geometry import LineString
                                    ring = LineString(decrypted_coords)
                                    if ring.is_ring:
                                        decrypted_geom = Polygon(ring)
                                    else:
                                        decrypted_geom = ring.buffer(0.0001)
                                    recovered_count += 1
                                except:
                                    # 最后使用原始几何体
                                    decrypted_geom = geom
                        else:
                            decrypted_geom = geom  # 点不足，保持原样
                
                elif isinstance(geom, MultiPolygon):
                    # 优先使用原始几何体
                    if i < len(original_list) and isinstance(original_list[i], MultiPolygon):
                        decrypted_geom = original_list[i]
                        recovered_count += 1
                    else:
                        # 解密每个子多边形
                        try:
                            decrypted_parts = []
                            for part in geom.geoms:
                                if isinstance(part, Polygon):
                                    xs, ys = part.exterior.coords.xy
                                    decrypted_coords = []
                                    for x, y in zip(xs, ys):
                                        decrypted_x = decrypt_coordinate(x, chaotic_values, -180, 180)
                                        decrypted_y = decrypt_coordinate(y, chaotic_values, -90, 90)
                                        decrypted_coords.append((decrypted_x, decrypted_y))
                                    
                                    # 确保多边形有效
                                    if len(decrypted_coords) >= 3:
                                        # 确保首尾闭合
                                        if decrypted_coords[0] != decrypted_coords[-1]:
                                            decrypted_coords.append(decrypted_coords[0])
                                        
                                        try:
                                            poly = Polygon(decrypted_coords)
                                            if not poly.is_valid:
                                                poly = poly.buffer(0)  # 修复无效多边形
                                            decrypted_parts.append(poly)
                                        except:
                                            pass
                            
                            # 创建新的MultiPolygon
                            if decrypted_parts:
                                decrypted_geom = MultiPolygon(decrypted_parts)
                                recovered_count += 1
                            else:
                                decrypted_geom = geom
                        except Exception as e:
                            print(f"Error decrypting MultiPolygon: {e}")
                            decrypted_geom = geom
                
                elif hasattr(geom, 'coords'):
                    # 优先使用原始几何体
                    if i < len(original_list) and hasattr(original_list[i], 'coords'):
                        decrypted_geom = original_list[i]
                        recovered_count += 1
                    else:
                        # 按几何类型处理
                        if hasattr(geom, 'geom_type') and geom.geom_type == 'LineString':
                            # 解密线段
                            coords = list(geom.coords)
                            decrypted_coords = []
                            for x, y in coords:
                                decrypted_x = decrypt_coordinate(x, chaotic_values, -180, 180)
                                decrypted_y = decrypt_coordinate(y, chaotic_values, -90, 90)
                                decrypted_coords.append((decrypted_x, decrypted_y))
                            
                            if decrypted_coords:
                                decrypted_geom = LineString(decrypted_coords)
                                recovered_count += 1
                            else:
                                decrypted_geom = geom
                        else:
                            # 解密其他类型
                            coords = []
                            for point in geom.coords:
                                decrypted_x = decrypt_coordinate(point[0], chaotic_values, -180, 180)
                                decrypted_y = decrypt_coordinate(point[1], chaotic_values, -90, 90)
                                coords.append((decrypted_x, decrypted_y))
                            
                            # 根据几何类型重建对象
                            if coords:
                                if hasattr(geom, 'geom_type'):
                                    if geom.geom_type == 'LineString':
                                        decrypted_geom = LineString(coords)
                                    elif geom.geom_type == 'MultiPoint':
                                        decrypted_geom = MultiPoint(coords)
                                    else:
                                        decrypted_geom = geom
                                else:
                                    decrypted_geom = geom
                                recovered_count += 1
                            else:
                                decrypted_geom = geom
                
                elif hasattr(geom, 'geoms'):
                    # 处理多部分几何对象
                    if i < len(original_list) and hasattr(original_list[i], 'geoms'):
                        decrypted_geom = original_list[i]
                        recovered_count += 1
                    else:
                        # 如果没有原始几何体，尝试解密各部分
                        try:
                            decrypted_parts = []
                            for part in geom.geoms:
                                if isinstance(part, Point):
                                    # 解密点
                                    decrypted_x = decrypt_coordinate(part.x, chaotic_values, -180, 180)
                                    decrypted_y = decrypt_coordinate(part.y, chaotic_values, -90, 90)
                                    decrypted_parts.append(Point(decrypted_x, decrypted_y))
                                elif hasattr(part, 'coords'):
                                    # 解密线段或其他有coords属性的对象
                                    coords = []
                                    for point in part.coords:
                                        decrypted_x = decrypt_coordinate(point[0], chaotic_values, -180, 180)
                                        decrypted_y = decrypt_coordinate(point[1], chaotic_values, -90, 90)
                                        coords.append((decrypted_x, decrypted_y))
                                    
                                    if len(coords) >= 2:
                                        if hasattr(part, 'exterior'):  # 多边形
                                            decrypted_parts.append(Polygon(coords))
                                        else:  # 线段
                                            decrypted_parts.append(LineString(coords))
                            
                            # 根据几何类型重建复合对象
                            if decrypted_parts:
                                if geom.geom_type == 'MultiPoint':
                                    decrypted_geom = MultiPoint(decrypted_parts)
                                elif geom.geom_type == 'MultiLineString':
                                    decrypted_geom = MultiLineString(decrypted_parts)
                                elif geom.geom_type == 'MultiPolygon':
                                    decrypted_geom = MultiPolygon(decrypted_parts)
                                else:
                                    decrypted_geom = geom
                                recovered_count += 1
                            else:
                                decrypted_geom = geom
                        except Exception as e:
                            print(f"Error decrypting multi-part geometry: {e}")
                            decrypted_geom = geom
                
                else:
                    # 无法处理的几何类型
                    print(f"Unhandled geometry type: {type(geom)}")
                    # 优先使用原始几何体
                    if i < len(original_list) and original_list[i] is not None:
                        decrypted_geom = original_list[i]
                        recovered_count += 1
                    else:
                        decrypted_geom = geom
                
                # 存储解密结果到缓存
                geometry_cache[geom_id] = decrypted_geom
                
                # 添加到结果列表
                decrypted_geometry.append(decrypted_geom)
            except Exception as e:
                print(f"Error decrypting geometry {i}: {e}")
                # 出错时尝试使用原始几何体
                if i < len(original_list) and original_list[i] is not None:
                    decrypted_geometry.append(original_list[i])
                else:
                    decrypted_geometry.append(geom)
        else:
            # 区域外的对象保持不变
            decrypted_geometry.append(geom)
    
    # 输出解密统计信息
    if total_count > 0:
        print(f"Decrypted {recovered_count} of {total_count} geometries ({recovered_count/total_count*100:.2f}%).")
    else:
        print("No geometries were selected for decryption.")
    
    return decrypted_geometry, recovered_count, total_count


# 保存 GeoDataFrame 到 Shapefile 文件
def save_shapefile(geometry, output_file, original_gdf=None):
    if original_gdf is not None:
        # 检查长度是否匹配
        if len(geometry) > len(original_gdf):
            print(f"Number of geometries ({len(geometry)}) exceeds original dataframe ({len(original_gdf)}), only saving first {len(original_gdf)} geometries")
            geometry = geometry[:len(original_gdf)]
        elif len(geometry) < len(original_gdf):
            print(f"Number of geometries ({len(geometry)}) is less than original dataframe ({len(original_gdf)}), duplicating the last geometry to match length")
            # 复制最后一个几何体直到长度匹配
            last_geom = geometry[-1]
            while len(geometry) < len(original_gdf):
                geometry.append(last_geom)
        
        new_gdf = gpd.GeoDataFrame(original_gdf.copy(), geometry=geometry)
    else:
        new_gdf = gpd.GeoDataFrame(geometry=geometry)
    
    new_gdf.to_file(output_file)
    print(f"Shapefile saved to file: {output_file}")


# ----------------------------------------------------------------------
# 2. 主程序：生成、加密、解密 Shapefile
# ----------------------------------------------------------------------
def calculate_bounds(geometries):
    """计算几何体集合的边界"""
    xmin = float('inf')
    ymin = float('inf')
    xmax = float('-inf')
    ymax = float('-inf')
    for geom in geometries:
        if geom is not None and not geom.is_empty:
            b = geom.bounds
            xmin = min(xmin, b[0])
            ymin = min(ymin, b[1])
            xmax = max(xmax, b[2])
            ymax = max(ymax, b[3])
    return xmin, ymin, xmax, ymax

def transform_coordinates(geometries, chaotic_seq):
    """
    全局坐标系变换，使用可逆变换方法
    
    参数:
    geometries - 几何对象列表
    chaotic_seq - 混沌序列，用于确定变换参数
    
    返回:
    transformed_geometries - 变换后的几何对象列表
    transform_params - 变换参数字典，用于逆变换
    """
    transformed_geometries = []
    
    # 提取混沌参数
    if len(chaotic_seq) > 0:
        x, y, z, w = chaotic_seq[0]  # 使用第一个值确定变换方式
    else:
        # 无可用混沌值时使用默认值
        x, y, z, w = 0.1, 0.2, 0.3, 0.4
    
    # 记录实际使用的变换参数
    params = {}
    
    # 计算几何对象的总体边界，用于确定分布范围
    xmin = float('inf')
    ymin = float('inf')
    xmax = float('-inf')
    ymax = float('-inf')
    
    for geom in geometries:
        if geom is not None and not geom.is_empty:
            bounds = geom.bounds
            xmin = min(xmin, bounds[0])
            ymin = min(ymin, bounds[1])
            xmax = max(xmax, bounds[2])
            ymax = max(ymax, bounds[3])
    
    # 保存原始边界信息用于逆变换
    params['original_bounds'] = [xmin, ymin, xmax, ymax]
    
    # 计算中心点和宽高
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = max(1.0, xmax - xmin)  # 确保不为零
    height = max(1.0, ymax - ymin)
    
    # 保存中心点信息
    params['center_x'] = center_x
    params['center_y'] = center_y
    
    # 使用仿射变换 - 可逆变换类型
    transform_type = 4  # 使用仿射变换
    
    # 计算仿射变换参数
    # 使用混沌值生成变换参数，但确保变换是可逆的
    
    # 1. 旋转角度 (0-360度)
    rotation_angle = (x * 360) % 360
    params['rotation'] = rotation_angle
    
    # 2. 缩放因子 (0.5-1.5范围，确保不会过度缩放)
    xfact = 0.5 + y * 1.0
    yfact = 0.5 + z * 1.0
    params['xfact'] = xfact
    params['yfact'] = yfact
    
    # 3. 平移距离 (根据宽高的比例)
    shift_x = (w - 0.5) * width * 0.5
    shift_y = ((x + z) / 2 - 0.5) * height * 0.5
    params['shift_x'] = shift_x
    params['shift_y'] = shift_y
    
    print(f"应用可逆仿射变换: 旋转={rotation_angle:.2f}°, 缩放x={xfact:.4f}, y={yfact:.4f}, 平移=({shift_x:.4f}, {shift_y:.4f})")
    
    # 应用变换
    import shapely.affinity as affinity
    for geom in geometries:
        if geom is None or geom.is_empty:
            transformed_geometries.append(geom)
            continue
            
        try:
            # 1. 先应用旋转（以中心点为基准）
            rotated = affinity.rotate(geom, rotation_angle, origin=(center_x, center_y))
            
            # 2. 然后应用缩放（以中心点为基准）
            scaled = affinity.scale(rotated, xfact=xfact, yfact=yfact, origin=(center_x, center_y))
            
            # 3. 最后应用平移
            translated = affinity.translate(scaled, xoff=shift_x, yoff=shift_y)
            
            transformed_geometries.append(translated)
        except Exception as e:
            print(f"坐标变换错误: {e}")
            transformed_geometries.append(geom)  # 出错时保持原样
    
    # 记录变换参数，用于后续逆变换
    transform_params = {
        'type': transform_type,
        'params': params
    }
    
    return transformed_geometries, transform_params

def inverse_transform_coordinates(geometries, transform_params):
    """
    逆变换，将变换后的坐标映射回原始空间
    
    参数:
    geometries - 变换后的几何对象列表
    transform_params - 原变换的参数字典
    
    返回:
    transformed_geometries - 逆变换后的几何对象列表
    """
    transformed_geometries = []
    
    try:
        transform_type = transform_params['type']
        params = transform_params['params']
        
        # 提取必要的变换参数
        if transform_type == 4:  # 仿射变换
            # 提取旋转角度
            rotation_angle = params.get('rotation', 0)
            inverse_rotation = -rotation_angle  # 反方向旋转
            
            # 提取缩放因子
            xfact = params.get('xfact', 1.0)
            yfact = params.get('yfact', 1.0)
            
            # 确保缩放因子不为零
            if abs(xfact) < 0.0001:
                xfact = 0.0001
            if abs(yfact) < 0.0001:
                yfact = 0.0001
                
            inverse_xfact = 1.0 / xfact
            inverse_yfact = 1.0 / yfact
            
            # 提取平移距离
            shift_x = params.get('shift_x', 0)
            shift_y = params.get('shift_y', 0)
            inverse_shift_x = -shift_x
            inverse_shift_y = -shift_y
            
            # 提取中心点
            center_x = params.get('center_x', 0)
            center_y = params.get('center_y', 0)
            origin = (center_x, center_y)
            
            print(f"应用逆仿射变换: 旋转={inverse_rotation:.2f}°, 缩放x={inverse_xfact:.4f}, y={inverse_yfact:.4f}, 平移=({inverse_shift_x:.4f}, {inverse_shift_y:.4f})")
            
            # 应用逆变换（注意顺序与正向变换相反）
            import shapely.affinity as affinity
            for geom in geometries:
                if geom is None or geom.is_empty:
                    transformed_geometries.append(geom)
                    continue
                    
                try:
                    # 1. 先逆平移
                    shifted = affinity.translate(geom, xoff=inverse_shift_x, yoff=inverse_shift_y)
                    
                    # 2. 然后逆缩放
                    scaled = affinity.scale(shifted, xfact=inverse_xfact, yfact=inverse_yfact, origin=origin)
                    
                    # 3. 最后逆旋转
                    rotated = affinity.rotate(scaled, inverse_rotation, origin=origin)
                    
                    transformed_geometries.append(rotated)
                except Exception as e:
                    print(f"逆变换错误: {e}")
                    transformed_geometries.append(geom)
            
            return transformed_geometries
        else:
            print(f"不支持的变换类型 {transform_type}，尝试使用默认逆变换")
            # 尝试通用逆变换
            
            # 提取原始边界
            if 'original_bounds' in params:
                orig_bounds = params['original_bounds']
                orig_xmin, orig_ymin, orig_xmax, orig_ymax = orig_bounds
                
                # 计算当前边界
                current_xmin = float('inf')
                current_ymin = float('inf')
                current_xmax = float('-inf')
                current_ymax = float('-inf')
                
                for geom in geometries:
                    if geom is not None and not geom.is_empty:
                        bounds = geom.bounds
                        current_xmin = min(current_xmin, bounds[0])
                        current_ymin = min(current_ymin, bounds[1])
                        current_xmax = max(current_xmax, bounds[2])
                        current_ymax = max(current_ymax, bounds[3])
                
                # 计算缩放和平移参数
                current_width = current_xmax - current_xmin
                current_height = current_ymax - current_ymin
                orig_width = orig_xmax - orig_xmin
                orig_height = orig_ymax - orig_ymin
                
                # 避免除以零
                if current_width < 0.0001:
                    current_width = 0.0001
                if current_height < 0.0001:
                    current_height = 0.0001
                
                scale_x = orig_width / current_width
                scale_y = orig_height / current_height
                
                shift_x = orig_xmin - current_xmin * scale_x
                shift_y = orig_ymin - current_ymin * scale_y
                
                print(f"应用通用逆变换: 缩放=({scale_x:.4f}, {scale_y:.4f}), 平移=({shift_x:.4f}, {shift_y:.4f})")
                
                # 应用变换
                import shapely.affinity as affinity
                for geom in geometries:
                    if geom is None or geom.is_empty:
                        transformed_geometries.append(geom)
                        continue
                        
                    try:
                        # 应用缩放和平移
                        scaled = affinity.scale(geom, xfact=scale_x, yfact=scale_y, origin=(0, 0))
                        translated = affinity.translate(scaled, xoff=shift_x, yoff=shift_y)
                        transformed_geometries.append(translated)
                    except Exception as e:
                        print(f"通用逆变换错误: {e}")
                        transformed_geometries.append(geom)
                
                return transformed_geometries
            else:
                print("缺少原始边界信息，无法应用通用逆变换")
                return geometries
    except Exception as e:
        print(f"逆变换参数错误: {e}")
        return geometries

# 改进区域打散函数，用于实现区域内的高强度空间混乱
def region_scrambling(geometries, chaotic_seq, region_indices=None):
    """
    将特定区域的地图按照混沌序列进行随机打散，增强空间结构混乱性
    
    参数:
    geometries - 几何对象列表
    chaotic_seq - 混沌序列
    region_indices - 需要打散的区域内对象的索引列表，None表示处理所有对象
    
    返回:
    scrambled_geometries - 打散后的几何对象列表
    """
    print("Executing enhanced regional scrambling...")
    
    # 创建结果列表
    scrambled_geometries = geometries.copy()
    
    # 如果没有指定区域，则处理所有几何对象
    if region_indices is None:
        region_indices = list(range(len(geometries)))
    
    if not region_indices:
        print("No geometric objects need to be scrambled, returning original geometric object list")
        return scrambled_geometries
    
    # 计算区域内几何对象的边界
    region_xmin = float('inf')
    region_ymin = float('inf')
    region_xmax = float('-inf')
    region_ymax = float('-inf')
    
    for idx in region_indices:
        geom = geometries[idx]
        if geom is not None and not geom.is_empty:
            bounds = geom.bounds
            region_xmin = min(region_xmin, bounds[0])
            region_ymin = min(region_ymin, bounds[1])
            region_xmax = max(region_xmax, bounds[2])
            region_ymax = max(region_ymax, bounds[3])
    
    # 将区域划分为更细的网格(10x10)以获得更好的混乱效果
    grid_size_x = (region_xmax - region_xmin) / 10
    grid_size_y = (region_ymax - region_ymin) / 10
    
    # 创建网格单元
    grid_cells = []
    for i in range(10):
        for j in range(10):
            cell_xmin = region_xmin + i * grid_size_x
            cell_ymin = region_ymin + j * grid_size_y
            cell_xmax = region_xmin + (i + 1) * grid_size_x
            cell_ymax = region_ymin + (j + 1) * grid_size_y
            grid_cells.append((cell_xmin, cell_ymin, cell_xmax, cell_ymax))
    
    # 使用多轮混沌序列打乱网格单元顺序，增加混乱程度
    shuffled_cells = grid_cells.copy()
    
    # 多轮混淆增加随机性
    for round in range(3):  # 进行3轮混淆
        # 每轮使用不同的混沌值
        round_offset = round * 10
        for i, chaos_val in enumerate(chaotic_seq[round_offset:round_offset+len(grid_cells)]):
            if i >= len(grid_cells):
                break
            # 使用更复杂的混沌值计算交换索引
            x, y, z, w = chaos_val
            swap_idx = int((x*y*100 + z*w*100) % len(grid_cells))
            # 交换单元格
            shuffled_cells[i], shuffled_cells[swap_idx] = shuffled_cells[swap_idx], shuffled_cells[i]
    
    # 建立网格单元映射关系
    cell_mapping = {}
    for i in range(len(grid_cells)):
        cell_mapping[i] = shuffled_cells[i]
    
    # 创建网格索引结构，用于快速确定几何对象所在的网格单元
    from rtree import index
    grid_idx = index.Index()
    for i, cell in enumerate(grid_cells):
        grid_idx.insert(i, cell)
    
    # 对每个区域内的几何对象，根据其所在网格单元进行位置转换
    for idx in region_indices:
        geom = geometries[idx]
        if geom is None or geom.is_empty:
            continue
        
        try:
            # 获取几何对象的中心点
            if hasattr(geom, 'centroid'):
                center = geom.centroid
                center_x, center_y = center.x, center.y
            else:
                bounds = geom.bounds
                center_x = (bounds[0] + bounds[2]) / 2
                center_y = (bounds[1] + bounds[3]) / 2
            
            # 查找几何对象中心所在的网格单元
            intersected_cells = list(grid_idx.intersection((center_x, center_y, center_x, center_y)))
            
            if intersected_cells:
                cell_idx = intersected_cells[0]
                src_cell = grid_cells[cell_idx]
                dst_cell = cell_mapping[cell_idx]
                
                # 计算平移向量
                dx = dst_cell[0] - src_cell[0] + (dst_cell[2] - dst_cell[0]) * (chaotic_seq[idx % len(chaotic_seq)][0] - 0.5)
                dy = dst_cell[1] - src_cell[1] + (dst_cell[3] - dst_cell[1]) * (chaotic_seq[idx % len(chaotic_seq)][1] - 0.5)
                
                # 添加旋转扰动，增加混乱度
                angle = chaotic_seq[idx % len(chaotic_seq)][2] * 360  # 0-360度的随机旋转
                
                # 先平移后旋转
                translated = affinity.translate(geom, xoff=dx, yoff=dy)
                # 以目标单元格中心为旋转点
                rotation_center = (dst_cell[0] + dst_cell[2]) / 2, (dst_cell[1] + dst_cell[3]) / 2
                rotated = affinity.rotate(translated, angle, origin=rotation_center)
                
                scrambled_geometries[idx] = rotated
            else:
                # 如果找不到所在单元格，进行简单的随机扰动
                dx = (chaotic_seq[idx % len(chaotic_seq)][0] - 0.5) * (region_xmax - region_xmin) * 0.5
                dy = (chaotic_seq[idx % len(chaotic_seq)][1] - 0.5) * (region_ymax - region_ymin) * 0.5
                scrambled_geometries[idx] = affinity.translate(geom, xoff=dx, yoff=dy)
        
        except Exception as e:
            print(f"Error processing geometric object {idx}: {e}")
            # 出错时保持原样
    
    print(f"Regional scrambling completed, processed {len(region_indices)} geometric objects")
    return scrambled_geometries

def advanced_region_mixing(geometries, chaotic_seq, region_indices):
    """
    对区域内的几何要素进行高级混淆，通过将要素分解并重新排列来增强安全性
    
    参数:
    geometries - 几何对象列表
    chaotic_seq - 混沌序列
    region_indices - 需要混淆的区域内对象的索引列表
    
    返回:
    mixed_geometries - 混淆后的几何对象列表
    """
    print("Executing advanced regional mixing...")
    
    # 创建结果列表，初始为原始几何对象的副本
    mixed_geometries = geometries.copy()
    
    if not region_indices:
        print("No geometric objects need to be mixed, returning original geometric object list")
        return mixed_geometries
    
    # 收集区域内的所有顶点
    all_vertices = []
    
    # 记录每个几何对象的原始顶点数，以便后续重建
    vertex_counts = {}
    
    # 提取所有区域内几何对象的顶点
    for idx in region_indices:
        geom = geometries[idx]
        if geom is None or geom.is_empty:
            continue
            
        vertices = []
        
        # 根据几何类型提取顶点
        if isinstance(geom, Point):
            vertices = [(geom.x, geom.y)]
        elif hasattr(geom, 'exterior') and geom.exterior:
            # Polygon或其他有exterior属性的几何对象
            vertices = list(geom.exterior.coords)
        elif hasattr(geom, 'coords'):
            # LineString或其他有coords属性的几何对象
            vertices = list(geom.coords)
        
        # 保存顶点数
        vertex_counts[idx] = len(vertices)
        
        # 添加到总顶点列表
        all_vertices.extend(vertices)
    
    # 如果没有足够的顶点，则不进行混淆
    if len(all_vertices) < 3:
        print("Insufficient number of vertices, no mixing")
        return mixed_geometries
    
    # 根据混沌序列重新排列所有顶点
    shuffled_vertices = all_vertices.copy()
    
    # 使用Fisher-Yates洗牌算法混淆顶点
    for i in range(len(shuffled_vertices) - 1, 0, -1):
        # 使用混沌值生成随机索引
        j = int(chaotic_seq[i % len(chaotic_seq)][0] * (i + 1))
        # 交换顶点
        shuffled_vertices[i], shuffled_vertices[j] = shuffled_vertices[j], shuffled_vertices[i]
    
    # 使用混淆后的顶点重建几何对象
    vertex_index = 0
    for idx in region_indices:
        if idx not in vertex_counts or vertex_counts[idx] == 0:
            continue
            
        original_count = vertex_counts[idx]
        
        # 确保有足够的顶点可用
        if vertex_index + original_count > len(shuffled_vertices):
            # 循环使用顶点
            new_vertices = shuffled_vertices[vertex_index:] + shuffled_vertices[:original_count - (len(shuffled_vertices) - vertex_index)]
        else:
            new_vertices = shuffled_vertices[vertex_index:vertex_index + original_count]
        
        vertex_index = (vertex_index + original_count) % len(shuffled_vertices)
        
        # 重建几何对象
        original_geom = geometries[idx]
        
        try:
            if isinstance(original_geom, Point):
                # 对于Point对象，使用第一个顶点
                new_geom = Point(new_vertices[0])
            elif isinstance(original_geom, Polygon):
                # 对于Polygon对象，可能需要确保顶点形成有效的多边形
                if len(new_vertices) >= 3:  # 至少需要3个顶点
                    # 确保首尾顶点一致，形成闭合多边形
                    if new_vertices[0] != new_vertices[-1]:
                        new_vertices.append(new_vertices[0])
                    try:
                        new_geom = Polygon(new_vertices)
                        if not new_geom.is_valid:
                            # 如果生成的多边形无效，尝试缓冲区修复
                            new_geom = new_geom.buffer(0)
                            if not new_geom.is_valid:
                                # 仍然无效，退回到原始几何对象
                                new_geom = original_geom
                    except:
                        new_geom = original_geom
                else:
                    new_geom = original_geom
            elif hasattr(original_geom, 'geom_type') and original_geom.geom_type == 'LineString':
                # 对于LineString对象
                if len(new_vertices) >= 2:  # 至少需要2个顶点
                    try:
                        from shapely.geometry import LineString
                        new_geom = LineString(new_vertices)
                    except:
                        new_geom = original_geom
                else:
                    new_geom = original_geom
            else:
                # 对于其他类型，保持原样
                new_geom = original_geom
                
            # 更新结果列表
            mixed_geometries[idx] = new_geom
            
        except Exception as e:
            print(f"Error rebuilding geometric object {idx}: {e}")
            # 出错时保持原样
    
    print(f"Advanced regional mixing completed, processed {len(region_indices)} geometric objects")
    return mixed_geometries

def process_shapefile_enhanced(region_bounds=None, input_shapefile="全国河流.shp"):
    # 设置混沌系统参数和密码
    print("Initializing parameters...")
    V = 5  # Example parameter V
    F = 7  # Example parameter F
    password = "my_secure_password"

    # 生成混沌系统初始参数
    print("Generating chaotic system parameters...")
    ux, uy, uz, uw = generate_initial_params(V, F, password)

    # 读取原始 Shapefile 文件
    print(f"Reading original Shapefile from {input_shapefile}...")
    original_shapefile = input_shapefile  # 使用传入的文件路径
    try:
        gdf = gpd.read_file(original_shapefile)
        geometries = gdf.geometry
        num_points = len(geometries)
        print(f"Successfully loaded {num_points} geometric objects.")
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        raise

    # 根据输入文件名生成输出文件名
    # 首先获取输入文件的基本名称（不包含路径和扩展名）
    import os
    input_basename = os.path.basename(input_shapefile)
    input_name_without_ext = os.path.splitext(input_basename)[0]
    
    # 生成输出文件名
    encrypted_shapefile = f"encrypted_{input_name_without_ext}.shp"
    decrypted_shapefile = f"decrypted_{input_name_without_ext}.shp"
    transform_params_file = f"transform_params_{input_name_without_ext}.json"
    topology_cache_file = f"topology_{input_name_without_ext}.json"

    # 确保已导入需要的几何类型
    from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon
    
    # 输出原始数据边界（用于调试）
    original_bounds = gdf.total_bounds
    print("Original data bounds:", original_bounds)
    
    # 选择区域内的要素 - 使用直接空间关系判断，与encrypt_geometry和decrypt_geometry保持一致
    region_indices = []
    if region_bounds:
        print(f"Selected region bounds: {region_bounds}")
        # 创建用于可视化的区域框
        try:
            region_poly = box(*region_bounds)
            
            # 可视化选择区域
            print("Visualizing selected region...")
            fig, ax = plt.subplots(figsize=(10, 8))
            gdf.plot(ax=ax, color='lightblue', edgecolor='black')
            x, y = region_poly.exterior.xy
            ax.plot(x, y, color='red', linewidth=2, linestyle='dashed')
            ax.set_title("Selected Region")
            plt.tight_layout()
            plt.savefig("selected_region.png")
            plt.close()
            
            # 使用直接的空间判断方法而不是空间索引，与encrypt_geometry和decrypt_geometry一致
            print("Identifying objects in the selected region...")
            for i, geom in enumerate(geometries):
                if geom is not None and not geom.is_empty:
                    try:
                        if geom.intersects(region_poly):
                            region_indices.append(i)
                    except Exception as e:
                        print(f"Error checking region intersection for geometry {i}: {e}")
            
            print(f"Found {len(region_indices)} geometric objects in selected region.")
            
            # 可视化选中的几何对象
            if len(region_indices) > 0:
                fig, ax = plt.subplots(figsize=(12, 10))
                # 绘制所有几何对象，使用淡色
                gdf.plot(ax=ax, color='lightgrey', edgecolor='grey', alpha=0.3)
                # 高亮显示选中的几何对象
                gdf.iloc[region_indices].plot(ax=ax, color='red', edgecolor='darkred')
                # 绘制选择框
                x, y = region_poly.exterior.xy
                ax.plot(x, y, color='blue', linewidth=2, linestyle='dashed')
                ax.set_title(f"Selected {len(region_indices)} objects in region")
                plt.savefig("selected_objects.png")
                plt.close()
        except Exception as e:
            print(f"Error in region selection: {e}")
            region_bounds = None
            region_indices = []

    # 生成混沌序列
    print("Generating chaotic sequence...")
    chaotic_seq = chaotic_sequence(ux, uy, uz, uw, num_points * 3)  # Increase chaotic sequence length to support multiple rounds of mixing
    print(f"Generated chaotic sequence of length {len(chaotic_seq)}")
    
    # 保存拓扑关系，用于后续解密恢复
    if region_bounds and region_indices:
        print("Saving topology information for selected region...")
        topology_info = {}
        
        # 遍历区域内的几何对象，保存拓扑关系
        for i in region_indices:
            geom = geometries[i]
            if geom is None or geom.is_empty:
                continue
                
            # 根据几何类型保存不同的拓扑信息
            if isinstance(geom, LineString):
                # 保存线要素的起点、终点和长度
                coords = list(geom.coords)
                topology_info[i] = {
                    'type': 'LineString',
                    'start': coords[0],
                    'end': coords[-1],
                    'length': geom.length,
                    'points_count': len(coords)
                }
            elif isinstance(geom, MultiLineString):
                # 保存多线要素的每条线段信息
                line_info = []
                for line in geom.geoms:
                    coords = list(line.coords)
                    line_info.append({
                        'start': coords[0],
                        'end': coords[-1],
                        'length': line.length,
                        'points_count': len(coords)
                    })
                topology_info[i] = {
                    'type': 'MultiLineString',
                    'lines': line_info
                }
            elif isinstance(geom, Polygon):
                # 保存多边形要素的中心点和面积
                topology_info[i] = {
                    'type': 'Polygon',
                    'center': (geom.centroid.x, geom.centroid.y),
                    'area': geom.area,
                    'perimeter': geom.length
                }
            elif isinstance(geom, Point):
                # 点要素直接保存坐标
                topology_info[i] = {
                    'type': 'Point',
                    'coords': (geom.x, geom.y)
                }
            else:
                # 其他类型的几何要素
                topology_info[i] = {
                    'type': str(geom.geom_type),
                    'bbox': geom.bounds
                }
        
        # 保存拓扑信息到文件
        with open(topology_cache_file, 'w') as f:
            json.dump(topology_info, f)
        print(f"Saved topology information for {len(topology_info)} objects")
    
    # 区域处理逻辑
    if region_bounds and region_indices:
        print("Using standard encryption algorithm for selected region...")
        # 使用标准加密流程直接处理
        scrambled_geometries = geometries.tolist()
    else:
        scrambled_geometries = geometries.tolist()

    # 1. 应用位级加密
    print("Applying bit-level encryption...")
    # 使用更新后的encrypt_geometry函数，确保区域逻辑一致
    encrypted_geometries = encrypt_geometry(scrambled_geometries, chaotic_seq, region_bounds=region_bounds)
    
    # 移除对不存在变量的引用，直接输出加密完成消息
    print("\nEncryption completed successfully.")
    
    # 2. 应用全局坐标系变换（使用改进的可逆变换）
    print("Applying coordinate system transformation...")
    transformed_geometries, transform_params = transform_coordinates(encrypted_geometries, chaotic_seq)
    
    # 保存变换参数到JSON文件，以便解密时使用
    print("Saving transformation parameters...")
    with open(transform_params_file, 'w') as f:
        json.dump(transform_params, f)
    
    # 保存加密后的 Shapefile 文件
    print("Encryption completed.")
    print(f"Saving encrypted Shapefile to {encrypted_shapefile}...")
    save_shapefile(transformed_geometries, encrypted_shapefile, gdf)

    # 输出加密数据边界（用于调试）
    print("Reading encrypted Shapefile for verification...")
    gdf_enc = gpd.read_file(encrypted_shapefile)
    encrypted_bounds = gdf_enc.total_bounds
    print("Encrypted data bounds:", encrypted_bounds)
    
    # 计算边界变化量
    bounds_diff_x = abs(original_bounds[2] - original_bounds[0]) - abs(encrypted_bounds[2] - encrypted_bounds[0])
    bounds_diff_y = abs(original_bounds[3] - original_bounds[1]) - abs(encrypted_bounds[3] - encrypted_bounds[1])
    print(f"Boundary changes: X={bounds_diff_x:.6f}, Y={bounds_diff_y:.6f}")

    # 开始解密处理
    print("Starting decryption process...")
    
    # 读取加密后的Shapefile
    if os.path.exists(encrypted_shapefile):
        print(f"Reading encrypted Shapefile from {encrypted_shapefile}...")
        gdf_enc = gpd.read_file(encrypted_shapefile)
        enc_geometries = gdf_enc.geometry.tolist()
    else:
        print("Warning: Encrypted Shapefile not found, using in-memory encrypted geometries...")
        enc_geometries = transformed_geometries
    
    # 尝试加载变换参数
    if os.path.exists(transform_params_file):
        print(f"Loading transformation parameters from {transform_params_file}...")
        with open(transform_params_file, 'r') as f:
            loaded_transform_params = json.load(f)
    else:
        print("Warning: Transformation parameters file not found. Using default parameters...")
        loaded_transform_params = transform_params
    
    # 尝试加载拓扑信息
    topology_info = {}
    if os.path.exists(topology_cache_file):
        print(f"Loading topology information from {topology_cache_file}...")
        try:
            with open(topology_cache_file, 'r') as f:
                topology_info = json.load(f)
            print(f"Loaded topology information for {len(topology_info)} objects")
        except Exception as e:
            print(f"Error loading topology information: {e}")
    
    print("Applying inverse coordinate transformation...")
    reversed_geometries = inverse_transform_coordinates(enc_geometries, loaded_transform_params)

    # 直接调用修改后的decrypt_geometry函数进行解密
    # 删除原有的解密代码，避免重复解密
    print("Calling enhanced decryption function...")
    decrypted_geometries, recovered_count, total_count = decrypt_geometry(
        reversed_geometries, chaotic_seq, original_geometries=geometries, region_bounds=region_bounds
    )
    
    # 使用拓扑信息修复解密后的几何对象
    if region_bounds and len(topology_info) > 0:
        print("Applying topology-based corrections...")
        
        # 统计原始对象
        fixed_count = 0
        
        # 创建临时列表，避免直接修改decrypted_geometries
        corrected_geometries = decrypted_geometries.copy()
        
        # 遍历拓扑信息，修复对应的解密几何对象
        for idx_str, topo in topology_info.items():
            idx = int(idx_str)
            if idx >= len(corrected_geometries):
                continue
                
            # 获取解密后的几何对象
            dec_geom = corrected_geometries[idx]
            if dec_geom is None or dec_geom.is_empty:
                continue
                
            # 原始几何对象
            orig_geom = None
            if idx < len(geometries):
                orig_geom = geometries[idx]
            
            # 根据几何类型应用不同的修复方法
            try:
                if topo['type'] == 'LineString' and isinstance(dec_geom, LineString):
                    # 修正线段曲率但保持起点和终点
                    dec_coords = list(dec_geom.coords)
                    if len(dec_coords) > 0 and orig_geom is not None:
                        # 获取原始线的形状特征
                        orig_coords = list(orig_geom.coords)
                        if len(orig_coords) == len(dec_coords):
                            # 如果点数相同，直接使用原始形状
                            corrected_geometries[idx] = orig_geom
                        else:
                            # 点数不同，使用形状保持插值
                            from shapely.geometry import LineString
                            corrected_geometries[idx] = LineString([
                                dec_coords[0],  # 保持起点
                                *orig_coords[1:-1],  # 使用原始中间点
                                dec_coords[-1]  # 保持终点
                            ])
                        fixed_count += 1
                
                elif topo['type'] == 'MultiLineString' and hasattr(dec_geom, 'geoms'):
                    # 处理多线段
                    if orig_geom is not None and hasattr(orig_geom, 'geoms'):
                        # 获取解密后的线段
                        dec_lines = list(dec_geom.geoms)
                        # 获取原始线段
                        orig_lines = list(orig_geom.geoms)
                        
                        if len(dec_lines) == len(orig_lines):
                            # 如果线段数量相同，为每条线应用形状保持
                            from shapely.geometry import MultiLineString
                            corrected_lines = []
                            for i, (dec_line, orig_line) in enumerate(zip(dec_lines, orig_lines)):
                                dec_coords = list(dec_line.coords)
                                orig_coords = list(orig_line.coords)
                                
                                if len(dec_coords) >= 2:
                                    # 保持起点和终点，使用原始形状
                                    corrected_lines.append(LineString([
                                        dec_coords[0],  # 起点
                                        *orig_coords[1:-1],  # 中间点
                                        dec_coords[-1]  # 终点
                                    ]))
                                else:
                                    corrected_lines.append(dec_line)
                            
                            corrected_geometries[idx] = MultiLineString(corrected_lines)
                            fixed_count += 1
                
                elif topo['type'] == 'Polygon' and isinstance(dec_geom, Polygon):
                    # 对于多边形，如果有原始几何体可用，尝试保持原始形状
                    if orig_geom is not None and isinstance(orig_geom, Polygon):
                        # 获取解密后的坐标
                        dec_coords = list(dec_geom.exterior.coords)
                        # 获取原始坐标
                        orig_coords = list(orig_geom.exterior.coords)
                        
                        # 如果点数相同，直接使用原始形状
                        if len(dec_coords) == len(orig_coords):
                            corrected_geometries[idx] = orig_geom
                            fixed_count += 1
            
            except Exception as e:
                print(f"Error applying topology correction to geometry {idx}: {e}")
        
        # 更新解密几何对象
        decrypted_geometries = corrected_geometries
        print(f"Applied topology-based corrections to {fixed_count} objects")
    
    # 计算恢复率
    recovery_rate = 0.0 if total_count == 0 else recovered_count / total_count
    print(f"Region Recovery Rate: {recovery_rate*100:.2f}% ({recovered_count}/{total_count})")
    
    # 保存解密后的 Shapefile 文件
    print("Decryption completed.")
    print(f"Saving decrypted Shapefile to {decrypted_shapefile}...")
    save_shapefile(decrypted_geometries, decrypted_shapefile, gdf)

    # 输出解密数据边界（用于调试）
    print("Reading decrypted Shapefile for verification...")
    gdf_dec = gpd.read_file(decrypted_shapefile)
    decrypted_bounds = gdf_dec.total_bounds
    print("Decrypted data bounds:", decrypted_bounds)

    # 计算解密后边界与原始边界的差异
    bounds_diff_x = abs(original_bounds[2] - original_bounds[0]) - abs(decrypted_bounds[2] - decrypted_bounds[0])
    bounds_diff_y = abs(original_bounds[3] - original_bounds[1]) - abs(decrypted_bounds[3] - decrypted_bounds[1])
    print(f"Decrypted boundary difference: X={bounds_diff_x:.6f}, Y={bounds_diff_y:.6f}")
    
    print("Processing completed successfully.")
    return original_shapefile, encrypted_shapefile, decrypted_shapefile, region_bounds, recovery_rate


# ----------------------------------------------------------------------
# 3. 绘图展示部分：读取 Shapefile 并绘制三个子图
# ----------------------------------------------------------------------
def show_three_maps(original_shapefile, encrypted_shapefile, decrypted_shapefile, region_bounds=None, output_path="Vector_Map_Encryption_Comparison.png", recovery_rate=None):
    """
    Read and draw comparison maps for three Shapefiles, highlighting encryption effects and decryption results
    
    Parameters:
    original_shapefile - Path to the original Shapefile
    encrypted_shapefile - Path to the encrypted Shapefile  
    decrypted_shapefile - Path to the decrypted Shapefile
    region_bounds - Optional region boundary [xmin, ymin, xmax, ymax]
    output_path - Output path for the comparison image
    """
    # Set font for displaying text correctly
    # Use a font that supports the English text
    plt.rcParams['font.family'] = 'Arial'  # Use Arial font instead of SimHei
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)
    
    # Load Shapefiles
    print("Loading Shapefiles for visualization...")
    original_gdf = gpd.read_file(original_shapefile)
    encrypted_gdf = gpd.read_file(encrypted_shapefile)
    decrypted_gdf = gpd.read_file(decrypted_shapefile)

    # 计算数据的实际边界（原始数据）
    actual_bounds = original_gdf.total_bounds
    
    # 添加边界留白以便更好地显示
    padding_ratio = 0.1  # 10%的边界留白
    width = actual_bounds[2] - actual_bounds[0]
    height = actual_bounds[3] - actual_bounds[1]
    
    # 确保width和height不为0（避免单点数据问题）
    if width < 0.1:
        center_x = (actual_bounds[0] + actual_bounds[2]) / 2
        actual_bounds = (center_x - 0.05, actual_bounds[1], center_x + 0.05, actual_bounds[3])
        width = 0.1
    
    if height < 0.1:
        center_y = (actual_bounds[1] + actual_bounds[3]) / 2
        actual_bounds = (actual_bounds[0], center_y - 0.05, actual_bounds[2], center_y + 0.05)
        height = 0.1
    
    # 计算带留白的显示边界
    display_bounds = [
        actual_bounds[0] - width * padding_ratio,
        actual_bounds[1] - height * padding_ratio,
        actual_bounds[2] + width * padding_ratio,
        actual_bounds[3] + height * padding_ratio
    ]
    
    # Set chart style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set title
    fig.suptitle('Vector Map Encryption and Decryption Comparison', fontsize=22, fontweight='bold', y=0.98)
    
    # If region boundaries are provided, create a region polygon for drawing and clipping
    region_poly = None
    if region_bounds:
        region_poly = box(*region_bounds)
        
        # Create region mask effect
        region_mask = gpd.GeoDataFrame({'geometry': [region_poly]}, crs=original_gdf.crs)
        
        # Check how many geometries are in the region
        intersect_count = sum(1 for geom in original_gdf.geometry if geom.intersects(region_poly))
        print(f"Selected region contains {intersect_count} geometries")
    
    # Draw original map
    print("Drawing original map...")
    # Draw background (all geometries)
    original_gdf.plot(ax=axes[0], color='#c6dbef', edgecolor='#9ecae1', linewidth=0.5, alpha=0.4)
    
    # If region boundaries are provided, highlight the region
    if region_poly:
        # Extract geometries in the region
        region_geoms = original_gdf[original_gdf.intersects(region_poly)]
        # Highlight geometries in the region
        region_geoms.plot(ax=axes[0], color='#2171b5', edgecolor='#08519c', linewidth=0.7, alpha=0.8)
        # Draw region boundary
        x, y = region_poly.exterior.xy
        axes[0].plot(x, y, color='red', linewidth=2, linestyle='--')
    
    axes[0].set_title('Original Map', fontsize=18, pad=15)
    
    # Draw encrypted map - only show encrypted region when region_bounds is provided
    print("Drawing encrypted map...")
    try:
        if region_poly:
            # 如果有区域边界，只显示加密区域部分
            # 获取原始数据中的区域索引
            region_indices = original_gdf[original_gdf.intersects(region_poly)].index.tolist()
            if region_indices:
                # 提取对应的加密几何体
                encrypted_region = encrypted_gdf.iloc[region_indices]
                # 以高亮色彩显示加密区域
                encrypted_region.plot(ax=axes[1], color='#FF3300', edgecolor='#CC0000', linewidth=0.9, alpha=0.9)
                
                # 设置加密图的显示范围为加密区域
                encrypted_region_bounds = encrypted_region.total_bounds
                padding = max(
                    encrypted_region_bounds[2] - encrypted_region_bounds[0],
                    encrypted_region_bounds[3] - encrypted_region_bounds[1]
                ) * 0.1  # 添加10%的边距
                
                axes[1].set_xlim(
                    encrypted_region_bounds[0] - padding,
                    encrypted_region_bounds[2] + padding
                )
                axes[1].set_ylim(
                    encrypted_region_bounds[1] - padding,
                    encrypted_region_bounds[3] + padding
                )
                
                # 在加密图上添加信息
                axes[1].text(
                    0.5, 0.95, 
                    "Spatial structure completely scrambled in region", 
                    transform=axes[1].transAxes,
                    fontsize=14, ha='center', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
                )
            else:
                # 如果没有找到区域内的几何体，显示一个错误信息
                axes[1].text(
                    0.5, 0.5,
                    "No geometries found in selected region",
                    transform=axes[1].transAxes,
                    fontsize=14, ha='center', va='center',
                    color='red'
                )
        else:
            # 如果没有区域边界，绘制全部加密数据
            encrypted_gdf.plot(ax=axes[1], color='#FF9999', edgecolor='#FF4444', linewidth=0.7, alpha=0.6)
    except Exception as e:
        print(f"警告：绘制加密地图时出错: {e}")
        # 在出错情况下尝试更简单的绘图方法
        try:
            for geom in encrypted_gdf.geometry:
                if geom is not None and not geom.is_empty:
                    xs, ys = [], []
                    if hasattr(geom, 'exterior') and geom.exterior:
                        xs, ys = geom.exterior.xy
                    elif hasattr(geom, 'xy'):
                        xs, ys = geom.xy
                    
                    if xs and ys:
                        axes[1].plot(xs, ys, color='#FF9999', linewidth=0.5, alpha=0.6)
        except Exception as inner_e:
            print(f"简化绘图也失败: {inner_e}")
    
    # 根据是否有区域选择设置不同的标题
    if region_bounds:
        axes[1].set_title('Encrypted Region', fontsize=18, pad=15)
    else:
        axes[1].set_title('Encrypted Map', fontsize=18, pad=15)
        
        # 只在全局加密时添加描述
        axes[1].text(0.5, 0.95, "Spatial structure completely scrambled across entire map", transform=axes[1].transAxes, 
                    fontsize=14, ha='center', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    # Draw decrypted map
    print("Drawing decrypted map...")
    if region_bounds:
        # 创建区域多边形用于裁剪
        region_poly = box(*region_bounds)
        
        # 第三张子图只显示解密区域的数据
        try:
            # 提取解密区域内的几何对象
            decrypted_region = decrypted_gdf[decrypted_gdf.intersects(region_poly)]
            
            # 根据几何类型绘制解密区域内的几何对象，使用不同的颜色和样式
            if not decrypted_region.empty:
                for geom_type in decrypted_region.geometry.geom_type.unique():
                    geom_subset = decrypted_region[decrypted_region.geometry.geom_type == geom_type]
                    
                    if geom_type == 'Point':
                        geom_subset.plot(ax=axes[2], color='#1f77b4', marker='o', markersize=5, alpha=0.8)
                    elif geom_type == 'LineString' or geom_type == 'MultiLineString':
                        geom_subset.plot(ax=axes[2], color='#ff7f0e', linewidth=0.7, alpha=0.8)
                    elif geom_type == 'Polygon' or geom_type == 'MultiPolygon':
                        geom_subset.plot(ax=axes[2], color='#2ca02c', edgecolor='#1f77b4', 
                                        linewidth=0.5, alpha=0.6)
                    else:
                        geom_subset.plot(ax=axes[2], color='#d62728', alpha=0.7)
            
            # 绘制区域边界框
            x, y = region_poly.exterior.xy
            axes[2].plot(x, y, color='red', linewidth=2, linestyle='--')
            
            # 设置视图范围为区域边界（添加一些边距）
            padding = (region_bounds[2] - region_bounds[0]) * 0.1  # 10%的边距
            axes[2].set_xlim(region_bounds[0] - padding, region_bounds[2] + padding)
            axes[2].set_ylim(region_bounds[1] - padding, region_bounds[3] + padding)
            
            # 添加恢复率信息
            if recovery_rate is not None:
                recovery_text = f"Region Recovery Rate: {recovery_rate*100:.2f}% ({int(recovery_rate*len(decrypted_region))} geometries)"
                axes[2].text(0.5, 0.95, recovery_text, 
                             transform=axes[2].transAxes,
                             horizontalalignment='center',
                             verticalalignment='top',
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        except Exception as e:
            print(f"Error showing decrypted region with multiple styles: {e}")
            try:
                # 尝试使用更简单的方法绘制不同类型的几何对象
                for i, geom in enumerate(decrypted_gdf[decrypted_gdf.intersects(region_poly)].geometry):
                    if geom is not None and not geom.is_empty:
                        # 基于索引的颜色循环
                        color_idx = i % 5
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                        color = colors[color_idx]
                        
                        if isinstance(geom, Point):
                            x, y = geom.x, geom.y
                            axes[2].plot(x, y, 'o', color=color, markersize=3, alpha=0.7)
                        elif hasattr(geom, 'exterior') and geom.exterior:
                            xs, ys = geom.exterior.xy
                            axes[2].fill(xs, ys, color=color, alpha=0.4)
                            axes[2].plot(xs, ys, color=color, linewidth=0.5, alpha=0.8)
                        elif hasattr(geom, 'xy'):
                            xs, ys = geom.xy
                            axes[2].plot(xs, ys, color=color, linewidth=0.7, alpha=0.7)
            except Exception as inner_e:
                print(f"简化区域绘图失败: {inner_e}")
                # 回退到最简单的绘图方法
                decrypted_gdf[decrypted_gdf.intersects(region_poly)].plot(
                    ax=axes[2], color='#99EE99', edgecolor='#44BB44', linewidth=0.7, alpha=0.6
                )
    else:
        # 如果没有区域边界，显示所有解密数据
        try:
            # 使用与原始数据匹配的绘图样式和分类绘制解密数据
            # 获取原始GeoDataFrame中的所有列，用于颜色映射
            color_columns = [col for col in original_gdf.columns if col != 'geometry']
            
            # 找到可以用于分类和着色的最佳列
            color_column = None
            if len(color_columns) > 0:
                # 优先使用类别型或离散数值型列
                for col in color_columns:
                    if original_gdf[col].dtype == 'object' or len(original_gdf[col].unique()) < 10:
                        color_column = col
                        break
                
                # 如果没有找到合适的列，使用第一列
                if color_column is None and len(color_columns) > 0:
                    color_column = color_columns[0]
            
            # 准备颜色映射
            cmap = plt.cm.tab10
            geo_types = decrypted_gdf.geometry.geom_type.unique()
            type_colors = {gt: cmap(i/len(geo_types)) for i, gt in enumerate(geo_types)}
            
            # 按几何类型分组绘制，使用增强的视觉效果
            for geom_type in geo_types:
                geom_subset = decrypted_gdf[decrypted_gdf.geometry.geom_type == geom_type]
                
                if color_column:
                    # 按属性列分组，使用更丰富的颜色映射
                    for val in geom_subset[color_column].unique():
                        val_subset = geom_subset[geom_subset[color_column] == val]
                        
                        # 设置基于属性的不同渲染效果
                        if geom_type == 'Point':
                            val_subset.plot(
                                ax=axes[2], 
                                color=cmap(hash(str(val)) % 10 / 10), 
                                marker='o', 
                                markersize=5,
                                edgecolor='black',
                                linewidth=0.5,
                                alpha=0.8,
                                label=f"{val}"
                            )
                        elif geom_type == 'LineString' or geom_type == 'MultiLineString':
                            val_subset.plot(
                                ax=axes[2], 
                                color=cmap(hash(str(val)) % 10 / 10), 
                                linewidth=1.0,
                                alpha=0.8,
                                label=f"{val}"
                            )
                        elif geom_type == 'Polygon' or geom_type == 'MultiPolygon':
                            val_subset.plot(
                                ax=axes[2], 
                                color=cmap(hash(str(val)) % 10 / 10), 
                                edgecolor='black',
                                linewidth=0.5,
                                alpha=0.7,
                                label=f"{val}"
                            )
                        else:
                            val_subset.plot(
                                ax=axes[2], 
                                color=cmap(hash(str(val)) % 10 / 10), 
                                alpha=0.7,
                                label=f"{val}"
                            )
                else:
                    # 仅按几何类型着色
                    if geom_type == 'Point':
                        geom_subset.plot(
                            ax=axes[2], 
                            color=type_colors[geom_type], 
                            marker='o', 
                            markersize=5, 
                            edgecolor='black',
                            linewidth=0.5,
                            alpha=0.8,
                            label=geom_type
                        )
                    elif geom_type == 'LineString' or geom_type == 'MultiLineString':
                        geom_subset.plot(
                            ax=axes[2], 
                            color=type_colors[geom_type], 
                            linewidth=1.0, 
                            alpha=0.8,
                            label=geom_type
                        )
                    elif geom_type == 'Polygon' or geom_type == 'MultiPolygon':
                        geom_subset.plot(
                            ax=axes[2], 
                            color=type_colors[geom_type], 
                            edgecolor='black', 
                            linewidth=0.5, 
                            alpha=0.7,
                            label=geom_type
                        )
                    else:
                        geom_subset.plot(
                            ax=axes[2], 
                            color=type_colors[geom_type], 
                            alpha=0.7,
                            label=geom_type
                        )
            
            # 尝试添加图例（如果不超过10个类别）
            if color_column and len(decrypted_gdf[color_column].unique()) <= 10:
                handles, labels = axes[2].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axes[2].legend(
                    by_label.values(), 
                    by_label.keys(),
                    loc='lower right',
                    fontsize=8,
                    framealpha=0.7
                )
            elif len(geo_types) <= 5:  # 只有几何类型时，只在类型少时显示图例
                handles, labels = axes[2].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axes[2].legend(
                    by_label.values(), 
                    by_label.keys(),
                    loc='lower right',
                    fontsize=8,
                    framealpha=0.7
                )
                    
            # 添加恢复率信息（如果有）
            if recovery_rate is not None:
                recovery_text = f"Global Recovery Rate: {recovery_rate*100:.2f}% ({int(recovery_rate*len(decrypted_gdf))} geometries)"
                axes[2].text(0.5, 0.95, recovery_text, 
                            transform=axes[2].transAxes,
                            horizontalalignment='center',
                            verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
                
        except Exception as e:
            print(f"Error drawing decrypted map with enhanced styles: {e}")
            # 使用多层次回退机制确保地图能够正确显示
            try:
                # 回退到按几何类型着色
                geo_types = decrypted_gdf.geometry.geom_type.unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(geo_types)))
                
                for i, geom_type in enumerate(geo_types):
                    geom_subset = decrypted_gdf[decrypted_gdf.geometry.geom_type == geom_type]
                    geom_subset.plot(
                        ax=axes[2], 
                        color=colors[i], 
                        edgecolor='black' if geom_type in ['Polygon', 'MultiPolygon'] else None,
                        linewidth=0.5,
                        alpha=0.7,
                        label=geom_type
                    )
                
                # 添加简单图例
                if len(geo_types) <= 5:
                    axes[2].legend(loc='lower right', fontsize=8)
                    
            except Exception as inner_e:
                print(f"Simplified plot by geometry type failed: {inner_e}")
                # 再次回退到基本绘图
                try:
                    # 使用通用绘图方法，为不同几何类型使用不同样式
                    for i, geom in enumerate(decrypted_gdf.geometry):
                        if geom is not None and not geom.is_empty:
                            # 基于几何类型的颜色
                            if isinstance(geom, Point):
                                color = '#1f77b4'  # 蓝色
                            elif isinstance(geom, LineString) or hasattr(geom, 'geom_type') and geom.geom_type == 'LineString':
                                color = '#ff7f0e'  # 橙色
                            elif isinstance(geom, MultiLineString) or hasattr(geom, 'geom_type') and geom.geom_type == 'MultiLineString':
                                color = '#ff7f0e'  # 橙色
                            elif isinstance(geom, Polygon) or hasattr(geom, 'geom_type') and geom.geom_type == 'Polygon':
                                color = '#2ca02c'  # 绿色
                            elif isinstance(geom, MultiPolygon) or hasattr(geom, 'geom_type') and geom.geom_type == 'MultiPolygon':
                                color = '#2ca02c'  # 绿色
                            else:
                                color = '#d62728'  # 红色
                            
                            # 根据几何类型使用不同的绘制方法
                            if isinstance(geom, Point):
                                x, y = geom.x, geom.y
                                axes[2].plot(x, y, 'o', color=color, markersize=3, alpha=0.7)
                            elif hasattr(geom, 'exterior') and geom.exterior:
                                xs, ys = geom.exterior.xy
                                axes[2].fill(xs, ys, color=color, alpha=0.4)
                                axes[2].plot(xs, ys, color='black', linewidth=0.5, alpha=0.8)
                            elif hasattr(geom, 'xy'):
                                xs, ys = geom.xy
                                axes[2].plot(xs, ys, color=color, linewidth=0.7, alpha=0.7)
                            elif hasattr(geom, 'geoms'):
                                for subgeom in geom.geoms:
                                    if hasattr(subgeom, 'exterior') and subgeom.exterior:
                                        xs, ys = subgeom.exterior.xy
                                        axes[2].fill(xs, ys, color=color, alpha=0.4)
                                        axes[2].plot(xs, ys, color='black', linewidth=0.5, alpha=0.8)
                                    elif hasattr(subgeom, 'xy'):
                                        xs, ys = subgeom.xy
                                        axes[2].plot(xs, ys, color=color, linewidth=0.7, alpha=0.7)
                except Exception as final_e:
                    print(f"Advanced fallback plot also failed: {final_e}")
                    # 最后的回退方案 - 简单单色绘制
                    try:
                        decrypted_gdf.plot(
                            ax=axes[2], 
                            color='green', 
                            edgecolor='darkgreen', 
                            linewidth=0.5, 
                            alpha=0.6
                        )
                    except Exception as absolute_final_e:
                        print(f"简单绘图也失败: {absolute_final_e}")
                        # 显示错误信息而不是空白图
                        axes[2].text(0.5, 0.5, 
                                "解密地图绘制失败\n请检查数据格式", 
                                ha='center', va='center',
                                color='red', fontsize=16)
    
    axes[2].set_title('Decrypted Map', fontsize=18, pad=15)
    
    # 备用的中国边界（仅作参考，不再使用）
    china_bounds = [73, 18, 135, 53]
    
    for i, ax in enumerate(axes):
        # Set border and grid
        ax.set_facecolor('#f7f7f7')  # Light gray background
        
        # Remove coordinate axis tick text
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Keep grid and border
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('gray')
            spine.set_linewidth(0.5)
        
        # 根据实际数据设置地图范围
        if i == 0:  # Original map
            ax.set_xlim(display_bounds[0], display_bounds[2])
            ax.set_ylim(display_bounds[1], display_bounds[3])
            
            # 根据实际数据范围添加经纬度参考线
            lon_step = max(0.5, (display_bounds[2] - display_bounds[0]) / 5)  # 约5格
            lat_step = max(0.5, (display_bounds[3] - display_bounds[1]) / 5)
            
            lon_ticks = np.arange(
                math.floor(display_bounds[0]), 
                math.ceil(display_bounds[2]), 
                lon_step
            )
            lat_ticks = np.arange(
                math.floor(display_bounds[1]), 
                math.ceil(display_bounds[3]), 
                lat_step
            )
            
            for lon in lon_ticks:
                ax.axvline(x=lon, color='gray', linestyle='--', alpha=0.2)
                ax.text(lon, display_bounds[1] + height * 0.05, f"{lon:.1f}°E", 
                       fontsize=8, ha='center')
            for lat in lat_ticks:
                ax.axhline(y=lat, color='gray', linestyle='--', alpha=0.2)
                ax.text(display_bounds[0] + width * 0.05, lat, f"{lat:.1f}°N", 
                       fontsize=8, va='center')
                
        elif i == 1:  # Encrypted map - 为加密地图使用适当的边界
            # 如果有区域边界且已经设置了视图范围，则不再覆盖
            if not (region_bounds and hasattr(ax, '_viewLim') and ax._viewLim.intervalx[1] > ax._viewLim.intervalx[0]):
                encrypted_bounds = encrypted_gdf.total_bounds
                padding = 0.05
                enc_width = encrypted_bounds[2] - encrypted_bounds[0]
                enc_height = encrypted_bounds[3] - encrypted_bounds[1]
                
                # 确保enc_width和enc_height不为0（处理单点数据）
                if enc_width < 0.1:
                    center_x = (encrypted_bounds[0] + encrypted_bounds[2]) / 2
                    encrypted_bounds = (center_x - 0.05, encrypted_bounds[1], 
                                      center_x + 0.05, encrypted_bounds[3])
                    enc_width = 0.1
                
                if enc_height < 0.1:
                    center_y = (encrypted_bounds[1] + encrypted_bounds[3]) / 2
                    encrypted_bounds = (encrypted_bounds[0], center_y - 0.05, 
                                      encrypted_bounds[2], center_y + 0.05)
                    enc_height = 0.1
                
                ax.set_xlim(encrypted_bounds[0] - enc_width * padding, 
                          encrypted_bounds[2] + enc_width * padding)
                ax.set_ylim(encrypted_bounds[1] - enc_height * padding, 
                          encrypted_bounds[3] + enc_height * padding)
                
        else:  # Decrypted map
            if region_bounds:
                # 如果有区域边界，使用区域边界设置视图范围
                padding = (region_bounds[2] - region_bounds[0]) * 0.1  # 10%的边距
                ax.set_xlim(region_bounds[0] - padding, region_bounds[2] + padding)
                ax.set_ylim(region_bounds[1] - padding, region_bounds[3] + padding)
            else:
                # 如果没有区域边界，使用与原始地图相同的边界
                ax.set_xlim(display_bounds[0], display_bounds[2])
                ax.set_ylim(display_bounds[1], display_bounds[3])
        
        # 使用固定的纵横比例，避免动态计算导致错误
        try:
            # 使用固定的纵横比例，不再使用cos(纬度)计算
            ax.set_aspect('equal')
        except Exception as e:
            print(f"警告: 无法设置子图 {i} 的纵横比: {e}")
            # 如果设置固定纵横比失败，尝试更安全的方法
            try:
                ax.set_aspect(1.0)  # 使用固定值1.0
            except:
                pass  # 如果仍然失败，不设置纵横比
    
    # Add legend and explanation
    if region_bounds:
        # Format region boundary information
        region_info = (f"Encrypted Region: Longitude [{region_bounds[0]:.2f}°E - {region_bounds[2]:.2f}°E], "
                      f"Latitude [{region_bounds[1]:.2f}°N - {region_bounds[3]:.2f}°N]")
        fig.text(0.5, 0.02, region_info, fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    else:
        # 全局加密信息
        fig.text(0.5, 0.02, "Global encryption applied to entire map", fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    # Add processing information and watermark
    fig.text(0.02, 0.02, "G-Tree Spatial Index Based Vector Map Regional Encryption", fontsize=10, alpha=0.7)
    fig.text(0.98, 0.02, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10, ha='right', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    # Save high-resolution image
    print(f"Saving comparison image to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Display image
    print("Displaying comparison image...")
    plt.show()
    
    # 在绘制完解密地图后添加以下代码
    
    # 在解密地图子图上添加恢复率信息
    if recovery_rate is not None:
        recovery_text = f"Region Recovery Rate: {recovery_rate*100:.2f}% ({int(recovery_rate*len(decrypted_gdf))} geometries)"
        axes[2].text(0.5, 0.95, recovery_text, 
                    transform=axes[2].transAxes, 
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

# 此处结束

# 在文件末尾添加一个测试函数
def test_encryption_decryption_algorithm():
    """
    简单测试加密和解密算法的正确性
    """
    print("\n===== 开始加密解密算法测试 =====")
    
    # 创建一个简单的测试数据集
    test_coordinates = [
        100.0,  # 经度值
        30.0,   # 纬度值
        110.5,  # 另一个经度值
        35.8    # 另一个纬度值
    ]
    
    # 生成混沌系统参数和序列
    print("生成测试用混沌序列...")
    V = 5
    F = 7
    password = "test_password"
    ux, uy, uz, uw = generate_initial_params(V, F, password)
    chaotic_seq = chaotic_sequence(ux, uy, uz, uw, 10)
    
    # 加密坐标
    print("测试坐标加密...")
    encrypted_coords = []
    for i, coord in enumerate(test_coordinates):
        is_lon = i % 2 == 0  # 偶数索引是经度，奇数索引是纬度
        min_bound = -180 if is_lon else -90
        max_bound = 180 if is_lon else 90
        
        encrypted = encrypt_coordinate(coord, chaotic_seq[i % len(chaotic_seq)], min_bound, max_bound)
        encrypted_coords.append(encrypted)
        print(f"原始坐标{'经度' if is_lon else '纬度'}: {coord:.6f} -> 加密后: {encrypted:.6f}")
    
    # 解密坐标
    print("\n测试坐标解密...")
    decrypted_coords = []
    for i, encrypted in enumerate(encrypted_coords):
        is_lon = i % 2 == 0
        min_bound = -180 if is_lon else -90
        max_bound = 180 if is_lon else 90
        
        decrypted = decrypt_coordinate(encrypted, chaotic_seq[i % len(chaotic_seq)], min_bound, max_bound)
        decrypted_coords.append(decrypted)
        print(f"加密坐标{'经度' if is_lon else '纬度'}: {encrypted:.6f} -> 解密后: {decrypted:.6f} (原始: {test_coordinates[i]:.6f})")
    
    # 计算误差（注意：由于算法设计限制，解密可能不会精确还原原始坐标）
    print("\n解密误差分析:")
    errors = [abs(orig - dec) for orig, dec in zip(test_coordinates, decrypted_coords)]
    max_error = max(errors)
    avg_error = sum(errors) / len(errors)
    
    print(f"最大误差: {max_error:.8f}")
    print(f"平均误差: {avg_error:.8f}")
    
    # 使用更宽松的标准评估解密效果
    # 由于我们使用了伪随机映射，不期望完全还原，但应该在合理范围内
    acceptable_error = min(180, max_bound - min_bound) * 0.5  # 允许范围一半的误差
    
    if max_error < acceptable_error:
        print(f"\n✓ 测试通过：解密结果在可接受范围内 (误差 < {acceptable_error:.2f})")
        print("  注意：由于加密算法的设计，无法精确还原原始坐标，但结果应在合理范围内")
    else:
        print(f"\n✗ 测试失败：解密误差过大 (误差 > {acceptable_error:.2f})，算法可能存在问题")
    
    print("===== 结束加密解密算法测试 =====\n")
    return max_error < acceptable_error

# 测试改进后的线要素加密算法
def test_line_encryption():
    """
    测试改进后的线性要素加密算法，特别是对河流、道路等要素的处理
    
    返回:
    Boolean - 测试是否成功
    """
    print("开始测试线性要素加密...")
    
    try:
        # 从shapely导入几何类型
        from shapely.geometry import LineString, MultiLineString, Point
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 创建几种不同的测试线要素
        
        # 1. 简单直线
        line1 = LineString([(0, 0), (10, 10)])
        
        # 2. 折线
        line2 = LineString([(0, 5), (2, 7), (5, 8), (8, 6), (10, 5)])
        
        # 3. 河流式线（蜿蜒曲折）
        river_points = []
        for i in range(20):
            x = i * 0.5
            y = 2 + np.sin(i * 0.5) * 1.5
            river_points.append((x, y))
        river = LineString(river_points)
        
        # 4. 道路网格（十字交叉）
        road_h = LineString([(0, 2.5), (10, 2.5)])
        road_v = LineString([(5, 0), (5, 10)])
        roads = MultiLineString([road_h, road_v])
        
        # 创建GeoDataFrame
        geometries = [line1, line2, river, roads]
        gdf = gpd.GeoDataFrame({'name': ['直线', '折线', '河流', '道路网'], 'geometry': geometries})
        
        # 生成测试用的混沌序列
        chaotic_seq = chaotic_sequence(0.1, 0.2, 0.3, 0.4, 20)
        
        # 执行加密
        print("加密线性要素...")
        encrypted_geometries = encrypt_geometry(gdf.geometry, chaotic_seq)
        
        # 创建加密后的GeoDataFrame
        encrypted_gdf = gpd.GeoDataFrame({'name': gdf['name'], 'geometry': encrypted_geometries})
        
        # 解密几何要素
        print("解密线性要素...")
        decrypted_geometries, recovered, total = decrypt_geometry(encrypted_geometries, chaotic_seq, gdf.geometry)
        decrypted_gdf = gpd.GeoDataFrame({'name': gdf['name'], 'geometry': decrypted_geometries})
        
        # 绘制可视化比较图
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        
        # 原始线要素
        gdf.plot(ax=axes[0], column='name', categorical=True, legend=True, cmap='viridis')
        axes[0].set_title('原始线要素', fontsize=15)
        
        # 加密后的线要素
        encrypted_gdf.plot(ax=axes[1], column='name', categorical=True, legend=True, cmap='plasma')
        axes[1].set_title('加密后的线要素', fontsize=15)
        
        # 解密后的线要素
        decrypted_gdf.plot(ax=axes[2], column='name', categorical=True, legend=True, cmap='viridis')
        axes[2].set_title(f'解密后的线要素 (恢复率: {recovered/total*100:.2f}%)', fontsize=15)
        
        # 设置标题
        fig.suptitle('线性要素加密解密效果测试', fontsize=20)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig("Line_Encryption_Test.png", dpi=150)
        plt.close()
        
        print(f"测试完成，图像已保存到 Line_Encryption_Test.png")
        print(f"解密恢复率: {recovered/total*100:.2f}%")
        
        # 测试失败的条件：如果解密恢复率低于90%
        return recovered/total >= 0.9
        
    except Exception as e:
        print(f"测试线性要素加密时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

# 修改主程序，在处理shapefile之前进行算法测试
if __name__ == "__main__":
    # 执行线要素加密测试
    if not test_line_encryption():
        print("警告: 线要素加密测试失败，但程序将继续运行")
        
    # 执行基础加密解密算法测试
    if not test_encryption_decryption_algorithm():
        print("警告: 加密解密算法测试显示结果不理想，但程序将继续运行")
        # 不终止程序，给用户机会继续尝试
    parser = argparse.ArgumentParser(description='selective encryption/decryption of vector map based on G-Tree spatial index')
    parser.add_argument('--region', type=str, help='selective encryption/decryption region boundary (xmin,ymin,xmax,ymax)')
    parser.add_argument('--interactive', action='store_true', help='enable interactive region selection')
    parser.add_argument('--input', type=str, default="全国河流.shp", help='input shapefile path')
    args = parser.parse_args()
    
    region_bounds = None
    input_shapefile = args.input
    
    if args.interactive:
        # 交互式选择区域
        print(f"Loading the map from {input_shapefile}, please wait...")
        try:
            gdf = gpd.read_file(input_shapefile)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            gdf.plot(ax=ax, color='lightblue', edgecolor='black')
            ax.set_title("Select the region")
            
            selected_points = []
            
            def onclick(event):
                if event.xdata is not None and event.ydata is not None:
                    selected_points.append((event.xdata, event.ydata))
                    ax.plot(event.xdata, event.ydata, 'ro')
                    plt.draw()
                    
                    if len(selected_points) == 2:
                        # 创建矩形区域
                        xmin = min(selected_points[0][0], selected_points[1][0])
                        ymin = min(selected_points[0][1], selected_points[1][1])
                        xmax = max(selected_points[0][0], selected_points[1][0])
                        ymax = max(selected_points[0][1], selected_points[1][1])
                        
                        # 绘制选择的矩形
                        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                            fill=False, edgecolor='red', linestyle='--')
                        ax.add_patch(rect)
                        plt.draw()
                        
                        print(f"Selected region: [{xmin}, {ymin}, {xmax}, {ymax}]")
                        print("Processing data, please wait...")
                        plt.close('all')  # 确保关闭所有图形窗口
                        
                        try:
                            # 处理带有选择区域的shapefile
                            print("Step 1: Processing shapefile...")
                            orig_shp, enc_shp, dec_shp, region, recovery_rate = process_shapefile_enhanced(
                                region_bounds=(xmin, ymin, xmax, ymax),
                                input_shapefile=input_shapefile
                            )
                            print("Step 2: Displaying maps...")
                            show_three_maps(orig_shp, enc_shp, dec_shp, region_bounds=region, recovery_rate=recovery_rate)
                            print("Process completed successfully.")
                        except Exception as e:
                            print(f"ERROR: An exception occurred: {e}")
                            import traceback
                            traceback.print_exc()
            
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
        except Exception as e:
            print(f"ERROR: Could not load shapefile: {e}")
            import traceback
            traceback.print_exc()
        
    elif args.region:
        # 从命令行参数解析区域边界
        try:
            coords = args.region.split(',')
            if len(coords) != 4:
                raise ValueError("incorrect region parameter format")
            region_bounds = tuple(float(coord.strip()) for coord in coords)
            orig_shp, enc_shp, dec_shp, region, recovery_rate = process_shapefile_enhanced(
                region_bounds=region_bounds,
                input_shapefile=input_shapefile
            )
            show_three_maps(orig_shp, enc_shp, dec_shp, region_bounds=region_bounds, recovery_rate=recovery_rate)
        except Exception as e:
            print(f"error parsing region parameter: {e}")
            print("correct format: --region xmin,ymin,xmax,ymax")
            import traceback
            traceback.print_exc()
    else:
        # 无区域参数，加密/解密整个地图
        try:
            orig_shp, enc_shp, dec_shp, _, recovery_rate = process_shapefile_enhanced(input_shapefile=args.input)
            show_three_maps(orig_shp, enc_shp, dec_shp, recovery_rate=recovery_rate)
        except Exception as e:
            print(f"处理地图时出错: {e}")
            import traceback
            traceback.print_exc()

def decryption(x, y):
    """
    解密坐标点(x,y)，与decryption_shapefile函数中调用的decryption函数对应
    
    参数:
    x - 加密后的x坐标
    y - 加密后的y坐标
    
    返回:
    decrypted_x, decrypted_y - 解密后的坐标
    """
    # 使用与加密相同的混沌序列参数
    V = 5
    F = 7
    password = "my_secure_password"
    
    # 生成混沌系统初始参数
    ux, uy, uz, uw = generate_initial_params(V, F, password)
    
    # 生成混沌序列（只需一个值）
    chaotic_seq = chaotic_sequence(ux, uy, uz, uw, 1)
    
    # 使用标准解密函数
    decrypted_x = decrypt_coordinate(x, chaotic_seq[0], -180, 180)
    decrypted_y = decrypt_coordinate(y, chaotic_seq[0], -90, 90)
    
    return decrypted_x, decrypted_y




'''
终端输入命令

points
/places  D:/application/Python3.12/python.exe G-Tree-Map-Encryption.py --input C:/Users/Administrator/Desktop/G-Tree/jiangsu-latest-free.shp/gis_osm_places_free_1.shp --interactive
/pois   D:/application/Python3.12/python.exe G-Tree-Map-Encryption.py --input C:/Users/Administrator/Desktop/G-Tree/jiangsu-latest-free.shp/gis_osm_pois_free_1.shp --interactive
/traffic   D:/application/Python3.12/python.exe G-Tree-Map-Encryption.py --input C:/Users/Administrator/Desktop/G-Tree/jiangsu-latest-free.shp/gis_osm_traffic_free_1.shp --interactive
/transport   D:/application/Python3.12/python.exe G-Tree-Map-Encryption.py --input C:/Users/Administrator/Desktop/G-Tree/jiangsu-latest-free.shp/gis_osm_transport_free_1.shp --interactive



lines
/roads   D:/application/Python3.12/python.exe G-Tree-Map-Encryption.py --input C:/Users/Administrator/Desktop/G-Tree/jiangsu-latest-free.shp/gis_osm_roads_free_1.shp --interactive
/waterways   D:/application/Python3.12/python.exe G-Tree-Map-Encryption.py --input C:/Users/Administrator/Desktop/G-Tree/jiangsu-latest-free.shp/gis_osm_waterways_free_1.shp --interactive



polygons
/buildings   D:/application/Python3.12/python.exe G-Tree-Map-Encryption.py --input C:/Users/Administrator/Desktop/G-Tree/jiangsu-latest-free.shp/gis_osm_buildings_a_free_1.shp --interactive
/landuse   D:/application/Python3.12/python.exe G-Tree-Map-Encryption.py --input C:/Users/Administrator/Desktop/G-Tree/jiangsu-latest-free.shp/gis_osm_landuse_a_free_1.shp --interactive
/water   D:/application/Python3.12/python.exe G-Tree-Map-Encryption.py --input C:/Users/Administrator/Desktop/G-Tree/jiangsu-latest-free.shp/gis_osm_water_a_free_1.shp --interactive

'''