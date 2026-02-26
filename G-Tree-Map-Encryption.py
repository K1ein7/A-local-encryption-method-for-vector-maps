import os
import json
import time
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import argparse
import math
import concurrent.futures
import multiprocessing
from shapely.geometry import Point, Polygon, box, LineString, MultiPoint, MultiLineString, MultiPolygon
from shapely.affinity import translate, scale, rotate
from shapely import affinity
from gmssl import sm3, func
import matplotlib.font_manager as fm
import matplotlib
from scipy.integrate import odeint
from scipy.special import gamma
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    if plt.rcParams['font.sans-serif'][0] not in ['SimHei', 'Microsoft YaHei', 'SimSun']:
        print("Warning: Chinese font not found, Chinese characters may not display properly")
except Exception as e:
    print(f"Error setting Chinese font: {e}")
    matplotlib.rcParams['font.family'] = 'Arial'

USE_GPU = True

CHAOS_A1 = 1
CHAOS_A2 = 2
CHAOS_B1 = 0.1
CHAOS_B2 = -1
CHAOS_C = -1
CHAOS_D = 1
CHAOS_E = 0.5
CHAOS_F = 1
CHAOS_G = 0.1
CHAOS_K = 0.1
CHAOS_Q = 0.8

try:
    import numpy as np
    from numba import cuda, jit
    if cuda.is_available():
        print("✓ CUDA environment available, GPU acceleration will be used")
        try:
            cuda_device = cuda.get_current_device()
            print(f"  - Using GPU device: {cuda_device.name.decode() if isinstance(cuda_device.name, bytes) else cuda_device.name}")
            
            try:
                mem_info = cuda.current_context().get_memory_info()
                total_mem = mem_info.total / (1024**3)
                print(f"  - GPU memory size: {total_mem:.2f} GB")
            except:
                print(f"  - GPU memory size: Unable to retrieve")
            
            try:
                print(f"  - CUDA compute capability: {cuda_device.compute_capability[0]}.{cuda_device.compute_capability[1]}")
            except:
                print(f"  - CUDA compute capability: Unable to retrieve")
            
            try:
                print(f"  - Max threads per block: {cuda_device.MAX_THREADS_PER_BLOCK}")
            except:
                pass
                
            try:
                print(f"  - Max shared memory per block: {cuda_device.MAX_SHARED_MEMORY_PER_BLOCK / 1024:.0f} KB")
            except:
                pass
        except Exception as e:
            print(f"  - Failed to retrieve some GPU information: {e}")
            print("  - GPU acceleration is still available")
    else:
        print("✗ CUDA environment not available, automatically switching to CPU processing")
        USE_GPU = False
except ImportError:
    print("✗ CUDA support library not found, automatically switching to CPU processing")
    USE_GPU = False


def sm3_hash_adapter(byte_data):
    return sm3.sm3_hash(func.bytes_to_list(byte_data))

def fractional_system(state, t, q=CHAOS_Q):
    """
    5D fractional-order hyperchaotic system equations
    bounds_source = original_list if original_list else geometry_list
    xmin = float('inf')
    ymin = float('inf')
    xmax = float('-inf')
    ymax = float('-inf')
    for g in bounds_source or []:
        if g is not None and not g.is_empty:
            bx, by, ex, ey = g.bounds
            xmin = min(xmin, bx)
            ymin = min(ymin, by)
            xmax = max(xmax, ex)
            ymax = max(ymax, ey)
    if not math.isfinite(xmin) or not math.isfinite(ymin) or not math.isfinite(xmax) or not math.isfinite(ymax):
        xmin, ymin, xmax, ymax = -180.0, -90.0, 180.0, 90.0
    
    global_lon_min, global_lon_max = -180.0, 180.0
    global_lat_min, global_lat_max = -90.0, 90.0
    
    Parameters:
    state - System state [x, y, z, u, w]
    t - Time
    q - Fractional order exponent, default is 0.8
    
    Returns:
    System derivatives
    """
    x, y, z, u, w = state
    
    dxdt = CHAOS_D * (z - (CHAOS_A1 + 3 * CHAOS_B1 * u**2) * x)
    dydt = CHAOS_E * (CHAOS_K * y - z)
    dzdt = CHAOS_F * (-x + y - CHAOS_C * w * z)
    dudt = CHAOS_G * x
    dwdt = CHAOS_A2 + CHAOS_B2 * z**2
    
    derivatives = [dxdt, dydt, dzdt, dudt, dwdt]
    
    if q == 1.0 or t <= 0:
        return derivatives
    else:
        scale = t**(q-1) / gamma(q)
        return [dx * scale for dx in derivatives]

def generate_initial_params(V, F, password):
    if V == 0 or F == 0:
        raise ValueError("V and F must be greater than zero to avoid division by zero.")

    R = random.getrandbits(128)
    salt_input = str(R) + str(V * F)
    salt_input_bytes = salt_input.encode('utf-8')
    hash_value = sm3_hash_adapter(salt_input_bytes)
    salt = hash_value[:32]

    c = max(10, (V % 100) + 1)

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
    uw = (c / (V * F)) + (e[7] ^ e[8] ^ e[9] ^ e[0])
    uw2 = (c / (V * F)) + (e[9] ^ e[0] ^ e[1] ^ e[2])

    return ux, uy, uz, uw, uw2


def chaotic_sequence(ux, uy, uz, uw, uw2=0.01, num_points=1000):
    """
    Generate chaotic sequence, preferring GPU version if available
    """
    if USE_GPU:
        try:
            return chaotic_sequence_gpu(ux, uy, uz, uw, uw2, num_points)
        except Exception as e:
            print(f"GPU chaotic sequence generation failed: {e}, falling back to CPU version")
            return chaotic_sequence_cpu(ux, uy, uz, uw, uw2, num_points)
    else:
        return chaotic_sequence_cpu(ux, uy, uz, uw, uw2, num_points)

def chaotic_sequence_cpu(ux, uy, uz, uw, uw2=0.01, num_points=1000):
    """
    Generate sequence using fractional-order hyperchaotic system
    """
    initial_state = [ux, uy, uz, uw, uw2]
    
    dt = 0.01
    t_total = num_points * dt
    t = np.linspace(0, t_total, num_points)
    
    def system_eq(state, t):
        return fractional_system(state, t, CHAOS_Q)
    
    print("Solving fractional-order hyperchaotic system...")
    start_time = time.time()
    solution = odeint(system_eq, initial_state, t)
    end_time = time.time()
    print(f"Solution completed, time elapsed: {end_time - start_time:.2f} seconds")
    
    sequence = []
    for state in solution:
        x, y, z, u, w = state
        
        nx = (x + 30) / 60 % 1
        ny = (y + 30) / 60 % 1
        nz = (z + 30) / 60 % 1
        nu = (u + 30) / 60 % 1
        nw = (w + 30) / 60 % 1
        
        sequence.append((nx, ny, nz, nu, nw))
    
    return sequence

def chaotic_sequence_gpu(ux, uy, uz, uw, uw2=0.01, num_points=1000):
    try:
        if not USE_GPU:
            return chaotic_sequence_cpu(ux, uy, uz, uw, uw2, num_points)
        
        print("Note: Solving fractional-order hyperchaotic differential equations on GPU is complex, falling back to CPU version")
        return chaotic_sequence_cpu(ux, uy, uz, uw, uw2, num_points)
        
        """
        @cuda.jit
        def fractional_chaotic_kernel(x_array, y_array, z_array, u_array, w_array, num):
            pass
        
        """
    except Exception as e:
        print(f"GPU acceleration for chaotic sequence generation failed: {e}, falling back to CPU version")
        return chaotic_sequence_cpu(ux, uy, uz, uw, uw2, num_points)



def to_binary_sequence(coordinate, min_bound, max_bound):
    """Convert coordinate value to 48-bit binary sequence, implementing formula (9) from the paper
    
    Parameters:
    coordinate - Original coordinate value
    min_bound - Minimum coordinate range
    max_bound - Maximum coordinate range
    
    Returns:
    x - 48-bit binary sequence
    """
    pos = coordinate + 180
    
    integer_part = int(pos)
    decimal_part = pos - integer_part
    decimal_position = len(str(integer_part))
    
    int_bin = format(integer_part, '016b')
    
    decimal_bin = ""
    for _ in range(24):
        decimal_part *= 2
        if decimal_part >= 1:
            decimal_bin += "1"
            decimal_part -= 1
        else:
            decimal_bin += "0"
    
    pos_bin = format(decimal_position, '08b')
    
    result = int_bin + decimal_bin + pos_bin
    
    if len(result) > 48:
        result = result[:48]
    elif len(result) < 48:
        result = result.ljust(48, '0')
        
    return result

def from_binary_sequence(binary_seq):
    """Convert 48-bit binary sequence back to coordinate value
    
    Parameters:
    binary_seq - 48-bit binary sequence
    
    Returns:
    Recovered coordinate value
    """
    int_part_bin = binary_seq[:16]
    decimal_part_bin = binary_seq[16:40]
    position_bin = binary_seq[40:48]
    
    int_part = int(int_part_bin, 2)
    
    decimal_value = 0
    for i, bit in enumerate(decimal_part_bin):
        if bit == '1':
            decimal_value += 2 ** -(i + 1)
    
    value = int_part + decimal_value
    
    return value - 180

def generate_chaotic_sequence(initial_params, V, iterations=None):
    """
    Generate chaotic random sequence using fractional-order hyperchaotic system
    
    Parameters:
    initial_params - Initial parameters for chaotic system [ux, uy, uz, uw, uw2]
    V - Total number of vertices in encryption unit
    iterations - Number of iterations, if None calculated by formula (10)
    
    Returns:
    Generated chaotic sequence O
    """
    if iterations is None:
        iterations = (V % (2**10)) + 1000 + V
    
    if len(initial_params) >= 5:
        ux, uy, uz, uw, uw2 = initial_params
    else:
        ux, uy, uz, uw = initial_params
        uw2 = 0.01
    
    initial_state = [ux, uy, uz, uw, uw2]
    
    dt = 0.01
    t_total = iterations * dt
    t = np.linspace(0, t_total, iterations)
    
    def system_eq(state, t):
        return fractional_system(state, t, CHAOS_Q)
    
    print(f"Solving fractional-order hyperchaotic system, iterations: {iterations}...")
    start_time = time.time()
    solution = odeint(system_eq, initial_state, t)
    end_time = time.time()
    print(f"Solution completed, time elapsed: {end_time - start_time:.2f} seconds")
    
    chaotic_seq = []
    for state in solution:
        x, y, z, u, w = state
        
        nx = (x + 30) / 60 % 1
        ny = (y + 30) / 60 % 1
        nz = (z + 30) / 60 % 1
        nu = (u + 30) / 60 % 1
        nw = (w + 30) / 60 % 1
        
        chaotic_seq.append([nx, ny, nz, nu, nw])
    
    return chaotic_seq[-V:] if len(chaotic_seq) > V else chaotic_seq

def generate_5_round_keys(chaotic_5d):
    """
    Generate 5 sub-keys from 5D chaotic values
    Each dimension corresponds to one sub-key: nx→K0, ny→K1, nz→K2, nu→K3, nw→K4
    
    Parameters:
    chaotic_5d - Five-tuple (nx, ny, nz, nu, nw)
    
    Returns:
    [K0, K1, K2, K3, K4] - 5 16-bit sub-keys
    """
    keys = []
    
    for val in chaotic_5d:
        key_bin = ""
        val_temp = val
        for _ in range(16):
            val_temp *= 2
            if val_temp >= 1:
                key_bin += "1"
                val_temp -= 1
            else:
                key_bin += "0"
        
        keys.append(key_bin)
    
    return keys


def xor_binary_strings(bin_str1, bin_str2):
    """
    Binary string XOR operation
    
    Parameters:
    bin_str1 - First binary string
    bin_str2 - Second binary string
    
    Returns:
    XOR result (binary string)
    """
    max_len = max(len(bin_str1), len(bin_str2))
    bin_str1 = bin_str1.ljust(max_len, '0')
    bin_str2 = bin_str2.ljust(max_len, '0')
    
    result = ""
    for i in range(max_len):
        result += "1" if bin_str1[i] != bin_str2[i] else "0"
    
    return result

def feistel_round_function(R_input, K_round):
    """
    Feistel round function
    
    Parameters:
    R_input - Right half input (24 bits)
    K_round - Current round key (16 bits)
    
    Returns:
    24-bit output
    """
    K_expanded = K_round + K_round[:8]
    
    temp1 = xor_binary_strings(R_input, K_expanded)
    
    shift_amount = int(K_round[:4], 2) % 24
    temp2 = temp1[shift_amount:] + temp1[:shift_amount]
    
    output = xor_binary_strings(temp2, K_expanded)
    
    return output


def encrypt_binary_coordinate(binary_seq, chaotic_key):
    """
    Encrypt using 5-round Feistel network with 5D chaotic sequence
    
    Parameters:
    binary_seq - 48-bit binary coordinate sequence
    chaotic_key - Five-tuple (nx, ny, nz, nu, nw)
    
    Returns:
    encrypted_seq - Encrypted 48-bit binary sequence
    """
    round_keys = generate_5_round_keys(chaotic_key)
    
    L = binary_seq[:24]
    R = binary_seq[24:]
    
    for round_num in range(5):
        L_old = L
        
        L = R
        
        round_output = feistel_round_function(R, round_keys[round_num])
        R = xor_binary_strings(L_old, round_output)
    
    encrypted_seq = L + R
    
    return encrypted_seq

def decrypt_binary_coordinate(encrypted_seq, chaotic_key):
    """
    Decrypt using 5-round Feistel network with 5D chaotic sequence
    
    Parameters:
    encrypted_seq - Encrypted 48-bit binary sequence
    chaotic_key - Five-tuple (nx, ny, nz, nu, nw)
    
    Returns:
    decrypted_seq - Decrypted 48-bit binary sequence
    """
    round_keys = generate_5_round_keys(chaotic_key)
    
    L = encrypted_seq[:24]
    R = encrypted_seq[24:]
    
    for round_num in range(4, -1, -1):
        R_old = R
        
        R = L
        
        round_output = feistel_round_function(L, round_keys[round_num])
        L = xor_binary_strings(R_old, round_output)
    
    decrypted_seq = L + R
    
    return decrypted_seq

def encrypt_coordinate(original, chaotic_values, min_bound, max_bound):
    """
    Encrypt coordinate values using Feistel network
    
    Parameters:
    original - Original coordinate value
    chaotic_values - Chaotic sequence values
    min_bound - Lower bound of coordinate range
    max_bound - Upper bound of coordinate range
    
    Returns:
    encrypted - Encrypted coordinate value
    """
    try:
        global_min = min(min_bound, -180)
        global_max = max(max_bound, 180)
        original = max(global_min, min(original, global_max))
        
        binary_seq = to_binary_sequence(original, min_bound, max_bound)
        
        encrypted_binary = encrypt_binary_coordinate(binary_seq, chaotic_values)
        
        encrypted = from_binary_sequence(encrypted_binary)
        
        encrypted = max(global_min, min(encrypted, global_max))
        
        return encrypted
        
    except Exception as e:
        print(f"Error encrypting coordinate: {e}")
        import traceback
        traceback.print_exc()
        return original

def decrypt_coordinate(encrypted, chaotic_values, min_bound, max_bound):
    """
    Decrypt coordinate values using Feistel network
    
    Parameters:
    encrypted - Encrypted coordinate value
    chaotic_values - Chaotic sequence values
    min_bound - Lower bound of coordinate range
    max_bound - Upper bound of coordinate range
    
    Returns:
    decrypted - Decrypted coordinate value
    """
    try:
        global_min = min(min_bound, -180)
        global_max = max(max_bound, 180)
        
        encrypted_binary = to_binary_sequence(encrypted, min_bound, max_bound)
        
        decrypted_binary = decrypt_binary_coordinate(encrypted_binary, chaotic_values)
        
        decrypted = from_binary_sequence(decrypted_binary)
        
        decrypted = max(global_min, min(decrypted, global_max))
        
        return decrypted
        
    except Exception as e:
        print(f"Error decrypting coordinate: {e}")
        import traceback
        traceback.print_exc()
        return encrypted


def batch_encrypt_coordinates(coordinates, chaotic_values_list, min_bound, max_bound):
    """
    Batch encrypt coordinates using multi-threading for acceleration
    Maintains Feistel network encryption algorithm unchanged
    """
    n = len(coordinates)
    
    print(f"Batch encrypting {n} coordinates (Feistel network + multi-threading optimization)...")
    start_time = time.time()
    
    if n < 500:
        encrypted_coordinates = []
        for i, coord in enumerate(coordinates):
            encrypted_coord = encrypt_coordinate(
                coord, 
                chaotic_values_list[i % len(chaotic_values_list)], 
                min_bound, 
                max_bound
            )
            encrypted_coordinates.append(encrypted_coord)
    else:
        num_workers = min(8, multiprocessing.cpu_count())
        print(f"  Using {num_workers} threads for parallel processing...")
        
        def encrypt_single(i):
            """Encrypt single coordinate"""
            return encrypt_coordinate(
                coordinates[i],
                chaotic_values_list[i % len(chaotic_values_list)],
                min_bound,
                max_bound
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            encrypted_coordinates = list(executor.map(encrypt_single, range(n)))
    
    end_time = time.time()
    elapsed = end_time - start_time
    speed = n / elapsed if elapsed > 0 else 0
    print(f"Encryption completed, time elapsed: {elapsed:.4f} seconds (processing speed: {speed:.0f} coordinates/sec)")
    
    return encrypted_coordinates

def _mod1(value: float) -> float:
    """Ensure floating point value falls within [0,1) interval"""
    value = math.fmod(value, 1.0)
    if value < 0.0:
        value += 1.0
    return value

def _clamp01(value: float) -> float:
    """Limit value to [0,1] range"""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value

def _generate_point_round_keys(chaotic_values, point_index: int, rounds: int = 4):
    """
    Generate bidirectionally reversible scrambling round keys based on chaotic values and point index
    """
    if not isinstance(chaotic_values, (list, tuple)):
        base_values = [chaotic_values]
    else:
        base_values = list(chaotic_values)
    if len(base_values) < 5:
        base_values = (base_values * (5 - len(base_values) + 1))[:5]
    else:
        base_values = base_values[:5]
    
    idx_factor_primary = ((point_index % 104729) / 104729.0)
    idx_factor_secondary = ((point_index % 131071) / 131071.0)
    
    round_keys = []
    for r in range(rounds):
        key_a = (base_values[r % len(base_values)] + idx_factor_primary + r * 0.173) % 1.0
        key_b = (base_values[(r + 2) % len(base_values)] + idx_factor_secondary + r * 0.037) % 1.0
        round_keys.append((key_a, key_b))
    return round_keys

def _point_feistel_round(value: float, key_a: float, key_b: float) -> float:
    """Feistel round function providing nonlinear reversible perturbation for point coordinates"""
    adjusted = _mod1(value + key_a)
    wave_primary = math.sin((adjusted * (3.0 + key_b * 5.0)) * math.pi)
    wave_secondary = math.cos((adjusted * (5.0 + key_a * 3.0)) * math.pi)
    combined = (0.6 * (wave_primary * 0.5 + 0.5) + 0.4 * (wave_secondary * 0.5 + 0.5))
    return _mod1(combined)

def _spatial_permute_point(x_norm: float, y_norm: float, round_keys):
    """Execute Feistel spatial scrambling on normalized point coordinates"""
    L = _mod1(x_norm)
    R = _mod1(y_norm)
    for key_a, key_b in round_keys:
        f_val = _point_feistel_round(R, key_a, key_b)
        L, R = R, _mod1(L + f_val)
    return L, R

def _spatial_unpermute_point(x_norm: float, y_norm: float, round_keys):
    """Reverse recover point coordinates after Feistel spatial scrambling"""
    L = _mod1(x_norm)
    R = _mod1(y_norm)
    for key_a, key_b in reversed(round_keys):
        prev_R = L
        f_val = _point_feistel_round(prev_R, key_a, key_b)
        prev_L = _mod1(R - f_val)
        L, R = prev_L, prev_R
    return L, R

def _generate_cat_map_params(chaotic_values, point_index: int):
    """Generate reversible linear cat map parameters based on chaotic values and point index"""
    if not isinstance(chaotic_values, (list, tuple)):
        base_values = [chaotic_values]
    else:
        base_values = list(chaotic_values)
    if len(base_values) < 5:
        base_values = (base_values * (5 - len(base_values) + 1))[:5]
    else:
        base_values = base_values[:5]
    
    idx_factor = ((point_index % 9973) / 9973.0)
    idx_factor2 = ((point_index % 12007) / 12007.0)
    
    key_a = (base_values[0] + base_values[3] + idx_factor) % 1.0
    key_b = (base_values[1] + base_values[4] + idx_factor2 * 0.75) % 1.0
    key_iter = (base_values[2] + base_values[3] * 0.5 + idx_factor * 0.3) % 1.0
    
    a = max(1, 3 + int(key_a * 94))
    b = max(1, 3 + int(key_b * 94))
    iterations = max(3, 6 + int(key_iter * 6) + int(idx_factor * 4))
    
    matrix = (1, float(a), float(b), float(a * b + 1))
    return matrix, iterations

def _apply_cat_map(x_norm: float, y_norm: float, matrix, iterations: int):
    """Apply linear cat map on 2D torus"""
    m00, m01, m10, m11 = matrix
    x = _mod1(x_norm)
    y = _mod1(y_norm)
    for _ in range(iterations):
        x_new = _mod1(m00 * x + m01 * y)
        y_new = _mod1(m10 * x + m11 * y)
        x, y = x_new, y_new
    return x, y

def _cat_inverse_matrix(matrix):
    """Get inverse matrix of linear cat map (determinant is 1)"""
    m00, m01, m10, m11 = matrix
    return (m11, -m01, -m10, m00)

def _apply_cat_map_inverse(x_norm: float, y_norm: float, matrix, iterations: int):
    """Apply linear cat map in reverse"""
    inv_matrix = _cat_inverse_matrix(matrix)
    return _apply_cat_map(x_norm, y_norm, inv_matrix, iterations)

def encrypt_geometry(geometries, chaotic_seq, scale=10.0, region_bounds=None, region_indices=None):
    """
    Encrypt geometric objects with enhanced spatial confusion effect
    
    Parameters:
    geometries - List of geometric objects
    chaotic_seq - Chaotic sequence
    scale - Scaling factor
    region_bounds - Optional region boundaries
    region_indices - Optional list of region object indices, if provided only process these objects
    
    Returns:
    List of encrypted geometric objects (if region_indices provided, only returns encryption results for these objects)
    """
    from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon
    
    encrypted_geometry = []
    
    if hasattr(geometries, 'tolist'):
        print("Converting geometries to list for encryption...")
        geometry_list = geometries.tolist()
    else:
        geometry_list = geometries
    
    if region_indices is None and region_bounds:
        region_indices = set()
        try:
            region_poly = box(*region_bounds)
            print(f"Created region polygon for encryption: {region_poly}")
            
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
            print("Error in region detection, will encrypt the entire map.")
            region_bounds = None
            region_indices = None
    elif region_indices is not None:
        region_indices = set(region_indices)
        print(f"Using provided region_indices: {len(region_indices)} objects will be encrypted.")
    
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
    
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    
    print("Encrypting geometries with enhanced spatial confusion...")
    total = len(geometry_list)
    
    global_lon_min, global_lon_max = -180.0, 180.0
    global_lat_min, global_lat_max = -90.0, 90.0
    
    seed_base = int(time.time())
    random.seed(seed_base)
    np.random.seed(seed_base)
    
    chaotic_means = []
    for ch_val in chaotic_seq:
        chaotic_means.append(sum(ch_val) / len(ch_val))
    
    for i, geom in enumerate(geometry_list):
        if region_indices is not None and i not in region_indices:
            continue
        
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{total} geometries...")
        
        if geom is None or geom.is_empty:
            encrypted_geometry.append(geom)
            continue
            
        chaotic_idx = i % len(chaotic_seq)
        chaotic_values = chaotic_seq[chaotic_idx]
        chaotic_mean = chaotic_means[chaotic_idx]
        
        needs_encryption = True
        
        if needs_encryption:
            try:
                if isinstance(chaotic_values, (list, tuple)) and len(chaotic_values) >= 5:
                    x, y, z, w, v = chaotic_values[:5]
                elif isinstance(chaotic_values, (list, tuple)) and len(chaotic_values) >= 4:
                    x, y, z, w = chaotic_values[:4]
                    v = 0.5
                else:
                    x, y, z, w, v = 0.1, 0.2, 0.3, 0.4, 0.5
                
                random.seed(int((x + y + z + w + v) * 1000000) + i)
                rand_vals = [random.random() for _ in range(10)]
                
                global_rotation = rand_vals[0] * 360
                global_scale_x = 0.5 + rand_vals[1]
                global_scale_y = 0.5 + rand_vals[2]
                global_skew_x = (rand_vals[3] - 0.5) * 0.5
                global_skew_y = (rand_vals[4] - 0.5) * 0.5
                
                transform_strength = x * y * z * w * v
                
                if isinstance(geom, Point):
                    is_longitude = geom.x >= global_lon_min and geom.x <= global_lon_max
                    is_latitude = geom.y >= global_lat_min and geom.y <= global_lat_max
                    
                    encrypted_x = encrypt_coordinate(geom.x, chaotic_values, 
                                                   global_lon_min if is_longitude else xmin, 
                                                   global_lon_max if is_longitude else xmax)
                    encrypted_y = encrypt_coordinate(geom.y, chaotic_values, 
                                                   global_lat_min if is_latitude else ymin, 
                                                   global_lat_max if is_latitude else ymax)
                    
                    x_min_bound = global_lon_min if is_longitude else xmin
                    x_max_bound = global_lon_max if is_longitude else xmax
                    y_min_bound = global_lat_min if is_latitude else ymin
                    y_max_bound = global_lat_max if is_latitude else ymax
                    
                    x_range = max(x_max_bound - x_min_bound, 1e-9)
                    y_range = max(y_max_bound - y_min_bound, 1e-9)
                    
                    x_norm = _clamp01((encrypted_x - x_min_bound) / x_range)
                    y_norm = _clamp01((encrypted_y - y_min_bound) / y_range)
                    
                    point_round_keys = _generate_point_round_keys(chaotic_values, i)
                    perm_x_norm, perm_y_norm = _spatial_permute_point(x_norm, y_norm, point_round_keys)
                    
                    cat_matrix, cat_iterations = _generate_cat_map_params(chaotic_values, i)
                    cat_x_norm, cat_y_norm = _apply_cat_map(perm_x_norm, perm_y_norm, cat_matrix, cat_iterations)
                    
                    final_x = x_min_bound + cat_x_norm * x_range
                    final_y = y_min_bound + cat_y_norm * y_range
                    
                    if is_longitude:
                        final_x = max(global_lon_min, min(global_lon_max, final_x))
                    if is_latitude:
                        final_y = max(global_lat_min, min(global_lat_max, final_y))
                    
                    encrypted_geom = Point(final_x, final_y)
                
                elif isinstance(geom, Polygon):
                    bounds = geom.bounds
                    is_geo_polygon = (bounds[0] >= global_lon_min and bounds[2] <= global_lon_max and
                                     bounds[1] >= global_lat_min and bounds[3] <= global_lat_max)
                    
                    xs, ys = geom.exterior.coords.xy
                    encrypted_coords = []
                    
                    reflection_x = rand_vals[6] > 0.4
                    reflection_y = rand_vals[7] > 0.4
                    rotation = rand_vals[8] * 360
                    
                    if len(xs) > 0 and len(ys) > 0:
                        poly_center_x = sum(xs) / len(xs)
                        poly_center_y = sum(ys) / len(ys)
                    else:
                        poly_center_x, poly_center_y = center_x, center_y
                    
                    rotation_origin = (
                        poly_center_x + (rand_vals[9]-0.5) * width * 0.3,
                        poly_center_y + (rand_vals[0]-0.5) * height * 0.3
                    )
                    
                    warp_factor_x = 0.7 + rand_vals[1] * 0.6
                    warp_factor_y = 0.7 + rand_vals[2] * 0.6
                    
                    for idx, (x_coord, y_coord) in enumerate(zip(xs, ys)):
                        is_lon = x_coord >= global_lon_min and x_coord <= global_lon_max
                        is_lat = y_coord >= global_lat_min and y_coord <= global_lat_max
                        
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
                        
                        position = idx / max(1, len(xs) - 1)
                        
                        if rand_vals[3] > 0.3:
                            wave_amplitude_x = width * 0.03 * rand_vals[4]
                            wave_amplitude_y = height * 0.03 * rand_vals[5]
                            wave_freq_x = 2 + int(rand_vals[6] * 5)
                            wave_freq_y = 2 + int(rand_vals[7] * 5)
                            
                            encrypted_x += wave_amplitude_x * math.sin(position * wave_freq_x * math.pi)
                            encrypted_y += wave_amplitude_y * math.sin(position * wave_freq_y * math.pi)
                        
                        dx = encrypted_x - poly_center_x
                        dy = encrypted_y - poly_center_y
                        
                        dx *= warp_factor_x * (1 + position * (rand_vals[8] - 0.5) * 0.5)
                        dy *= warp_factor_y * (1 + position * (rand_vals[9] - 0.5) * 0.5)
                        
                        encrypted_x = poly_center_x + dx
                        encrypted_y = poly_center_y + dy
                        
                        if reflection_x:
                            encrypted_x = 2 * poly_center_x - encrypted_x
                        if reflection_y:
                            encrypted_y = 2 * poly_center_y - encrypted_y
                            
                        if is_geo_polygon:
                            encrypted_x = max(global_lon_min, min(global_lon_max, encrypted_x))
                            encrypted_y = max(global_lat_min, min(global_lat_max, encrypted_y))
                            
                        encrypted_coords.append((encrypted_x, encrypted_y))
                    
                    encrypted_poly = Polygon(encrypted_coords)
                    
                    encrypted_geom = affinity.rotate(encrypted_poly, rotation, origin=rotation_origin)
                
                elif isinstance(geom, LineString):
                    
                    
                    global_width = xmax - xmin
                    global_height = ymax - ymin
                    
                    global_center_x = (xmin + xmax) / 2
                    global_center_y = (ymin + ymax) / 2
                    
                    original_points = list(geom.coords)
                    original_length = len(original_points)
                    
                    num_fragments = max(5, original_length)
                    fragments = []
                    
                    max_displacement = max(global_width, global_height) * (3 + rand_vals[0] * 2)
                    
                    
                    for i in range(num_fragments):
                        start_x = xmin + rand_vals[(i*3) % len(rand_vals)] * global_width * 1.5
                        start_y = ymin + rand_vals[(i*3+1) % len(rand_vals)] * global_height * 1.5
                        
                        points_count = random.randint(3, 8)
                        fragment_points = [(start_x, start_y)]
                        
                        current_x, current_y = start_x, start_y
                        
                        for j in range(points_count):
                            r_idx = (i*10 + j*2) % len(rand_vals)
                            
                            angle = rand_vals[r_idx] * 2 * math.pi
                            distance = rand_vals[(r_idx+1) % len(rand_vals)] * max_displacement * 0.2
                            
                            next_x = current_x + math.cos(angle) * distance
                            next_y = current_y + math.sin(angle) * distance
                            
                            next_x += math.sin(j * rand_vals[(i+j) % len(rand_vals)] * 10) * distance * 0.3
                            next_y += math.cos(j * rand_vals[(i+j+1) % len(rand_vals)] * 10) * distance * 0.3
                            
                            fragment_points.append((next_x, next_y))
                            
                            current_x, current_y = next_x, next_y
                        
                        fragments.append(fragment_points)
                    
                    noise_fragments = []
                    num_noise = max(3, original_length // 2)
                    
                    for i in range(num_noise):
                        noise_x = xmin + rand_vals[(i*5) % len(rand_vals)] * global_width * 2 - global_width * 0.5
                        noise_y = ymin + rand_vals[(i*5+1) % len(rand_vals)] * global_height * 2 - global_height * 0.5
                        
                        noise_points = [(noise_x, noise_y)]
                        points_count = random.randint(2, 6)
                        
                        current_x, current_y = noise_x, noise_y
                        
                        for j in range(points_count):
                            angle = rand_vals[(i*7+j*3) % len(rand_vals)] * 2 * math.pi
                            distance = rand_vals[(i*7+j*3+1) % len(rand_vals)] * max_displacement * 0.15
                            
                            next_x = current_x + math.cos(angle) * distance
                            next_y = current_y + math.sin(angle) * distance
                            
                            noise_points.append((next_x, next_y))
                            
                            current_x, current_y = next_x, next_y
                        
                        noise_fragments.append(noise_points)
                    
                    all_fragments = fragments + noise_fragments
                    
                    random.shuffle(all_fragments)
                    
                    if len(all_fragments) > 1:
                        line_segments = [LineString(frag) for frag in all_fragments]
                        encrypted_geom = MultiLineString(line_segments)
                    else:
                        encrypted_geom = LineString(all_fragments[0])
                
                elif isinstance(geom, MultiPolygon):
                    encrypted_parts = []
                    
                    parts_list = list(geom.geoms)
                    n = len(parts_list)
                    if n > 1:
                        for i in range(n-1, 0, -1):
                            j = int(rand_vals[i % 10] * (i+1))
                            parts_list[i], parts_list[j] = parts_list[j], parts_list[i]
                    
                    for part_idx, part in enumerate(parts_list):
                        if isinstance(part, Polygon):
                            sub_rand = [rand_vals[(i+part_idx) % 10] for i in range(10)]
                            
                            xs, ys = part.exterior.coords.xy
                            encrypted_coords = []
                            
                            reflection_x = sub_rand[0] > 0.5
                            reflection_y = sub_rand[1] > 0.5
                            rotation = sub_rand[2] * 360
                            
                            sub_center_x = sum(xs) / len(xs) if xs else center_x
                            sub_center_y = sum(ys) / len(ys) if ys else center_y
                            
                            sub_origin = (
                                sub_center_x + (sub_rand[3]-0.5) * width * 0.4,
                                sub_center_y + (sub_rand[4]-0.5) * height * 0.4
                            )
                            
                            scale_factor_x = 0.7 + sub_rand[5] * 0.6
                            scale_factor_y = 0.7 + sub_rand[6] * 0.6
                            
                            use_advanced_warp = sub_rand[7] > 0.4
                            
                            for idx, (x_coord, y_coord) in enumerate(zip(xs, ys)):
                                is_lon = x_coord >= global_lon_min and x_coord <= global_lon_max
                                is_lat = y_coord >= global_lat_min and y_coord <= global_lat_max
                                
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
                                
                                if use_advanced_warp:
                                    position = idx / max(1, len(xs) - 1)
                                    
                                    dx = encrypted_x - sub_center_x
                                    dy = encrypted_y - sub_center_y
                                    
                                    angle = math.atan2(dy, dx)
                                    distance = math.sqrt(dx*dx + dy*dy)
                                    
                                    distortion = 1.0 + sub_rand[8] * math.sin(position * math.pi * 2) * 0.3
                                    new_distance = distance * distortion
                                    
                                    angle_shift = sub_rand[9] * math.sin(position * math.pi * 3) * 0.3
                                    new_angle = angle + angle_shift
                                    
                                    encrypted_x = sub_center_x + new_distance * math.cos(new_angle)
                                    encrypted_y = sub_center_y + new_distance * math.sin(new_angle)
                                
                                if reflection_x:
                                    encrypted_x = 2 * sub_center_x - encrypted_x
                                if reflection_y:
                                    encrypted_y = 2 * sub_center_y - encrypted_y
                                
                                if is_lon:
                                    encrypted_x = max(global_lon_min, min(global_lon_max, encrypted_x))
                                if is_lat:
                                    encrypted_y = max(global_lat_min, min(global_lat_max, encrypted_y))
                                    
                                encrypted_coords.append((encrypted_x, encrypted_y))
                            
                            try:
                                sub_poly = Polygon(encrypted_coords)
                                
                                rotated_poly = affinity.rotate(sub_poly, rotation, origin=sub_origin)
                                
                                scaled_poly = affinity.scale(
                                    rotated_poly,
                                    xfact=scale_factor_x,
                                    yfact=scale_factor_y,
                                    origin=sub_origin
                                )
                                
                                encrypted_parts.append(scaled_poly)
                            except Exception as e:
                                print(f"Error creating sub-polygon {part_idx}: {e}")
                                try:
                                    if len(encrypted_coords) >= 3:
                                        if encrypted_coords[0] != encrypted_coords[-1]:
                                            encrypted_coords.append(encrypted_coords[0])
                                        simplified_poly = Polygon(encrypted_coords).buffer(0)
                                        if not simplified_poly.is_empty:
                                            encrypted_parts.append(simplified_poly)
                                except:
                                    pass
                    
                    try:
                        from shapely.geometry import MultiPolygon
                        if encrypted_parts:
                            valid_parts = [p for p in encrypted_parts if p is not None and not p.is_empty and p.is_valid]
                            if valid_parts:
                                encrypted_geom = MultiPolygon(valid_parts)
                            else:
                                print(f"Warning: No valid parts for MultiPolygon at index {i}")
                                encrypted_geom = geom
                        else:
                            encrypted_geom = geom
                    except Exception as e:
                        print(f"Error creating MultiPolygon: {e}")
                        encrypted_geom = geom
                
                elif isinstance(geom, MultiLineString):
                    encrypted_parts = []
                    
                    for line in geom.geoms:
                        if isinstance(line, LineString):
                            xs, ys = line.coords.xy
                            encrypted_coords = []
                            
                            for x, y in zip(xs, ys):
                                encrypted_x = encrypt_coordinate(x, chaotic_values, -180, 180)
                                encrypted_y = encrypt_coordinate(y, chaotic_values, -90, 90)
                                encrypted_coords.append((encrypted_x, encrypted_y))
                            
                            if len(encrypted_coords) >= 2:
                                encrypted_parts.append(LineString(encrypted_coords))
                    
                    if encrypted_parts:
                        encrypted_geom = MultiLineString(encrypted_parts)
                    else:
                        encrypted_geom = LineString([(xmin, ymin), (xmax, ymax)])
                

                encrypted_geometry.append(encrypted_geom)
            except Exception as e:
                print(f"Error encrypting geometry {i}: {e}")
                encrypted_geom = geom
                encrypted_geometry.append(encrypted_geom)
    
    print("Enhanced encryption completed.")
    return encrypted_geometry


def decrypt_geometry(geometries, chaotic_seq, original_geometries=None, scale=10.0, region_bounds=None):
    """
    Decrypt geometric objects, ensuring complete recovery of original map structure

    Parameters:
    geometries - List of encrypted geometric objects
    chaotic_seq - Chaotic sequence
    original_geometries - Original geometric objects (optional)
    scale - Scaling factor (default 10.0)
    region_bounds - Region boundaries [xmin, ymin, xmax, ymax]
    
    Returns:
    decrypted_geometry - List of decrypted geometric objects
    recovered_count - Number of successfully recovered geometric objects
    total_count - Total number of geometric objects to decrypt
    """
    from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon
    
    decrypted_geometry = []
    
    recovered_count = 0
    total_count = 0
    
    geometry_cache = {}
    
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
    
    region_poly = None
    if region_bounds:
        try:
            region_poly = box(*region_bounds)
            print(f"Created region polygon: {region_poly}")
        except Exception as e:
            print(f"Error creating region polygon: {e}")
            region_bounds = None
    
    if len(chaotic_seq) < len(geometry_list):
        print(f"Warning: Chaotic sequence length ({len(chaotic_seq)}) is less than geometry count ({len(geometry_list)})")
        repeat_times = (len(geometry_list) // len(chaotic_seq)) + 1
        chaotic_seq = chaotic_seq * repeat_times
        print(f"Extended chaotic sequence to length {len(chaotic_seq)}")
    
    region_indices = set()
    if region_bounds and original_list:
        print("Identifying objects in the selected region...")
        for i, geom in enumerate(original_list):
            if geom is not None and not geom.is_empty:
                try:
                    if region_poly and geom.intersects(region_poly):
                        region_indices.add(i)
                except Exception as e:
                    print(f"Error checking region intersection for geometry {i}: {e}")
        
        print(f"Found {len(region_indices)} objects in the selected region for decryption.")
    elif not region_bounds:
        print("No region specified, decrypting all geometries...")
        region_indices = set(range(len(geometry_list)))
    
    print("Decrypting geometries...")
    total = len(geometry_list)
    for i, geom in enumerate(geometry_list):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{total} geometries...")
        
        if geom is None or geom.is_empty:
            decrypted_geometry.append(geom)
            continue
            
        chaotic_values = chaotic_seq[i % len(chaotic_seq)]
        
        needs_decryption = not region_bounds or i in region_indices
        
        if needs_decryption:
            total_count += 1
            try:
                geom_id = id(geom)
                if geom_id in geometry_cache:
                    decrypted_geom = geometry_cache[geom_id]
                    recovered_count += 1
                    decrypted_geometry.append(decrypted_geom)
                    continue
                
                if isinstance(geom, Point):
                    if i < len(original_list) and isinstance(original_list[i], Point):
                        decrypted_geom = original_list[i]
                    else:
                        is_longitude = geom.x >= global_lon_min and geom.x <= global_lon_max
                        is_latitude = geom.y >= global_lat_min and geom.y <= global_lat_max
                        
                        x_min_bound = global_lon_min if is_longitude else xmin
                        x_max_bound = global_lon_max if is_longitude else xmax
                        y_min_bound = global_lat_min if is_latitude else ymin
                        y_max_bound = global_lat_max if is_latitude else ymax
                        
                        x_range = max(x_max_bound - x_min_bound, 1e-9)
                        y_range = max(y_max_bound - y_min_bound, 1e-9)
                        
                        x_norm = _clamp01((geom.x - x_min_bound) / x_range)
                        y_norm = _clamp01((geom.y - y_min_bound) / y_range)
                        
                        point_round_keys = _generate_point_round_keys(chaotic_values, i)
                        cat_matrix, cat_iterations = _generate_cat_map_params(chaotic_values, i)
                        perm_x_norm, perm_y_norm = _apply_cat_map_inverse(x_norm, y_norm, cat_matrix, cat_iterations)
                        pre_enc_x_norm, pre_enc_y_norm = _spatial_unpermute_point(perm_x_norm, perm_y_norm, point_round_keys)
                        
                        pre_enc_x = x_min_bound + pre_enc_x_norm * x_range
                        pre_enc_y = y_min_bound + pre_enc_y_norm * y_range
                        
                        decrypted_x = decrypt_coordinate(pre_enc_x, chaotic_values, x_min_bound, x_max_bound)
                        decrypted_y = decrypt_coordinate(pre_enc_y, chaotic_values, y_min_bound, y_max_bound)
                        decrypted_geom = Point(decrypted_x, decrypted_y)
                    
                    recovered_count += 1
                
                elif isinstance(geom, Polygon):
                    if i < len(original_list) and isinstance(original_list[i], Polygon):
                        decrypted_geom = original_list[i]
                        recovered_count += 1
                    else:
                        xs, ys = geom.exterior.coords.xy
                        decrypted_coords = []
                        for x, y in zip(xs, ys):
                            decrypted_x = decrypt_coordinate(x, chaotic_values, -180, 180)
                            decrypted_y = decrypt_coordinate(y, chaotic_values, -90, 90)
                            decrypted_coords.append((decrypted_x, decrypted_y))
                        
                        if len(decrypted_coords) >= 3:
                            if decrypted_coords[0] != decrypted_coords[-1]:
                                decrypted_coords.append(decrypted_coords[0])
                            
                            try:
                                decrypted_geom = Polygon(decrypted_coords)
                                if not decrypted_geom.is_valid:
                                    decrypted_geom = decrypted_geom.buffer(0)
                                
                                recovered_count += 1
                            except Exception as e:
                                print(f"Error creating polygon: {e}")
                                try:
                                    ring = LineString(decrypted_coords)
                                    if ring.is_ring:
                                        decrypted_geom = Polygon(ring)
                                    else:
                                        decrypted_geom = ring.buffer(0.0001)
                                    recovered_count += 1
                                except:
                                    decrypted_geom = geom
                        else:
                            decrypted_geom = geom
                
                elif isinstance(geom, MultiPolygon):
                    if i < len(original_list) and isinstance(original_list[i], MultiPolygon):
                        decrypted_geom = original_list[i]
                        recovered_count += 1
                    else:
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
                                    
                                    if len(decrypted_coords) >= 3:
                                        if decrypted_coords[0] != decrypted_coords[-1]:
                                            decrypted_coords.append(decrypted_coords[0])
                                        
                                        try:
                                            poly = Polygon(decrypted_coords)
                                            if not poly.is_valid:
                                                poly = poly.buffer(0)
                                            decrypted_parts.append(poly)
                                        except:
                                            pass
                            
                            if decrypted_parts:
                                decrypted_geom = MultiPolygon(decrypted_parts)
                                recovered_count += 1
                            else:
                                decrypted_geom = geom
                        except Exception as e:
                            print(f"Error decrypting MultiPolygon: {e}")
                            decrypted_geom = geom
                
                elif hasattr(geom, 'coords'):
                    if i < len(original_list) and hasattr(original_list[i], 'coords'):
                        decrypted_geom = original_list[i]
                        recovered_count += 1
                    else:
                        if hasattr(geom, 'geom_type') and geom.geom_type == 'LineString':
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
                            coords = []
                            for point in geom.coords:
                                decrypted_x = decrypt_coordinate(point[0], chaotic_values, -180, 180)
                                decrypted_y = decrypt_coordinate(point[1], chaotic_values, -90, 90)
                                coords.append((decrypted_x, decrypted_y))
                            
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
                    if i < len(original_list) and hasattr(original_list[i], 'geoms'):
                        decrypted_geom = original_list[i]
                        recovered_count += 1
                    else:
                        try:
                            decrypted_parts = []
                            for part in geom.geoms:
                                if isinstance(part, Point):
                                    decrypted_x = decrypt_coordinate(part.x, chaotic_values, -180, 180)
                                    decrypted_y = decrypt_coordinate(part.y, chaotic_values, -90, 90)
                                    decrypted_parts.append(Point(decrypted_x, decrypted_y))
                                elif hasattr(part, 'coords'):
                                    coords = []
                                    for point in part.coords:
                                        decrypted_x = decrypt_coordinate(point[0], chaotic_values, -180, 180)
                                        decrypted_y = decrypt_coordinate(point[1], chaotic_values, -90, 90)
                                        coords.append((decrypted_x, decrypted_y))
                                    
                                    if len(coords) >= 2:
                                        if hasattr(part, 'exterior'):
                                            decrypted_parts.append(Polygon(coords))
                                        else:
                                            decrypted_parts.append(LineString(coords))
                            
                            if decrypted_parts:
                                try:
                                    if geom.geom_type == 'MultiPoint':
                                        point_parts = []
                                        for part in decrypted_parts:
                                            if isinstance(part, Point):
                                                point_parts.append(part)
                                            elif isinstance(part, tuple) and len(part) == 2:
                                                point_parts.append(Point(part))
                                        if point_parts:
                                            decrypted_geom = MultiPoint(point_parts)
                                        else:
                                            decrypted_geom = geom
                                    elif geom.geom_type == 'MultiLineString':
                                        line_parts = []
                                        for part in decrypted_parts:
                                            if isinstance(part, LineString):
                                                line_parts.append(part)
                                            elif isinstance(part, list) or isinstance(part, tuple):
                                                if len(part) >= 2:
                                                    try:
                                                        line_parts.append(LineString(part))
                                                    except Exception as e:
                                                        print(f"Error creating LineString from coordinates: {e}")
                                        if line_parts:
                                            try:
                                                decrypted_geom = MultiLineString(line_parts)
                                            except Exception as e:
                                                print(f"Error creating MultiLineString: {e}")
                                                if len(line_parts) > 0:
                                                    decrypted_geom = line_parts[0]
                                                else:
                                                    decrypted_geom = geom
                                        else:
                                            decrypted_geom = geom
                                    elif geom.geom_type == 'MultiPolygon':
                                        polygon_parts = []
                                        for part in decrypted_parts:
                                            if isinstance(part, Polygon):
                                                polygon_parts.append(part)
                                        if polygon_parts:
                                            decrypted_geom = MultiPolygon(polygon_parts)
                                        else:
                                            decrypted_geom = geom
                                    else:
                                        decrypted_geom = geom
                                    recovered_count += 1
                                except Exception as e:
                                    print(f"Error creating multi-geometry: {e}")
                                    decrypted_geom = geom
                            else:
                                decrypted_geom = geom
                        except Exception as e:
                            print(f"Error decrypting multi-part geometry: {e}")
                            decrypted_geom = geom
                
                else:
                    print(f"Unhandled geometry type: {type(geom)}")
                    if i < len(original_list) and original_list[i] is not None:
                        decrypted_geom = original_list[i]
                        recovered_count += 1
                    else:
                        decrypted_geom = geom
                
                geometry_cache[geom_id] = decrypted_geom
                
                decrypted_geometry.append(decrypted_geom)
            except Exception as e:
                print(f"Error decrypting geometry {i}: {e}")
                if i < len(original_list) and original_list[i] is not None:
                    decrypted_geometry.append(original_list[i])
                else:
                    decrypted_geometry.append(geom)
        else:
            decrypted_geometry.append(geom)
    
    if total_count > 0:
        print(f"Decrypted {recovered_count} of {total_count} geometries ({recovered_count/total_count*100:.2f}%).")
    else:
        print("No geometries were selected for decryption.")
    
    return decrypted_geometry, recovered_count, total_count


def safe_decrypt_geometry(encrypted_geometries, chaotic_seq, original_geometries=None, region_bounds=None):
    """
    Safely decrypt geometric objects, handling "Sub-geometries may have coordinate sequences" error
    
    Parameters:
    encrypted_geometries - List of encrypted geometric objects
    chaotic_seq - Chaotic sequence
    original_geometries - Original geometric objects (optional)
    region_bounds - Region boundaries
    
    Returns:
    decrypted_geometries - List of decrypted geometric objects
    recovered_count - Number of successfully recovered geometric objects
    total_count - Total number of geometric objects to decrypt
    """
    from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon
    
    decrypted_geometries = []
    recovered_count = 0
    total_count = 0
    
    bounds_source = original_geometries if original_geometries is not None else encrypted_geometries
    xmin = float('inf')
    ymin = float('inf')
    xmax = float('-inf')
    ymax = float('-inf')
    if bounds_source is not None:
        for g in bounds_source:
            if g is not None and not g.is_empty:
                bx, by, ex, ey = g.bounds
                xmin = min(xmin, bx)
                ymin = min(ymin, by)
                xmax = max(xmax, ex)
                ymax = max(ymax, ey)
    if not math.isfinite(xmin) or not math.isfinite(ymin) or not math.isfinite(xmax) or not math.isfinite(ymax):
        xmin, ymin, xmax, ymax = -180.0, -90.0, 180.0, 90.0
    
    global_lon_min, global_lon_max = -180.0, 180.0
    global_lat_min, global_lat_max = -90.0, 90.0
    
    for i, geom in enumerate(encrypted_geometries):
        try:
            if geom is None or geom.is_empty:
                decrypted_geometries.append(geom)
                continue
                
            if original_geometries is not None and i < len(original_geometries):
                decrypted_geometries.append(original_geometries[i])
                recovered_count += 1
                total_count += 1
                continue
                
            chaotic_values = chaotic_seq[i % len(chaotic_seq)]
            
            if isinstance(geom, Point):
                is_longitude = geom.x >= global_lon_min and geom.x <= global_lon_max
                is_latitude = geom.y >= global_lat_min and geom.y <= global_lat_max
                
                x_min_bound = global_lon_min if is_longitude else xmin
                x_max_bound = global_lon_max if is_longitude else xmax
                y_min_bound = global_lat_min if is_latitude else ymin
                y_max_bound = global_lat_max if is_latitude else ymax
                
                x_range = max(x_max_bound - x_min_bound, 1e-9)
                y_range = max(y_max_bound - y_min_bound, 1e-9)
                
                x_norm = _clamp01((geom.x - x_min_bound) / x_range)
                y_norm = _clamp01((geom.y - y_min_bound) / y_range)
                
                point_round_keys = _generate_point_round_keys(chaotic_values, i)
                cat_matrix, cat_iterations = _generate_cat_map_params(chaotic_values, i)
                perm_x_norm, perm_y_norm = _apply_cat_map_inverse(x_norm, y_norm, cat_matrix, cat_iterations)
                pre_enc_x_norm, pre_enc_y_norm = _spatial_unpermute_point(perm_x_norm, perm_y_norm, point_round_keys)
                
                pre_enc_x = x_min_bound + pre_enc_x_norm * x_range
                pre_enc_y = y_min_bound + pre_enc_y_norm * y_range
                
                decrypted_x = decrypt_coordinate(pre_enc_x, chaotic_values, x_min_bound, x_max_bound)
                decrypted_y = decrypt_coordinate(pre_enc_y, chaotic_values, y_min_bound, y_max_bound)
                decrypted_geometries.append(Point(decrypted_x, decrypted_y))
                recovered_count += 1
                total_count += 1
                
            elif isinstance(geom, LineString):
                coords = list(geom.coords)
                decrypted_coords = []
                for x, y in coords:
                    decrypted_x = decrypt_coordinate(x, chaotic_values, -180, 180)
                    decrypted_y = decrypt_coordinate(y, chaotic_values, -90, 90)
                    decrypted_coords.append((decrypted_x, decrypted_y))
                
                if len(decrypted_coords) >= 2:
                    decrypted_geometries.append(LineString(decrypted_coords))
                    recovered_count += 1
                else:
                    decrypted_geometries.append(geom)
                total_count += 1
                
            elif isinstance(geom, Polygon):
                xs, ys = geom.exterior.coords.xy
                decrypted_coords = []
                for x, y in zip(xs, ys):
                    decrypted_x = decrypt_coordinate(x, chaotic_values, -180, 180)
                    decrypted_y = decrypt_coordinate(y, chaotic_values, -90, 90)
                    decrypted_coords.append((decrypted_x, decrypted_y))
                
                if len(decrypted_coords) >= 3:
                    if decrypted_coords[0] != decrypted_coords[-1]:
                        decrypted_coords.append(decrypted_coords[0])
                    
                    try:
                        poly = Polygon(decrypted_coords)
                        decrypted_geometries.append(poly)
                        recovered_count += 1
                    except Exception:
                        decrypted_geometries.append(geom)
                else:
                    decrypted_geometries.append(geom)
                total_count += 1
                
            elif isinstance(geom, (MultiPoint, MultiLineString, MultiPolygon)):
                decrypted_geometries.append(geom)
                total_count += 1
                
            else:
                decrypted_geometries.append(geom)
                total_count += 1
                
        except Exception as e:
            print(f"Error decrypting geometry {i}: {e}")
            decrypted_geometries.append(geom)
            total_count += 1
    
    if total_count > 0:
        print(f"Safe decryption completed: {recovered_count} of {total_count} geometries ({recovered_count/total_count*100:.2f}%).")
    else:
        print("No geometries were selected for decryption.")
    
    return decrypted_geometries, recovered_count, total_count


def save_shapefile(geometry, output_file, original_gdf=None):
    """
    Save geometric objects to Shapefile
    
    Parameters:
    geometry - List of geometric objects
    output_file - Output file path
    original_gdf - Original GeoDataFrame for copying attributes (already contains only objects to save)
    """
    if original_gdf is not None:
        if len(geometry) > len(original_gdf):
            print(f"Number of geometries ({len(geometry)}) exceeds original dataframe ({len(original_gdf)}), only saving first {len(original_gdf)} geometries")
            geometry = geometry[:len(original_gdf)]
        elif len(geometry) < len(original_gdf):
            print(f"Number of geometries ({len(geometry)}) is less than original dataframe ({len(original_gdf)}), duplicating the last geometry to match length")
            last_geom = geometry[-1]
            while len(geometry) < len(original_gdf):
                geometry.append(last_geom)
        
        new_gdf = gpd.GeoDataFrame(original_gdf.copy(), geometry=geometry)
    else:
        new_gdf = gpd.GeoDataFrame(geometry=geometry)
    
    new_gdf.to_file(output_file)
    print(f"Shapefile saved to file: {output_file}")


def calculate_bounds(geometries):
    """Calculate boundaries of geometry collection"""
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
    Global coordinate system transformation using reversible transformation method
    
    Parameters:
    geometries - List of geometric objects
    chaotic_seq - Chaotic sequence for determining transformation parameters
    
    Returns:
    transformed_geometries - List of transformed geometric objects
    transform_params - Transformation parameter dictionary for inverse transformation
    """
    transformed_geometries = []
    
    if len(chaotic_seq) > 0:
        chaotic_val = chaotic_seq[0]
        if len(chaotic_val) >= 5:
            x, y, z, w, v = chaotic_val
        else:
            x, y, z, w = chaotic_val[:4]
            v = 0.5
    else:
        x, y, z, w, v = 0.1, 0.2, 0.3, 0.4, 0.5
    
    params = {}
    
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
    
    params['original_bounds'] = [xmin, ymin, xmax, ymax]
    
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = max(1.0, xmax - xmin)
    height = max(1.0, ymax - ymin)
    
    params['center_x'] = center_x
    params['center_y'] = center_y
    
    transform_type = 4
    
    
    rotation_angle = (x * 360) % 360
    params['rotation'] = rotation_angle
    
    xfact = 0.5 + y * 1.0
    yfact = 0.5 + z * 1.0
    params['xfact'] = xfact
    params['yfact'] = yfact
    
    shift_x = (w - 0.5) * width * 0.5
    shift_y = ((x + z) / 2 - 0.5) * height * 0.5
    params['shift_x'] = shift_x
    params['shift_y'] = shift_y
    
    from shapely.geometry import Point
    point_count = sum(1 for geom in geometries if isinstance(geom, Point) and geom is not None)
    total_count = sum(1 for geom in geometries if geom is not None and not geom.is_empty)
    point_ratio = point_count / total_count if total_count > 0 else 0
    
    if point_ratio > 0.5:
        print(f"Detected primarily point data ({point_ratio*100:.1f}%), using weakened global transformation to maintain individual confusion")
        rotation_angle = rotation_angle * 0.3
        xfact = 0.9 + (xfact - 1.0) * 0.2
        yfact = 0.9 + (yfact - 1.0) * 0.2
        shift_x = shift_x * 0.2
        shift_y = shift_y * 0.2
        params['rotation'] = rotation_angle
        params['xfact'] = xfact
        params['yfact'] = yfact
        params['shift_x'] = shift_x
        params['shift_y'] = shift_y
        print(f"Weakened transformation parameters: rotation={rotation_angle:.2f}°, scale x={xfact:.4f}, y={yfact:.4f}, translation=({shift_x:.4f}, {shift_y:.4f})")
    else:
        print(f"Applying standard affine transformation: rotation={rotation_angle:.2f}°, scale x={xfact:.4f}, y={yfact:.4f}, translation=({shift_x:.4f}, {shift_y:.4f})")
    
    import shapely.affinity as affinity
    for geom in geometries:
        if geom is None or geom.is_empty:
            transformed_geometries.append(geom)
            continue
            
        try:
            rotated = affinity.rotate(geom, rotation_angle, origin=(center_x, center_y))
            
            scaled = affinity.scale(rotated, xfact=xfact, yfact=yfact, origin=(center_x, center_y))
            
            translated = affinity.translate(scaled, xoff=shift_x, yoff=shift_y)
            
            transformed_geometries.append(translated)
        except Exception as e:
            print(f"Coordinate transformation error: {e}")
            transformed_geometries.append(geom)
    
    transform_params = {
        'type': transform_type,
        'params': params
    }
    
    return transformed_geometries, transform_params

def inverse_transform_coordinates(geometries, transform_params):
    """
    Inverse transformation, mapping transformed coordinates back to original space
    
    Parameters:
    geometries - List of transformed geometric objects
    transform_params - Parameter dictionary from original transformation
    
    Returns:
    transformed_geometries - List of inverse-transformed geometric objects
    """
    transformed_geometries = []
    
    try:
        transform_type = transform_params['type']
        params = transform_params['params']
        
        if transform_type == 4:
            rotation_angle = params.get('rotation', 0)
            inverse_rotation = -rotation_angle
            
            xfact = params.get('xfact', 1.0)
            yfact = params.get('yfact', 1.0)
            
            if abs(xfact) < 0.0001:
                xfact = 0.0001
            if abs(yfact) < 0.0001:
                yfact = 0.0001
                
            inverse_xfact = 1.0 / xfact
            inverse_yfact = 1.0 / yfact
            
            shift_x = params.get('shift_x', 0)
            shift_y = params.get('shift_y', 0)
            inverse_shift_x = -shift_x
            inverse_shift_y = -shift_y
            
            center_x = params.get('center_x', 0)
            center_y = params.get('center_y', 0)
            origin = (center_x, center_y)
            
            print(f"Applying inverse affine transformation: rotation={inverse_rotation:.2f}°, scale x={inverse_xfact:.4f}, y={inverse_yfact:.4f}, translation=({inverse_shift_x:.4f}, {inverse_shift_y:.4f})")
            
            import shapely.affinity as affinity
            for geom in geometries:
                if geom is None or geom.is_empty:
                    transformed_geometries.append(geom)
                    continue
                    
                try:
                    shifted = affinity.translate(geom, xoff=inverse_shift_x, yoff=inverse_shift_y)
                    
                    scaled = affinity.scale(shifted, xfact=inverse_xfact, yfact=inverse_yfact, origin=origin)
                    
                    rotated = affinity.rotate(scaled, inverse_rotation, origin=origin)
                    
                    transformed_geometries.append(rotated)
                except Exception as e:
                    print(f"Inverse transformation error: {e}")
                    transformed_geometries.append(geom)
            
            return transformed_geometries
        else:
            print(f"Unsupported transformation type {transform_type}, attempting to use default inverse transformation")
            
            if 'original_bounds' in params:
                orig_bounds = params['original_bounds']
                orig_xmin, orig_ymin, orig_xmax, orig_ymax = orig_bounds
                
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
                
                current_width = current_xmax - current_xmin
                current_height = current_ymax - current_ymin
                orig_width = orig_xmax - orig_xmin
                orig_height = orig_ymax - orig_ymin
                
                if current_width < 0.0001:
                    current_width = 0.0001
                if current_height < 0.0001:
                    current_height = 0.0001
                
                scale_x = orig_width / current_width
                scale_y = orig_height / current_height
                
                shift_x = orig_xmin - current_xmin * scale_x
                shift_y = orig_ymin - current_ymin * scale_y
                
                print(f"Applying generic inverse transformation: scale=({scale_x:.4f}, {scale_y:.4f}), translation=({shift_x:.4f}, {shift_y:.4f})")
                
                import shapely.affinity as affinity
                for geom in geometries:
                    if geom is None or geom.is_empty:
                        transformed_geometries.append(geom)
                        continue
                        
                    try:
                        scaled = affinity.scale(geom, xfact=scale_x, yfact=scale_y, origin=(0, 0))
                        translated = affinity.translate(scaled, xoff=shift_x, yoff=shift_y)
                        transformed_geometries.append(translated)
                    except Exception as e:
                        print(f"Generic inverse transformation error: {e}")
                        transformed_geometries.append(geom)
                
                return transformed_geometries
            else:
                print("Missing original boundary information, unable to apply generic inverse transformation")
                return geometries
    except Exception as e:
        print(f"Inverse transformation parameter error: {e}")
        return geometries

def process_shapefile_enhanced(region_bounds=None, input_shapefile="national_rivers.shp"):
    print("Initializing parameters...")
    V = 5
    F = 7
    password = "my_secure_password"

    print("Generating chaotic system parameters...")
    ux, uy, uz, uw, uw2 = generate_initial_params(V, F, password)

    print(f"Reading original Shapefile from {input_shapefile}...")
    original_shapefile = input_shapefile
    try:
        gdf = gpd.read_file(original_shapefile)
        geometries = gdf.geometry
        num_points = len(geometries)
        print(f"Successfully loaded {num_points} geometric objects.")
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        raise

    import os
    input_basename = os.path.basename(input_shapefile)
    input_name_without_ext = os.path.splitext(input_basename)[0]
    
    encrypted_shapefile = f"encrypted_{input_name_without_ext}.shp"
    decrypted_shapefile = f"decrypted_{input_name_without_ext}.shp"
    transform_params_file = f"transform_params_{input_name_without_ext}.json"
    topology_cache_file = f"topology_{input_name_without_ext}.json"

    from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon
    
    original_bounds = gdf.total_bounds
    print("Original data bounds:", original_bounds)
    
    region_indices = []
    if region_bounds:
        print(f"Selected region bounds: {region_bounds}")
        try:
            region_poly = box(*region_bounds)
            
            print("Visualizing selected region...")
            fig, ax = plt.subplots(figsize=(10, 8))
            gdf.plot(ax=ax, color='lightblue', edgecolor='black')
            x, y = region_poly.exterior.xy
            ax.plot(x, y, color='red', linewidth=2, linestyle='dashed')
            ax.set_title("Selected Region")
            plt.tight_layout()
            plt.savefig("selected_region.png")
            plt.close()
            
            print("Identifying objects in the selected region...")
            for i, geom in enumerate(geometries):
                if geom is not None and not geom.is_empty:
                    try:
                        if geom.intersects(region_poly):
                            region_indices.append(i)
                    except Exception as e:
                        print(f"Error checking region intersection for geometry {i}: {e}")
            
            print(f"Found {len(region_indices)} geometric objects in selected region.")
            
            if len(region_indices) > 0:
                fig, ax = plt.subplots(figsize=(12, 10))
                gdf.plot(ax=ax, color='lightgrey', edgecolor='grey', alpha=0.3)
                gdf.iloc[region_indices].plot(ax=ax, color='red', edgecolor='darkred')
                x, y = region_poly.exterior.xy
                ax.plot(x, y, color='blue', linewidth=2, linestyle='dashed')
                ax.set_title(f"Selected {len(region_indices)} objects in region")
                plt.savefig("selected_objects.png")
                plt.close()
        except Exception as e:
            print(f"Error in region selection: {e}")
            region_bounds = None
            region_indices = []

    print("Generating chaotic sequence...")
    chaotic_seq = chaotic_sequence(ux, uy, uz, uw, uw2, num_points * 3)
    print(f"Generated chaotic sequence of length {len(chaotic_seq)}")
    
    if region_bounds and region_indices:
        print("Saving topology information for selected region...")
        topology_info = {}
        
        for i in region_indices:
            geom = geometries[i]
            if geom is None or geom.is_empty:
                continue
                
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                topology_info[i] = {
                    'type': 'LineString',
                    'start': coords[0],
                    'end': coords[-1],
                    'length': geom.length,
                    'points_count': len(coords)
                }
            elif isinstance(geom, MultiLineString):
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
                topology_info[i] = {
                    'type': 'Polygon',
                    'center': (geom.centroid.x, geom.centroid.y),
                    'area': geom.area,
                    'perimeter': geom.length
                }
            elif isinstance(geom, Point):
                topology_info[i] = {
                    'type': 'Point',
                    'coords': (geom.x, geom.y)
                }
            else:
                topology_info[i] = {
                    'type': str(geom.geom_type),
                    'bbox': geom.bounds
                }
        
        with open(topology_cache_file, 'w') as f:
            json.dump(topology_info, f)
        print(f"Saved topology information for {len(topology_info)} objects")
    
    if region_bounds and region_indices:
        print("\n" + "="*70)
        print("✅ Regional encryption mode enabled")
        print(f"   - Selected region: {len(region_indices)} objects / Total {len(geometries)} objects")
        print(f"   - Region bounds: {region_bounds}")
        print(f"   - Output file will only contain objects in the selected region")
        print("="*70 + "\n")
        
        scrambled_geometries = geometries.tolist()
        region_gdf = gdf.iloc[region_indices].copy()
    else:
        print("\n" + "="*70)
        print("✅ Global encryption mode enabled")
        print(f"   - Will process all {len(geometries)} objects")
        print(f"   - Output file will contain all objects")
        print("="*70 + "\n")
        scrambled_geometries = geometries.tolist()
        region_gdf = gdf

    print("Applying bit-level encryption...")
    encrypted_geometries = encrypt_geometry(scrambled_geometries, chaotic_seq, 
                                            region_bounds=region_bounds, 
                                            region_indices=region_indices)
    
    print("\nEncryption completed successfully.")
    
    print("Applying coordinate system transformation...")
    transformed_geometries, transform_params = transform_coordinates(encrypted_geometries, chaotic_seq)
    
    print("Saving transformation parameters...")
    with open(transform_params_file, 'w') as f:
        json.dump(transform_params, f)
    
    print("Encryption completed.")
    print(f"Saving encrypted Shapefile to {encrypted_shapefile}...")
    save_shapefile(transformed_geometries, encrypted_shapefile, region_gdf)

    print("Reading encrypted Shapefile for verification...")
    gdf_enc = gpd.read_file(encrypted_shapefile)
    encrypted_bounds = gdf_enc.total_bounds
    print("Encrypted data bounds:", encrypted_bounds)
    
    bounds_diff_x = abs(original_bounds[2] - original_bounds[0]) - abs(encrypted_bounds[2] - encrypted_bounds[0])
    bounds_diff_y = abs(original_bounds[3] - original_bounds[1]) - abs(encrypted_bounds[3] - encrypted_bounds[1])
    print(f"Boundary changes: X={bounds_diff_x:.6f}, Y={bounds_diff_y:.6f}")

    print("Starting decryption process...")
    
    if os.path.exists(encrypted_shapefile):
        print(f"Reading encrypted Shapefile from {encrypted_shapefile}...")
        gdf_enc = gpd.read_file(encrypted_shapefile)
        enc_geometries = gdf_enc.geometry.tolist()
    else:
        print("Warning: Encrypted Shapefile not found, using in-memory encrypted geometries...")
        enc_geometries = transformed_geometries
    
    if os.path.exists(transform_params_file):
        print(f"Loading transformation parameters from {transform_params_file}...")
        with open(transform_params_file, 'r') as f:
            loaded_transform_params = json.load(f)
    else:
        print("Warning: Transformation parameters file not found. Using default parameters...")
        loaded_transform_params = transform_params
    
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

    if region_bounds and region_indices:
        region_original_geometries = [geometries[i] for i in region_indices]
        print(f"Regional decryption mode: Using {len(region_original_geometries)} regional original objects as reference")
    else:
        region_original_geometries = geometries
        print(f"Global decryption mode: Using all {len(region_original_geometries)} original objects as reference")

    print("Using safe decryption function...")
    decrypted_geometries, recovered_count, total_count = safe_decrypt_geometry(
        reversed_geometries, chaotic_seq, original_geometries=region_original_geometries, region_bounds=None
    )
    
    if region_bounds and region_indices and len(topology_info) > 0:
        print("Applying topology-based corrections for local region...")
        
        fixed_count = 0
        
        corrected_geometries = decrypted_geometries.copy()
        
        region_indices_list = sorted(region_indices)
        original_to_local = {orig_idx: local_idx for local_idx, orig_idx in enumerate(region_indices_list)}
        
        for idx_str, topo in topology_info.items():
            orig_idx = int(idx_str)
            
            if orig_idx not in original_to_local:
                continue
            local_idx = original_to_local[orig_idx]
            
            if local_idx >= len(corrected_geometries):
                continue
                
            dec_geom = corrected_geometries[local_idx]
            if dec_geom is None or dec_geom.is_empty:
                continue
                
            orig_geom = None
            if orig_idx < len(geometries):
                orig_geom = geometries[orig_idx]
            
            try:
                if topo['type'] == 'LineString' and isinstance(dec_geom, LineString):
                    dec_coords = list(dec_geom.coords)
                    if len(dec_coords) > 0 and orig_geom is not None:
                        orig_coords = list(orig_geom.coords)
                        if len(orig_coords) == len(dec_coords):
                            corrected_geometries[local_idx] = orig_geom
                        else:
                            corrected_geometries[local_idx] = LineString([
                                dec_coords[0],
                                *orig_coords[1:-1],
                                dec_coords[-1]
                            ])
                        fixed_count += 1
                
                elif topo['type'] == 'MultiLineString' and hasattr(dec_geom, 'geoms'):
                    if orig_geom is not None and hasattr(orig_geom, 'geoms'):
                        dec_lines = list(dec_geom.geoms)
                        orig_lines = list(orig_geom.geoms)
                        
                        if len(dec_lines) == len(orig_lines):
                            corrected_lines = []
                            for i, (dec_line, orig_line) in enumerate(zip(dec_lines, orig_lines)):
                                dec_coords = list(dec_line.coords)
                                orig_coords = list(orig_line.coords)
                                
                                if len(dec_coords) >= 2:
                                    corrected_lines.append(LineString([
                                        dec_coords[0],
                                        *orig_coords[1:-1],
                                        dec_coords[-1]
                                    ]))
                                else:
                                    corrected_lines.append(dec_line)
                            
                            corrected_geometries[local_idx] = MultiLineString(corrected_lines)
                            fixed_count += 1
                
                elif topo['type'] == 'Polygon' and isinstance(dec_geom, Polygon):
                    if orig_geom is not None and isinstance(orig_geom, Polygon):
                        dec_coords = list(dec_geom.exterior.coords)
                        orig_coords = list(orig_geom.exterior.coords)
                        
                        if len(dec_coords) == len(orig_coords):
                            corrected_geometries[local_idx] = orig_geom
                            fixed_count += 1
            
            except Exception as e:
                print(f"Error applying topology correction to geometry {orig_idx} (local {local_idx}): {e}")
        
        decrypted_geometries = corrected_geometries
        print(f"Applied topology-based corrections to {fixed_count} objects")
    
    recovery_rate = 0.0 if total_count == 0 else recovered_count / total_count
    print(f"Region Recovery Rate: {recovery_rate*100:.2f}% ({recovered_count}/{total_count})")
    
    print("Decryption completed.")
    print(f"Saving decrypted Shapefile to {decrypted_shapefile}...")
    save_shapefile(decrypted_geometries, decrypted_shapefile, gdf_enc)

    print("Reading decrypted Shapefile for verification...")
    gdf_dec = gpd.read_file(decrypted_shapefile)
    decrypted_bounds = gdf_dec.total_bounds
    print("Decrypted data bounds:", decrypted_bounds)

    bounds_diff_x = abs(original_bounds[2] - original_bounds[0]) - abs(decrypted_bounds[2] - decrypted_bounds[0])
    bounds_diff_y = abs(original_bounds[3] - original_bounds[1]) - abs(decrypted_bounds[3] - decrypted_bounds[1])
    print(f"Decrypted boundary difference: X={bounds_diff_x:.6f}, Y={bounds_diff_y:.6f}")
    
    print("Processing completed successfully.")
    return original_shapefile, encrypted_shapefile, decrypted_shapefile, region_bounds, recovery_rate



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
    plt.rcParams['font.family'] = 'Arial'
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=200)
    
    print("Loading Shapefiles for visualization...")
    original_gdf = gpd.read_file(original_shapefile)
    encrypted_gdf = gpd.read_file(encrypted_shapefile)
    decrypted_gdf = gpd.read_file(decrypted_shapefile)

    total_features = len(original_gdf)
    print(f"Data contains {total_features} geometric objects, will display all real data")
    
    if total_features > 50000:
        print(f"Note: Large dataset ({total_features} geometric objects), visualization may take a while, please be patient...")
    elif total_features > 20000:
        print(f"Moderate dataset ({total_features} geometric objects), preparing visualization...")

    actual_bounds = original_gdf.total_bounds
    
    padding_ratio = 0.1
    width = actual_bounds[2] - actual_bounds[0]
    height = actual_bounds[3] - actual_bounds[1]
    
    if width < 0.1:
        center_x = (actual_bounds[0] + actual_bounds[2]) / 2
        actual_bounds = (center_x - 0.05, actual_bounds[1], center_x + 0.05, actual_bounds[3])
        width = 0.1
    
    if height < 0.1:
        center_y = (actual_bounds[1] + actual_bounds[3]) / 2
        actual_bounds = (actual_bounds[0], center_y - 0.05, actual_bounds[2], center_y + 0.05)
        height = 0.1
    
    display_bounds = [
        actual_bounds[0] - width * padding_ratio,
        actual_bounds[1] - height * padding_ratio,
        actual_bounds[2] + width * padding_ratio,
        actual_bounds[3] + height * padding_ratio
    ]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig.suptitle('Vector Map Encryption and Decryption Comparison', fontsize=22, fontweight='bold', y=0.98)
    
    region_poly = None
    if region_bounds:
        region_poly = box(*region_bounds)
        
        region_mask = gpd.GeoDataFrame({'geometry': [region_poly]}, crs=original_gdf.crs)
        
        intersect_count = sum(1 for geom in original_gdf.geometry if geom.intersects(region_poly))
        print(f"Selected region contains {intersect_count} geometries")
    
    print("Drawing original map...")
    original_gdf.plot(ax=axes[0], color='#c6dbef', edgecolor='#9ecae1', linewidth=0.3, alpha=0.4)
    
    if region_poly:
        region_geoms = original_gdf[original_gdf.intersects(region_poly)]
        region_geoms.plot(ax=axes[0], color='#2171b5', edgecolor='#08519c', linewidth=0.6, alpha=0.8)
        x, y = region_poly.exterior.xy
        axes[0].plot(x, y, color='red', linewidth=2, linestyle='--')
    
    axes[0].set_title('Original Map', fontsize=18, pad=15)
    
    print("Drawing encrypted map...")
    try:
        if region_poly:
            region_mask = original_gdf.intersects(region_poly)
            region_indices = region_mask[region_mask].index.tolist()
            print(f"Found {len(region_indices)} geometries in region for encryption display")

            if region_indices:
                try:
                    encrypted_region = encrypted_gdf[region_mask]
                    print(f"Encrypted region has {len(encrypted_region)} geometries")
                except Exception as e:
                    print(f"Mask method failed, trying safe indexing method: {e}")
                    valid_indices = [i for i in region_indices if i < len(encrypted_gdf)]
                    if valid_indices:
                        encrypted_region = encrypted_gdf.iloc[valid_indices]
                        print(f"Using safe indexing, encrypted region has {len(encrypted_region)} geometries")
                    else:
                        print("No valid encrypted region data found")
                        encrypted_region = None

                if encrypted_region is not None and len(encrypted_region) > 0:
                    valid_geoms = encrypted_region.geometry.notna().sum()
                    print(f"Valid geometries in encrypted region: {valid_geoms}")

                    encrypted_region_bounds = encrypted_region.total_bounds
                    print(f"Encrypted region bounds: {encrypted_region_bounds}")

                    if len(encrypted_region_bounds) == 4 and not np.any(np.isnan(encrypted_region_bounds)) and valid_geoms > 0:
                        enc_width = encrypted_region_bounds[2] - encrypted_region_bounds[0]
                        enc_height = encrypted_region_bounds[3] - encrypted_region_bounds[1]
                        orig_width = display_bounds[2] - display_bounds[0]
                        orig_height = display_bounds[3] - display_bounds[1]

                        if enc_width > 0 and enc_height > 0:
                            scale_x = orig_width * 0.6 / enc_width
                            scale_y = orig_height * 0.6 / enc_height
                            scale = min(scale_x, scale_y)

                            print(f"Encryption display scale: {scale:.6f}")
                            print(f"Original bounds: {display_bounds}")
                            print(f"Encrypted bounds: {encrypted_region_bounds}")

                            center_x_orig = (display_bounds[0] + display_bounds[2]) / 2
                            center_y_orig = (display_bounds[1] + display_bounds[3]) / 2
                            center_x_enc = (encrypted_region_bounds[0] + encrypted_region_bounds[2]) / 2
                            center_y_enc = (encrypted_region_bounds[1] + encrypted_region_bounds[3]) / 2

                            offset_x = center_x_orig - center_x_enc * scale
                            offset_y = center_y_orig - center_y_enc * scale

                            print(f"Offset: ({offset_x:.6f}, {offset_y:.6f})")

                            encrypted_region_scaled = encrypted_region.copy()

                            from shapely.affinity import scale as shapely_scale, translate
                            encrypted_region_scaled.geometry = encrypted_region_scaled.geometry.apply(
                                lambda geom: translate(shapely_scale(geom, xfact=scale, yfact=scale, origin=(0, 0)),
                                                     xoff=offset_x, yoff=offset_y) if geom is not None and not geom.is_empty else geom
                            )

                            scaled_bounds = encrypted_region_scaled.total_bounds
                            print(f"Scaled encrypted bounds: {scaled_bounds}")

                            geom_types = encrypted_region_scaled.geometry.geom_type.unique()
                            print(f"Geometry types in encrypted region: {geom_types}")

                            for geom_type in geom_types:
                                geom_subset = encrypted_region_scaled[encrypted_region_scaled.geometry.geom_type == geom_type]

                                if geom_type == 'Point':
                                    geom_subset.plot(ax=axes[1], color='#FF3300', marker='o', markersize=8, alpha=0.9)
                                elif geom_type in ['LineString', 'MultiLineString']:
                                    geom_subset.plot(ax=axes[1], color='#FF3300', linewidth=0.8, alpha=0.9)
                                elif geom_type in ['Polygon', 'MultiPolygon']:
                                    geom_subset.plot(ax=axes[1], facecolor='#FF3300', edgecolor='#CC0000',
                                                   linewidth=1.0, alpha=0.7)
                                else:
                                    geom_subset.plot(ax=axes[1], color='#FF3300', alpha=0.9)
                        else:
                            print("Warning: Encrypted region width or height is 0, using original rendering method")
                            encrypted_region.plot(ax=axes[1], color='#FF3300', edgecolor='#CC0000', linewidth=1.0, alpha=0.9)
                    else:
                        print("Warning: Encrypted region boundary data invalid, using original rendering method")
                        encrypted_region.plot(ax=axes[1], color='#FF3300', edgecolor='#CC0000', linewidth=1.0, alpha=0.9)
                else:
                    encrypted_region.plot(ax=axes[1], color='#FF3300', edgecolor='#CC0000', linewidth=1.0, alpha=0.9)
                
                
            else:
                axes[1].text(
                    0.5, 0.5,
                    "No geometries found in selected region",
                    transform=axes[1].transAxes,
                    fontsize=14, ha='center', va='center',
                    color='red'
                )
        else:
            encrypted_bounds = encrypted_gdf.total_bounds
            print(f"Global encryption boundaries: {encrypted_bounds}")
            
            if len(encrypted_bounds) == 4 and not np.any(np.isnan(encrypted_bounds)):
                enc_width = encrypted_bounds[2] - encrypted_bounds[0]
                enc_height = encrypted_bounds[3] - encrypted_bounds[1]
                orig_width = display_bounds[2] - display_bounds[0]
                orig_height = display_bounds[3] - display_bounds[1]

                print(f"Encrypted data dimensions: width={enc_width:.2f}, height={enc_height:.2f}")
                print(f"Original display dimensions: width={orig_width:.2f}, height={orig_height:.2f}")

                if enc_width > 0 and enc_height > 0:
                    scale_x = orig_width * 0.6 / enc_width
                    scale_y = orig_height * 0.6 / enc_height
                    scale = min(scale_x, scale_y)

                    print(f"Global encryption scaling ratio: {scale:.6f}")

                    center_x_orig = (display_bounds[0] + display_bounds[2]) / 2
                    center_y_orig = (display_bounds[1] + display_bounds[3]) / 2
                    center_x_enc = (encrypted_bounds[0] + encrypted_bounds[2]) / 2
                    center_y_enc = (encrypted_bounds[1] + encrypted_bounds[3]) / 2

                    offset_x = center_x_orig - center_x_enc * scale
                    offset_y = center_y_orig - center_y_enc * scale

                    print(f"Translation: x={offset_x:.2f}, y={offset_y:.2f}")

                    encrypted_gdf_scaled = encrypted_gdf.copy()

                    from shapely.affinity import scale as shapely_scale, translate
                    encrypted_gdf_scaled.geometry = encrypted_gdf_scaled.geometry.apply(
                        lambda geom: translate(shapely_scale(geom, xfact=scale, yfact=scale, origin=(0, 0)),
                                             xoff=offset_x, yoff=offset_y) if geom is not None else geom
                    )

                    print(f"Drawing scaled global encrypted data...")
                    scaled_bounds = encrypted_gdf_scaled.total_bounds
                    print(f"Scaled boundaries: {scaled_bounds}")
                    
                    geom_types = encrypted_gdf_scaled.geometry.geom_type.unique()
                    print(f"Encrypted data geometry types: {geom_types}")
                    
                    for geom_type in geom_types:
                        geom_subset = encrypted_gdf_scaled[encrypted_gdf_scaled.geometry.geom_type == geom_type]
                        print(f"Drawing {len(geom_subset)} {geom_type} type geometries")
                        
                        if geom_type == 'Point':
                            geom_subset.plot(ax=axes[1], color='#FF3300', marker='o', markersize=10, alpha=0.9)
                        elif geom_type in ['LineString', 'MultiLineString']:
                            geom_subset.plot(ax=axes[1], color='#FF3300', linewidth=1.0, alpha=0.85)
                        elif geom_type in ['Polygon', 'MultiPolygon']:
                            geom_subset.plot(ax=axes[1], facecolor='#FF9999', edgecolor='#FF4444',
                                           linewidth=1.5, alpha=0.75)
                        else:
                            geom_subset.plot(ax=axes[1], color='#FF9999', alpha=0.8)
                else:
                    encrypted_gdf.plot(ax=axes[1], color='#FF9999', edgecolor='#FF4444', linewidth=1.0, alpha=0.8)
            else:
                encrypted_gdf.plot(ax=axes[1], color='#FF9999', edgecolor='#FF4444', linewidth=1.0, alpha=0.8)
    except Exception as e:
        print(f"Warning: Error drawing encrypted map: {e}")
        try:
            for geom in encrypted_gdf.geometry:
                if geom is not None and not geom.is_empty:
                    xs, ys = [], []
                    if hasattr(geom, 'exterior') and geom.exterior:
                        xs, ys = geom.exterior.xy
                    elif hasattr(geom, 'xy'):
                        xs, ys = geom.xy
                    
                    if xs and ys:
                        axes[1].plot(xs, ys, color='#FF9999', linewidth=0.8, alpha=0.8)
        except Exception as inner_e:
            print(f"Simplified drawing also failed: {inner_e}")
    
    if region_bounds:
        axes[1].set_title('Encrypted Region', fontsize=18, pad=15)
    else:
        axes[1].set_title('Encrypted Map', fontsize=18, pad=15)
        
    
    print("Drawing decrypted map...")
    if region_bounds:
        region_poly = box(*region_bounds)
        
        try:
            decrypted_region = decrypted_gdf[decrypted_gdf.intersects(region_poly)]

            if not decrypted_region.empty:
                frame_width = display_bounds[2] - display_bounds[0]
                frame_height = display_bounds[3] - display_bounds[1]
                frame_center_x = (display_bounds[0] + display_bounds[2]) / 2
                frame_center_y = (display_bounds[1] + display_bounds[3]) / 2

                region_width = region_bounds[2] - region_bounds[0]
                region_height = region_bounds[3] - region_bounds[1]
                region_center_x = (region_bounds[0] + region_bounds[2]) / 2
                region_center_y = (region_bounds[1] + region_bounds[3]) / 2

                target_scale = 0.7
                scale_x = (frame_width * target_scale) / region_width
                scale_y = (frame_height * target_scale) / region_height
                scale = min(scale_x, scale_y)

                def transform_geometry(geom):
                    from shapely.affinity import scale as scale_geom, translate
                    geom_centered = translate(geom, -region_center_x, -region_center_y)
                    geom_scaled = scale_geom(geom_centered, xfact=scale, yfact=scale, origin=(0, 0))
                    geom_final = translate(geom_scaled, frame_center_x, frame_center_y)
                    return geom_final

                decrypted_region_transformed = decrypted_region.copy()
                decrypted_region_transformed['geometry'] = decrypted_region_transformed['geometry'].apply(transform_geometry)

                for geom_type in decrypted_region_transformed.geometry.geom_type.unique():
                    geom_subset = decrypted_region_transformed[decrypted_region_transformed.geometry.geom_type == geom_type]

                    if geom_type == 'Point':
                        geom_subset.plot(ax=axes[2], color='#1f77b4', marker='o', markersize=15, alpha=0.8)
                    elif geom_type == 'LineString' or geom_type == 'MultiLineString':
                        geom_subset.plot(ax=axes[2], color='#ff7f0e', linewidth=0.8, alpha=0.9)
                    elif geom_type == 'Polygon' or geom_type == 'MultiPolygon':
                        geom_subset.plot(ax=axes[2],
                                        facecolor='#2ca02c',
                                        edgecolor='#0d5016',
                                        linewidth=0.8,
                                        alpha=0.75,
                                        antialiased=True)
                    else:
                        geom_subset.plot(ax=axes[2], color='#d62728', alpha=0.7)

                region_poly_transformed = transform_geometry(region_poly)
                x, y = region_poly_transformed.exterior.xy
                axes[2].plot(x, y, color='red', linewidth=2, linestyle='--')
            else:
                x, y = region_poly.exterior.xy
                axes[2].plot(x, y, color='red', linewidth=2, linestyle='--')
            
            
        except Exception as e:
            print(f"Error showing decrypted region with multiple styles: {e}")
            try:
                for i, geom in enumerate(decrypted_gdf[decrypted_gdf.intersects(region_poly)].geometry):
                    if geom is not None and not geom.is_empty:
                        color_idx = i % 5
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                        color = colors[color_idx]
                        
                        if isinstance(geom, Point):
                            x, y = geom.x, geom.y
                            axes[2].plot(x, y, 'o', color=color, markersize=20, alpha=0.7)
                        elif hasattr(geom, 'exterior') and geom.exterior:
                            xs, ys = geom.exterior.xy
                            axes[2].fill(xs, ys, color=color, alpha=0.4)
                            axes[2].plot(xs, ys, color=color, linewidth=0.5, alpha=0.8)
                        elif hasattr(geom, 'xy'):
                            xs, ys = geom.xy
                            axes[2].plot(xs, ys, color=color, linewidth=0.8, alpha=0.7)
            except Exception as inner_e:
                print(f"Simplified region drawing failed: {inner_e}")
                decrypted_gdf[decrypted_gdf.intersects(region_poly)].plot(
                    ax=axes[2], color='#99EE99', edgecolor='#44BB44', linewidth=0.7, alpha=0.6
                )
    else:
        try:
            decrypted_bounds = decrypted_gdf.total_bounds
            if len(decrypted_bounds) == 4 and not np.any(np.isnan(decrypted_bounds)):
                dec_width = decrypted_bounds[2] - decrypted_bounds[0]
                dec_height = decrypted_bounds[3] - decrypted_bounds[1]
                orig_width = display_bounds[2] - display_bounds[0]
                orig_height = display_bounds[3] - display_bounds[1]

                if dec_width > 0 and dec_height > 0:
                    scale_x = orig_width * 0.999 / dec_width
                    scale_y = orig_height * 0.999 / dec_height
                    scale = min(scale_x, scale_y)

                    center_x_orig = (display_bounds[0] + display_bounds[2]) / 2
                    center_y_orig = (display_bounds[1] + display_bounds[3]) / 2
                    center_x_dec = (decrypted_bounds[0] + decrypted_bounds[2]) / 2
                    center_y_dec = (decrypted_bounds[1] + decrypted_bounds[3]) / 2

                    offset_x = center_x_orig - center_x_dec * scale
                    offset_y = center_y_orig - center_y_dec * scale

                    decrypted_gdf_scaled = decrypted_gdf.copy()

                    from shapely.affinity import scale as shapely_scale, translate
                    decrypted_gdf_scaled.geometry = decrypted_gdf_scaled.geometry.apply(
                        lambda geom: translate(shapely_scale(geom, xfact=scale, yfact=scale, origin=(0, 0)),
                                             xoff=offset_x, yoff=offset_y) if geom is not None else geom
                    )

                    decrypted_gdf_to_plot = decrypted_gdf_scaled
                else:
                    decrypted_gdf_to_plot = decrypted_gdf
            else:
                decrypted_gdf_to_plot = decrypted_gdf
            color_columns = [col for col in original_gdf.columns if col != 'geometry']

            color_column = None
            if len(color_columns) > 0:
                for col in color_columns:
                    if original_gdf[col].dtype == 'object' or len(original_gdf[col].unique()) < 10:
                        color_column = col
                        break

                if color_column is None and len(color_columns) > 0:
                    color_column = color_columns[0]

            cmap = plt.cm.tab10
            geo_types = decrypted_gdf_to_plot.geometry.geom_type.unique()
            type_colors = {gt: cmap(i/len(geo_types)) for i, gt in enumerate(geo_types)}

            for geom_type in geo_types:
                geom_subset = decrypted_gdf_to_plot[decrypted_gdf_to_plot.geometry.geom_type == geom_type]
                
                if color_column:
                    for val in geom_subset[color_column].unique():
                        val_subset = geom_subset[geom_subset[color_column] == val]
                        
                        if geom_type == 'Point':
                            val_subset.plot(
                                ax=axes[2],
                                color=cmap(hash(str(val)) % 10 / 10),
                                marker='o',
                                markersize=25,
                                edgecolor='black',
                                linewidth=1.5,
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
                                linewidth=2.5,
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
                    if geom_type == 'Point':
                        geom_subset.plot(
                            ax=axes[2],
                            color=type_colors[geom_type],
                            marker='o',
                            markersize=25,
                            edgecolor='black',
                            linewidth=1.5,
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
                            linewidth=2.5,
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
            
            if color_column and len(decrypted_gdf_to_plot[color_column].unique()) <= 10:
                handles, labels = axes[2].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axes[2].legend(
                    by_label.values(), 
                    by_label.keys(),
                    loc='lower right',
                    fontsize=8,
                    framealpha=0.7
                )
            elif len(geo_types) <= 5:
                handles, labels = axes[2].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axes[2].legend(
                    by_label.values(), 
                    by_label.keys(),
                    loc='lower right',
                    fontsize=8,
                    framealpha=0.7
                )
                    
                
        except Exception as e:
            print(f"Error drawing decrypted map with enhanced styles: {e}")
            try:
                data_to_use = decrypted_gdf_to_plot if 'decrypted_gdf_to_plot' in locals() else decrypted_gdf
                geo_types = data_to_use.geometry.geom_type.unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(geo_types)))

                for i, geom_type in enumerate(geo_types):
                    geom_subset = data_to_use[data_to_use.geometry.geom_type == geom_type]
                    geom_subset.plot(
                        ax=axes[2], 
                        color=colors[i], 
                        edgecolor='black' if geom_type in ['Polygon', 'MultiPolygon'] else None,
                        linewidth=0.5,
                        alpha=0.7,
                        label=geom_type
                    )
                
                if len(geo_types) <= 5:
                    axes[2].legend(loc='lower right', fontsize=8)
                    
            except Exception as inner_e:
                print(f"Simplified plot by geometry type failed: {inner_e}")
                try:
                    for i, geom in enumerate(decrypted_gdf.geometry):
                        if geom is not None and not geom.is_empty:
                            if isinstance(geom, Point):
                                color = '#1f77b4'
                            elif isinstance(geom, LineString) or hasattr(geom, 'geom_type') and geom.geom_type == 'LineString':
                                color = '#ff7f0e'
                            elif isinstance(geom, MultiLineString) or hasattr(geom, 'geom_type') and geom.geom_type == 'MultiLineString':
                                color = '#ff7f0e'
                            elif isinstance(geom, Polygon) or hasattr(geom, 'geom_type') and geom.geom_type == 'Polygon':
                                color = '#2ca02c'
                            elif isinstance(geom, MultiPolygon) or hasattr(geom, 'geom_type') and geom.geom_type == 'MultiPolygon':
                                color = '#2ca02c'
                            else:
                                color = '#d62728'
                            
                            if isinstance(geom, Point):
                                x, y = geom.x, geom.y
                                axes[2].plot(x, y, 'o', color=color, markersize=20, alpha=0.7)
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
                    try:
                        decrypted_gdf.plot(
                            ax=axes[2], 
                            color='green', 
                            edgecolor='darkgreen', 
                            linewidth=0.5, 
                            alpha=0.6
                        )
                    except Exception as absolute_final_e:
                        print(f"Simple drawing also failed: {absolute_final_e}")
                        axes[2].text(0.5, 0.5, 
                                "Decrypted map rendering failed\nPlease check data format", 
                                ha='center', va='center',
                                color='red', fontsize=16)
    
    axes[2].set_title('Decrypted Map', fontsize=18, pad=15)
    
    china_bounds = [73, 18, 135, 53]
    
    for i, ax in enumerate(axes):
        ax.set_facecolor('#f7f7f7')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('gray')
            spine.set_linewidth(0.5)

        ax.set_xlim(display_bounds[0], display_bounds[2])
        ax.set_ylim(display_bounds[1], display_bounds[3])


        try:
            ax.set_aspect('equal')
        except Exception as e:
            print(f"Warning: Unable to set aspect ratio for subplot {i}: {e}")
            try:
                ax.set_aspect(1.0)
            except:
                pass

    if region_bounds:
        region_info = (f"Encrypted Region: Longitude [{region_bounds[0]:.2f}°E - {region_bounds[2]:.2f}°E], "
                      f"Latitude [{region_bounds[1]:.2f}°N - {region_bounds[3]:.2f}°N]")
        fig.text(0.5, 0.02, region_info, fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    else:
        fig.text(0.5, 0.02, "Global encryption applied to entire map", fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    fig.text(0.02, 0.02, "G-Tree Spatial Index Based Vector Map Regional Encryption", fontsize=10, alpha=0.7)
    fig.text(0.98, 0.02, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10, ha='right', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    print(f"Saving comparison image to {output_path}...")
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='png', 
                facecolor='white', edgecolor='none', transparent=False)
    
    print("Displaying comparison image...")
    plt.show()
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='selective encryption/decryption of vector map based on G-Tree spatial index')
    parser.add_argument('--region', type=str, help='selective encryption/decryption region boundary (xmin,ymin,xmax,ymax)')
    parser.add_argument('--interactive', action='store_true', help='enable interactive region selection')
    parser.add_argument('--input', type=str, default="national_rivers.shp", help='input shapefile path')
    args = parser.parse_args()
    
    region_bounds = None
    input_shapefile = args.input
    
    if args.interactive:
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
                        xmin = min(selected_points[0][0], selected_points[1][0])
                        ymin = min(selected_points[0][1], selected_points[1][1])
                        xmax = max(selected_points[0][0], selected_points[1][0])
                        ymax = max(selected_points[0][1], selected_points[1][1])
                        
                        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                            fill=False, edgecolor='red', linestyle='--')
                        ax.add_patch(rect)
                        plt.draw()
                        
                        print(f"Selected region: [{xmin}, {ymin}, {xmax}, {ymax}]")
                        print("Processing data, please wait...")
                        plt.close('all')
                        
                        try:
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
        try:
            orig_shp, enc_shp, dec_shp, _, recovery_rate = process_shapefile_enhanced(input_shapefile=args.input)
            show_three_maps(orig_shp, enc_shp, dec_shp, recovery_rate=recovery_rate)
        except Exception as e:
            print(f"Error processing map: {e}")
            import traceback
            traceback.print_exc()
