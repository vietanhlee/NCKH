import numpy as np
from scipy.stats import friedmanchisquare, chi2

# Dữ liệu trích xuất chính xác 100% từ nhật ký thực nghiệm (Log)
data = {
    0.00: {
        'STGCN_Baseline': [3.2433, 3.2301, 3.2479, 3.2466, 3.2226],
        'Graph_WaveNet':  [3.2465, 3.2431, 3.2415, 3.2449, 3.2540],
        'ASTGCN':         [3.4041, 3.3838, 3.3891, 3.3752, 3.3838],
        'STAEformer':     [3.2584, 3.2759, 3.2358, 3.2575, 3.2876],
        'MegaCRN':        [3.3017, 3.3047, 3.3188, 3.3247, 3.3081],
        'DSTAGNN':        [3.3192, 3.2971, 3.3189, 3.2957, 3.3332],
        'iTransformer':   [3.3293, 3.3234, 3.3329, 3.3500, 3.3542],
        'TA-STGCN':       [3.2096, 3.2163, 3.2229, 3.2004, 3.1979],
    },
    3.49: {
        'STGCN_Baseline': [5.1001, 5.3055, 5.4367, 5.3950, 5.3092],
        'Graph_WaveNet':  [5.7386, 5.9036, 5.5860, 5.8043, 5.7026],
        'ASTGCN':         [5.7493, 5.7456, 5.5330, 5.8200, 6.0067],
        'STAEformer':     [5.1519, 5.5112, 5.5591, 4.7394, 5.1575],
        'MegaCRN':        [5.8983, 5.6801, 5.9557, 6.0467, 5.9497],
        'DSTAGNN':        [6.2038, 6.3122, 6.2180, 6.1617, 6.1871],
        'iTransformer':   [5.8193, 5.8357, 5.6498, 5.6310, 5.5991],
        'TA-STGCN':       [5.2004, 5.0787, 5.3778, 5.2448, 5.2986],
    },
    4.64: {
        'STGCN_Baseline': [6.3882, 6.7194, 6.8529, 6.7078, 6.6388],
        'Graph_WaveNet':  [7.1815, 7.4188, 7.0368, 7.3409, 7.2109],
        'ASTGCN':         [6.9868, 6.8402, 6.6931, 7.0700, 7.3880],
        'STAEformer':     [6.4344, 6.9047, 7.0045, 5.4773, 6.3093],
        'MegaCRN':        [7.3878, 7.0476, 7.2972, 7.4376, 7.4018],
        'DSTAGNN':        [7.3481, 8.1393, 7.9297, 7.8458, 7.8753],
        'iTransformer':   [6.9973, 7.2110, 6.8510, 6.7621, 6.7207],
        'TA-STGCN':       [6.2252, 5.9935, 6.5509, 6.3372, 6.4531],
    },
    10.25: {
        'STGCN_Baseline': [11.6634, 12.5256, 12.4261, 11.8489, 11.9027],
        'Graph_WaveNet':  [12.8755, 13.7175, 12.9030, 13.4656, 13.4000],
        'ASTGCN':         [12.0256, 10.8989, 12.0713, 11.8471, 12.5521],
        'STAEformer':     [11.4499, 10.8888, 12.1132, 8.0289, 10.4771],
        'MegaCRN':        [13.8943, 13.1509, 12.2633, 13.0654, 13.4037],
        'DSTAGNN':        [11.4573, 12.1765, 13.2459, 13.2337, 12.7632],
        'iTransformer':   [10.1395, 11.7177, 10.2602, 9.8591, 9.6504],
        'TA-STGCN':       [8.6827, 8.6256, 10.7696, 10.1305, 9.9823],
    },
    13.95: {
        'STGCN_Baseline': [12.7428, 13.5929, 13.5630, 12.8574, 13.1059],
        'Graph_WaveNet':  [14.5385, 15.7311, 14.6851, 15.3667, 15.2016],
        'ASTGCN':         [13.7410, 12.0758, 14.1353, 13.3121, 14.0781],
        'STAEformer':     [11.6577, 10.4711, 12.3306, 8.7441, 11.0236],
        'MegaCRN':        [16.0626, 15.5278, 13.4749, 15.1789, 15.3088],
        'DSTAGNN':        [14.7257, 13.8280, 13.0387, 13.0243, 12.2358],
        'iTransformer':   [10.6928, 12.6151, 10.7187, 10.2941, 9.8904],
        'TA-STGCN':       [8.7904, 9.0667, 11.7594, 10.9255, 10.2238],
    }
}

models = list(data[0.00].keys())

print("="*80)
print("CHI TIẾT KIỂM ĐỊNH FRIEDMAN CHO TỪNG LEVEL NHIỄU:")
print("="*80)

for lvl, mdict in data.items():
    arrays = [mdict[m] for m in models]
    stat, p = friedmanchisquare(*arrays)
    print(f"Noise Level {lvl:>5.2f} | Chi-Square = {stat:8.4f} | p-value = {p:.4e}")

# Omnibus L1 - L4 (Tập dữ liệu nhiễu thực tế)
arrays_l1_l4 = []
for m in models:
    combined = []
    for lvl in [3.49, 4.64, 10.25, 13.95]:
        combined.extend(data[lvl][m])
    arrays_l1_l4.append(combined)

stat_noise, p_noise = friedmanchisquare(*arrays_l1_l4)
print("-" * 80)
print(f"OMNIBUS NOISE ONLY (L1-L4, N=20 blocks) | Chi-Square = {stat_noise:8.4f} | p-value = {p_noise:.4e}")

# Omnibus L0 - L4 (Toàn bộ 5 levels)
arrays_all = []
for m in models:
    combined = []
    for lvl in [0.00, 3.49, 4.64, 10.25, 13.95]:
        combined.extend(data[lvl][m])
    arrays_all.append(combined)

stat_all, p_all = friedmanchisquare(*arrays_all)
print(f"OMNIBUS ALL LEVELS (L0-L4, N=25 blocks) | Chi-Square = {stat_all:8.4f} | p-value = {p_all:.4e}")
print("="*80)
