import os
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    rounds = []
    times = []
    train_losses = []
    ptest_accs = []
    g_accs = []
    
    round_pattern = re.compile(r'Train Round: (\d+)')
    time_pattern = re.compile(r'Time: ([\d-]+ [\d:]+)')
    train_loss_pattern = re.compile(r'train_loss: ([\d.]+)')
    ptest_acc_pattern = re.compile(r'ptest_acc:([\d.]+)')
    g_acc_pattern = re.compile(r"atest_acc:([\d.]+)")

    for line in lines:
        round_match = round_pattern.search(line)
        time_match = time_pattern.search(line)
        train_loss_match = train_loss_pattern.search(line)
        ptest_acc_match = ptest_acc_pattern.search(line)
        g_acc_match = g_acc_pattern.search(line)
        
        if round_match and time_match and train_loss_match and ptest_acc_match:
            rounds.append(int(round_match.group(1)))
            times.append(time_match.group(1))
            train_losses.append(float(train_loss_match.group(1)))
            ptest_accs.append(float(ptest_acc_match.group(1)))
            g_accs.append(float(g_acc_match.group(1)))
    
    return rounds, times, train_losses, ptest_accs,g_accs

def smooth_data(data, window_size=5):
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return smoothed_data

def plot_metrics(all_rounds, all_train_losses, all_ptest_accs, labels):
    plt.figure(figsize=(14, 7))
    
    for i in range(len(all_rounds)):
        rounds = all_rounds[i]
        train_losses = all_train_losses[i]
        ptest_accs = all_ptest_accs[i]
        
        smooth_train_losses = smooth_data(train_losses)
        smooth_ptest_accs = smooth_data(ptest_accs)
        smooth_rounds = rounds[:len(smooth_train_losses)]  # Adjust rounds to match smoothed data length
        
        # 绘制train_loss和round的变化
        plt.subplot(1, 2, 1)
        plt.plot(smooth_rounds, smooth_train_losses, marker='o', linestyle='-', label=labels[i])
        plt.xlabel('Round')
        plt.ylabel('Train Loss')
        
        # 绘制ptest_acc和round的变化
        plt.subplot(1, 2, 2)
        plt.plot(smooth_rounds, smooth_ptest_accs, marker='o', linestyle='-', label=labels[i])
        plt.xlabel('Round')
        plt.ylabel('PTest Accuracy')
    
    plt.subplot(1, 2, 1)
    plt.title('Train Loss vs. Round')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title('PTest Accuracy vs. Round')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('result_analyze/fedclp_clip.png')
    plt.show()
  
def plot_metrics_one(all_rounds, all_ptest_accs, labels):
    # 设置Nature风格的参数
    plt.rcParams.update({
        'font.family': 'sans-serif',  # 字体为无衬线字体，期刊常用 Arial
        'font.sans-serif': ['Arial'],  # 使用 Arial 字体
        'axes.linewidth': 1.0,  # 边框线宽
        'axes.spines.top': False,  # 去掉顶部边框
        'axes.spines.right': False,  # 去掉右侧边框
        'xtick.major.size': 5,  # x轴主刻度线长度
        'ytick.major.size': 5,  # y轴主刻度线长度
        'xtick.minor.size': 3,  # x轴次刻度线长度
        'ytick.minor.size': 3,  # y轴次刻度线长度
        'xtick.direction': 'in',  # 刻度线向内
        'ytick.direction': 'in',  # 刻度线向内
        'legend.frameon': False,  # 图例不带边框
        'legend.fontsize': 10,  # 图例字体大小
        'axes.labelsize': 12,  # 坐标轴标签字体大小
        'axes.titlesize': 14,  # 标题字体大小
        'lines.markersize': 4,  # 点的大小
        'lines.linewidth': 1.5,  # 线宽
        'figure.dpi': 300,  # 高分辨率
    })
    
    plt.figure(figsize=(7, 5))  # 设置图形大小
    
    for i in range(len(all_rounds)):
        rounds = all_rounds[i]
        ptest_accs = all_ptest_accs[i]
        
        smooth_ptest_accs = smooth_data(ptest_accs)
        smooth_rounds = rounds[:len(smooth_ptest_accs)]  # Adjust rounds to match smoothed data length
        
        # 绘制 PTest Accuracy 和 Round 的变化
        plt.plot(smooth_rounds, smooth_ptest_accs, marker='o', linestyle='-', label=labels[i])
    
    plt.xlabel('Round')
    plt.ylabel('PTest Accuracy')
    plt.title('PTest Accuracy vs. Round')
    plt.legend(loc='best')  # 图例放置在最佳位置
    
    plt.tight_layout()
    plt.savefig('result_analyze/fedclp_clip_accuracy_nature_style.png', dpi=300)  # 保存为高分辨率图像
    plt.show()

def compute_last_10_avg_and_std(all_ptest_accs, labels):
    for i in range(len(all_ptest_accs)):
        ptest_accs = all_ptest_accs[i]
        last_10_values = ptest_accs[-10:]
        avg_last_10 = np.mean(last_10_values)
        std_last_10 = np.std(last_10_values)
        print(f'{labels[i]}: Average of last 10 PTest Accuracies = {avg_last_10:.4f}, Std Dev = {std_last_10:.4f}')

# def read_and_plot_all_logs(dir_path):
#     all_rounds = []
#     all_train_losses = []
#     all_ptest_accs = []
#     labels = []
    
#     for subdir, _, files in os.walk(dir_path):
#         if 'fl_log.txt' in files:
#             file_path = os.path.join(subdir, 'fl_log.txt')
#             rounds, times, train_losses, ptest_accs = extract_data_from_txt(file_path)
#             all_rounds.append(rounds)
#             all_train_losses.append(train_losses)
#             all_ptest_accs.append(ptest_accs)
#             labels.append(os.path.basename(subdir))
    
#     plot_metrics(all_rounds, all_train_losses, all_ptest_accs, labels)
#     compute_last_10_avg_and_std(all_ptest_accs, labels)

def read_and_plot_logs_list(file_paths,labels):
    all_rounds = []
    all_train_losses = []
    all_ptest_accs = []
    all_g_accs = []
    labels = labels
    
    for file_path in file_paths:
        rounds, times, train_losses, ptest_accs ,g_accs = extract_data_from_txt(file_path)
        all_rounds.append(rounds)
        all_train_losses.append(train_losses)
        all_ptest_accs.append(ptest_accs)
        all_g_accs.append(g_accs)
        # labels.append(os.path.basename(os.path.dirname(file_path)))  # 使用文件所在目录作为标签
    
    # plot_metrics(all_rounds, all_train_losses, all_ptest_accs, labels)
    plot_metrics_one(all_rounds, all_ptest_accs, labels)
    # plot_metrics(all_rounds, all_train_losses, all_g_accs, labels)

    compute_last_10_avg_and_std(all_g_accs, labels)


file_paths = [
    'A:\\北航\\论文\\fedceaClp\\fedcea\\logs\\fedclp_logs\\FL_16_cifar10_alex_0.1_200\\20240906_143742.txt',
    'A:\\北航\\论文\\fedceaClp\\fedcea\\logs\\fedclp_CLIP_logs\\FL_16_cifar10_alex_0.1_200\\20240907_081138.txt',
    'A:\\北航\\论文\\fedceaClp\\fedcea\\logs\\fedclp_CLIP_logs\\FL_16_cifar10_alex_0.1_200_beta0.5\\20240908_183505.txt'
    ,'A:\\北航\\论文\\fedceaClp\\fedcea\\logs\\fedclp_CLIP_logs\\FL_16_cifar10_alex_alpha0.1_200_beta0.1\\20240910_093616.txt',
    'A:\\北航\\论文\\fedceaClp\\fedcea\\logs\\fedclp_CLIP_logs\\FL_16_cifar10_alex_alpha0.1_200_beta0.3\\20240909_174350.txt'
]
labels = ['fedclp_logs','fedclp_CLIP_alpha0.1_beta1','fedclp_CLIP_alpha0.1_beta0.5','fedclp_CLIP_alpha0.1_beta0.1',
          'fedclp_CLIP_alpha0.1_beta0.3'] 

read_and_plot_logs_list(file_paths,labels)

