"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

visualization_metrics.py

Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

'输入的数据形式 一个列表 3662个元素 每个是(24, 6)的矩阵 3662 (24, 6)'
'左边是经过TimeDataset处理后的真实数据 右边是生成数据 形式一样'
'对两个数据进行了一样的操作'

def visualization(ori_data, generated_data, analysis):
    """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])       # 取1000
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])   # 变成了一行
        else:
            prep_data = np.concatenate(
                (prep_data, np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len]))
            )
            prep_data_hat = np.concatenate(
                (prep_data_hat, np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len]))
            )

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    
    'pca是真假数据分开来做 这里也是取平均值后做pca 不是直接做 直接做可能更直观'
    if analysis == "pca":
        
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)                      # 可以直接对两个数据分别fit_transform 但结果类似
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)
        
        
#         pca = PCA(n_components=2)                 # 分别用fit_transform处理 但结果类似 
#         pca_results = pca.fit_transform(prep_data)           
#         pca = PCA(n_components=2)
#         pca_hat_results = pca.fit_transform(prep_data_hat)
        

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(
            pca_results[:, 0],
            pca_results[:, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            pca_hat_results[:, 0],
            pca_hat_results[:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )

        ax.legend()
        plt.title("PCA plot")
        plt.xlabel("x-pca")
        plt.ylabel("y_pca")
        plt.savefig('pca',dpi=500)
        plt.show()

        
        'tsne是真假数据一起做 应该是为了防止随机种子不一样导致图像差别太大'
        '可以设置固定的随机种子 但是由于原数据处理时也是有随机的过程 所以这里加不加随机种子都一样 原数据每次产生的图像也不一样'
        '而且每次模型生成的数据 也是有随机噪声生成的 应该和原数据不会一一对应'
        '-----能够每次都让真假数据比较重合的原因应该是拼接到一起处理的操作以及前面求平均值的操作 还有可能要联系到tsne的原理-----'
    elif analysis == "tsne":

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=20,n_iter=500)   # 固定随机种子
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(
            tsne_results[:anal_sample_no, 0],
            tsne_results[:anal_sample_no, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            tsne_results[anal_sample_no:, 0],
            tsne_results[anal_sample_no:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )

        ax.legend()

        plt.title("t-SNE plot")
        plt.xlabel("x-tsne")
        plt.ylabel("y_tsne")
        plt.savefig('tsne',dpi=500)
        plt.show()
        
        
        '把tsne分开来做 这种方法图像几乎不重合 不如原版'
        '其实用类似pca的操作 不用拼接 不用取平均 这样可能更直观 论文里这么做应该是为了体现某些特性'
#     elif analysis == "tsne":

#         tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=20,n_iter=500)   # 固定随机种子       
#         tsne_results = tsne.fit_transform(prep_data)
           
#         tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=20,n_iter=500)   # 固定随机种子
#         tsne_results1 = tsne.fit_transform(prep_data_hat)

#         f, ax = plt.subplots(1)

#         plt.scatter(
#             tsne_results[:, 0],
#             tsne_results[:, 1],
#             c=colors[:anal_sample_no],
#             alpha=0.2,
#             label="Original",
#         )
#         plt.scatter(
#             tsne_results1[:, 0],
#             tsne_results1[:, 1],
#             c=colors[anal_sample_no:],
#             alpha=0.2,
#             label="Synthetic",
#         )

#         ax.legend()

#         plt.title("t-SNE plot")
#         plt.xlabel("x-tsne")
#         plt.ylabel("y_tsne")
#         plt.savefig('tsne',dpi=500)
#         plt.show()


        
     
    