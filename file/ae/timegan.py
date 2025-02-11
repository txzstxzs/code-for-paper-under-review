from itertools import chain
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
# from time_dataset import TimeDataset, batch_generator
# from visualization_metrics import visualization
import argparse

'5个网络结构都一样 只是输入输出维度 RNN层数不一样'
class Net(nn.Module):
    def __init__( self, input_size, hidden_size, output_size, num_layers, rnn=nn.GRU, activation_fn=torch.sigmoid, ):
        super().__init__()
        self.rnn = rnn(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


def train(batch_size, max_steps, dataset, device, embedder, generator, supervisor, recovery, discriminator, gamma,):
    def _loss_e_t0(x_tilde, x):
        return F.mse_loss(x_tilde, x)

    def _loss_e_0(loss_e_t0):
        return torch.sqrt(loss_e_t0) * 10

    def _loss_e(loss_e_0, loss_s):
        return loss_e_0 + 0.1 * loss_s

    def _loss_s(h_hat_supervise, h):
        return F.mse_loss(h[:, 1:, :], h_hat_supervise[:, 1:, :])

    def _loss_g_u(y_fake):
        return F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))

    def _loss_g_u_e(y_fake_e):
        return F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))

    def _loss_g_v(x_hat, x):
        loss_g_v1 = torch.mean(
            torch.abs(torch.sqrt(torch.var(x_hat, 0) + 1e-6) - torch.sqrt(torch.var(x, 0) + 1e-6)) )
        loss_g_v2 = torch.mean(torch.abs(torch.mean(x_hat, 0) - torch.mean(x, 0)))
        return loss_g_v1 + loss_g_v2

    def _loss_g(loss_g_u, loss_g_u_e, loss_s, loss_g_v):
        return loss_g_u + gamma * loss_g_u_e + 100 * torch.sqrt(loss_s) + 100 * loss_g_v

    def _loss_d(y_real, y_fake, y_fake_e):
        loss_d_real = F.binary_cross_entropy_with_logits(y_real, torch.ones_like(y_real))
        loss_d_fake = F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))
        loss_d_fake_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.zeros_like(y_fake_e))
        return loss_d_real + loss_d_fake + gamma * loss_d_fake_e

    optimizer_er = optim.Adam(chain(embedder.parameters(), recovery.parameters()))
    optimizer_gs = optim.Adam(chain(generator.parameters(), supervisor.parameters()))
    optimizer_d = optim.Adam(discriminator.parameters())

    embedder.train()
    generator.train()
    supervisor.train()
    recovery.train()
    discriminator.train()

    print("Start Embedding Network Training")
    
    '训练ae'
    for step in range(1, max_steps + 1):
        x = batch_generator(dataset, batch_size).to(device)    # [128, 24, 6]

        h = embedder(x)
        x_tilde = recovery(h)

        loss_e_t0 = _loss_e_t0(x_tilde, x)
        loss_e_0 = _loss_e_0(loss_e_t0)
        optimizer_er.zero_grad()
        loss_e_0.backward()
        optimizer_er.step()

        if step % 1000 == 0:
            print(  "step: "+ str(step) + "/" + str(max_steps) + ", loss_e: " + str(np.round(np.sqrt(loss_e_t0.item()), 4)) )
            
    print("Finish Embedding Network Training")

    
    
    
    
    '训练完ae之后再训练生成器和监督器 应该是让其有初步的性能'
    print("Start Training with Supervised Loss Only")
    
    for step in range(1, max_steps + 1):
        x = batch_generator(dataset, batch_size).to(device)
        z = torch.randn(batch_size, x.size(1), x.size(2)).to(device)

        h = embedder(x)
        h_hat_supervise = supervisor(h)

        loss_s = _loss_s(h_hat_supervise, h)
        optimizer_gs.zero_grad()
        loss_s.backward()
        optimizer_gs.step()

        if step % 1000 == 0:
            print( "step: " + str(step) + "/" + str(max_steps) + ", loss_s: " + str(np.round(np.sqrt(loss_s.item()), 4)) )
            
    print("Finish Training with Supervised Loss Only")

    
    
    '多个组件联合训练 有些复杂 主要是有个监督器 还不知其作用'
    print("Start Joint Training")
    
    for step in range(1, max_steps + 1):
        
        '先优化生成器+监督器  再优化ae  最后优化判别器'
        for _ in range(2):
            
            x = batch_generator(dataset, batch_size).to(device)
            z = torch.randn(batch_size, x.size(1), x.size(2)).to(device)  # [128, 24, 6]
            
            h = embedder(x)
            e_hat = generator(z)
            
            h_hat = supervisor(e_hat)   # 监督器对两个隐码做处理
            h_hat_supervise = supervisor(h)
            
            x_hat = recovery(h_hat)
            y_fake = discriminator(h_hat)
            y_fake_e = discriminator(e_hat)

            loss_s = _loss_s(h_hat_supervise, h)   # 监督损失
            loss_g_u = _loss_g_u(y_fake)        # 两个判别损失
            loss_g_u_e = _loss_g_u_e(y_fake_e)
            loss_g_v = _loss_g_v(x_hat, x)       # 生成器的重构损失
            loss_g = _loss_g(loss_g_u, loss_g_u_e, loss_s, loss_g_v)  # 多个损失一起优化生成器和监督器
            
            optimizer_gs.zero_grad()         # 化生成器和监督器
            loss_g.backward()
            optimizer_gs.step()

            h = embedder(x)
            x_tilde = recovery(h)
            h_hat_supervise = supervisor(h)

            loss_e_t0 = _loss_e_t0(x_tilde, x)   # 编码器的重构损失
            loss_e_0 = _loss_e_0(loss_e_t0)
            loss_s = _loss_s(h_hat_supervise, h)
            loss_e = _loss_e(loss_e_0, loss_s)
            
            optimizer_er.zero_grad()        # 优化ae
            loss_e.backward()
            optimizer_er.step()

        '最后优化判别器'
        x = batch_generator(dataset, batch_size).to(device)
        z = torch.randn(batch_size, x.size(1), x.size(2)).to(device)

        h = embedder(x)
        e_hat = generator(z)
        h_hat = supervisor(e_hat)
        y_fake = discriminator(h_hat)
        y_real = discriminator(h)
        y_fake_e = discriminator(e_hat)

        loss_d = _loss_d(y_real, y_fake, y_fake_e)  # 判别器损失
        
        if loss_d.item() > 0.15:
            optimizer_d.zero_grad()        # 优化判别器
            loss_d.backward()
            optimizer_d.step()

        if step % 1000 == 0:
            print( "step: " + str(step) + "/" + str(max_steps) + ", loss_d: " + str(np.round(loss_d.item(), 4))
                + ", loss_g_u: " + str(np.round(loss_g_u.item(), 4))
                + ", loss_g_v: " + str(np.round(loss_g_v.item(), 4))
                + ", loss_s: " + str(np.round(np.sqrt(loss_s.item()), 4))
                + ", loss_e_t0: " + str(np.round(np.sqrt(loss_e_t0.item()), 4))  )
            
    print("Finish Joint Training")
    
    
    
    
    
    
'dataset是真实数据经过TimeDataset处理后的 有了时间步 也是一个列表' 
'生成的数据是一个三维矩阵 改成一样的列表显示 方便一起可视化 即下面的generated_data  3662个元素  每个是(24, 6)的矩阵'
def visualize(dataset, device, generator, supervisor, recovery):
    # generator.load_state_dict(torch.load("generator.pt"))
    # supervisor.load_state_dict(torch.load("supervisor.pt"))
    # recovery.load_state_dict(torch.load("recovery.pt"))
    
    seq_len = dataset[0].shape[0]    # 24
    input_size = dataset[0].shape[1]  # 6
    dataset_size = len(dataset)     # 3662

    generator.eval()
    supervisor.eval()
    recovery.eval()

    with torch.no_grad():
        z = torch.randn(dataset_size, seq_len, input_size).to(device)  # [3662, 24, 6]
        e_hat = generator(z)          
        h_hat = supervisor(e_hat)
        x_hat = recovery(h_hat)         # 生成的序列数据 (3662, 24, 6)
        
    #print('生成的序列数据',x_hat.shape)     
    
    generated_data_curr = x_hat.cpu().numpy()
    #print('生成数据转换成np格式',generated_data_curr.shape)  # (3662, 24, 6)
    
    generated_data = list()
    
    '取出生成数据里每一个时间步的样本'
    for i in range(dataset_size):
        temp = generated_data_curr[i, :, :]
        generated_data.append(temp)
        
    #print('生成数据放到列表里',len(generated_data),generated_data[0].shape)   # 3662 (24, 6)
    
    visualization(dataset, generated_data, "pca")
    visualization(dataset, generated_data, "tsne")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="datasets/stock_data.csv")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=10000)     
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--save-model", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '数据归一化 添加时间步'
    dataset = TimeDataset(args.data_path, args.seq_len)        # (3662, 24, 6)
    
    input_size = dataset[0].shape[1]  # 6
    
    '训练时隐藏维度最好大于原数据维度 这样效果好'
    hidden_size = 24     # 原来24     
    num_layers = 3

    '原版的参数设置'
#     embedder = Net(input_size, hidden_size, hidden_size, num_layers).to(device)
#     recovery = Net(hidden_size, hidden_size, input_size, num_layers).to(device)
#     generator = Net(input_size, hidden_size, hidden_size, num_layers).to(device)
#     supervisor = Net(hidden_size, hidden_size, hidden_size, num_layers - 1).to(device)
#     discriminator = Net(hidden_size, hidden_size, 1, num_layers, activation_fn=None).to(device)

    '把隐藏层参数减小 主要是编码器的输出维度要小于数据维度 其他的应该如全连接维度应大一些'
    embedder = Net(input_size, hidden_size, 3, num_layers).to(device)
    recovery = Net(3, hidden_size, 6, num_layers).to(device)
    generator = Net(input_size, hidden_size, 3, num_layers).to(device)
    supervisor = Net(3, hidden_size, 3, num_layers - 1).to(device)
    discriminator = Net(3, hidden_size, 1, num_layers, activation_fn=None).to(device)
    
    
    
    train(
        args.batch_size,
        args.max_steps,
        dataset,
        device,
        embedder,
        generator,
        supervisor,
        recovery,
        discriminator,
        args.gamma,  )

    visualize(dataset, device, generator, supervisor, recovery)  # 生成器+解码器的无监督合成

    if args.save_model:
        torch.save(embedder.state_dict(), "./ckpt/embedder.pt")
        torch.save(generator.state_dict(), "./ckpt/generator.pt")
        torch.save(supervisor.state_dict(), "./ckpt/supervisor.pt")
        torch.save(recovery.state_dict(), "./ckpt/recovery.pt")
        torch.save(discriminator.state_dict(), "./ckpt/discriminator.pt")


if __name__ == "__main__":
    main()
