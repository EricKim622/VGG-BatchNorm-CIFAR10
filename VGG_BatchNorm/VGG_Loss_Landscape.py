import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0, 1, 2, 3]
num_workers = 0  # <--- 改成 0，解决 Windows 报错
batch_size = 128

# add our package dir to path
module_path = os.path.dirname(os.path.abspath(__file__))
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Make sure you are using the right device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# Initialize your data loader
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)

# This function is used to calculate the accuracy
def get_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, pred = torch.max(outputs.data, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# Set random seeds
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Training function
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [0.0] * epochs_n
    train_accuracy_curve = [0.0] * epochs_n
    val_accuracy_curve = [0.0] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []
        grad = []
        total_train_loss = 0.0

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)

            loss_list.append(loss.item())
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad)
        learning_curve[epoch] = total_train_loss / batches_n

        # Validation
        train_acc = get_accuracy(model, train_loader)
        val_acc = get_accuracy(model, val_loader)
        train_accuracy_curve[epoch] = train_acc
        val_accuracy_curve[epoch] = val_acc

        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))
        axes[0].plot(learning_curve[:epoch + 1])
        axes[0].set_title("Train Loss")
        axes[1].plot(val_accuracy_curve[:epoch + 1])
        axes[1].set_title("Val Acc")
        plt.savefig(os.path.join(figures_path, f"train_curve_epoch_{epoch}.png"))
        plt.close()

    return losses_list, grads

# -----------------------------------------------------------------------------
# 下面是主程序入口，Windows 必须加这个！
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    import matplotlib as mpl

    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from torch import nn
    import numpy as np
    import torch
    import os
    import random

    # 关键修复：Windows不显示进度条bug + 加速
    from tqdm import tqdm
    from IPython import display

    from models.vgg import VGG_A
    from models.vgg import VGG_A_BatchNorm
    from data.loaders import get_cifar_loader

    # ================== 固定配置 ==================
    num_workers = 0  # Windows必须0
    batch_size = 128
    device = torch.device("cpu")
    print("使用设备:", device)

    # ================== 路径 ==================
    module_path = os.path.dirname(os.path.abspath(__file__))
    figures_path = os.path.join(module_path, 'reports', 'figures')
    models_path = os.path.join(module_path, 'reports', 'models')
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    # ================== 数据加载 ==================
    train_loader = get_cifar_loader(train=True, num_workers=0)
    val_loader = get_cifar_loader(train=False, num_workers=0)


    # ================== 工具函数 ==================
    def get_accuracy(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, pred = torch.max(outputs.data, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total


    def set_random_seeds(seed_value=2020):
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        random.seed(seed_value)


    # ================== 训练函数 ==================
    def train(model, optimizer, criterion, epochs_n=20):
        model.to(device)
        learning_curve = []
        val_acc_curve = []
        losses_list = []

        for epoch in range(epochs_n):
            model.train()
            total_loss = 0
            loss_list_epoch = []

            # 训练一个epoch
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs_n}"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                loss_list_epoch.append(loss.item())

            # 记录
            avg_loss = total_loss / len(train_loader)
            learning_curve.append(avg_loss)
            losses_list.append(loss_list_epoch)

            # 验证
            val_acc = get_accuracy(model, val_loader)
            val_acc_curve.append(val_acc)
            print(f"Epoch {epoch + 1:2d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        return losses_list, val_acc_curve


    # ================== 主程序（Windows必须加） ==================
    if __name__ == '__main__':
        set_random_seeds(2020)
        epo = 20
        lr = 0.001

        # ========== 训练无BN ==========
        print("\n===== 训练 VGG_A without BN =====")
        model_no_bn = VGG_A()
        opt_no_bn = torch.optim.Adam(model_no_bn.parameters(), lr=lr)
        loss_no_bn, val_no_bn = train(model_no_bn, opt_no_bn, nn.CrossEntropyLoss(), epo)

        # ========== 训练带BN ==========
        print("\n===== 训练 VGG_A with BN =====")
        model_bn = VGG_A_BatchNorm()
        opt_bn = torch.optim.Adam(model_bn.parameters(), lr=lr)
        loss_bn, val_bn = train(model_bn, opt_bn, nn.CrossEntropyLoss(), epo)


        # ================== 画损失区间图 ==================
        def get_min_max(losses):
            min_curve = []
            max_curve = []
            for i in range(epo):
                step = [ls[i] for ls in losses if i < len(ls)]
                if step:
                    min_curve.append(min(step))
                    max_curve.append(max(step))
            return min_curve, max_curve


        min_no, max_no = get_min_max(loss_no_bn)
        min_bn, max_bn = get_min_max(loss_bn)


        def plot_landscape(min_curve, max_curve, title, save_path):
            plt.figure(figsize=(10, 5))
            plt.fill_between(range(len(min_curve)), min_curve, max_curve, alpha=0.5)
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel("Loss Range")
            plt.savefig(save_path)
            plt.close()


        plot_landscape(min_no, max_no, "VGG-A without BN", os.path.join(figures_path, "loss_no_bn.png"))
        plot_landscape(min_bn, max_bn, "VGG-A with BN", os.path.join(figures_path, "loss_bn.png"))

        print("\n✅ 全部跑完！图已保存到：", figures_path)